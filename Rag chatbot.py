from __future__ import annotations
import re
from typing import Optional

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    ST_OK = True
except ImportError:
    ST_OK = False

try:
    import faiss
    FAISS_OK = True
except ImportError:
    FAISS_OK = False


# ── system prompt 
SYSTEM_PROMPT = """You are a senior business analyst AI assistant.

Answer ONLY using the provided dataset insights and model outputs.
Every answer MUST include:
- Specific numbers from the context
- Key drivers identified in the analysis
- Clear business reasoning
- Actionable recommendation

Do NOT give generic answers. If information is not in the context, say so clearly."""


# ── chunker 
def _chunk(text: str, size: int = 180, overlap: int = 40) -> list[str]:
    """Split context into overlapping word-level chunks."""
    words = text.split()
    chunks, step = [], size - overlap
    for i in range(0, max(1, len(words) - overlap), step):
        chunk = " ".join(words[i: i + size])
        if len(chunk.strip()) > 20:
            chunks.append(chunk)
    return chunks or [text]


# ── FAISS index 
class VectorIndex:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model   = None
        self.index   = None
        self.chunks  = []
        self._name   = model_name
        self._ready  = False

    def _load(self):
        if not self._ready and ST_OK:
            try:
                self.model  = SentenceTransformer(self._name)
                self._ready = True
            except Exception:
                self._ready = False

    def build(self, context: str):
        self.chunks = _chunk(context)
        if not self.chunks:
            return
        self._load()
        if not self._ready or not FAISS_OK:
            return   # keyword fallback will be used

        emb = self.model.encode(self.chunks, show_progress_bar=False,
                                batch_size=64).astype("float32")
        faiss.normalize_L2(emb)
        dim = emb.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(emb)

    def retrieve(self, query: str, k: int = 6) -> list[str]:
        if not self.chunks:
            return []
        if self.index is None or not self._ready:
            return self._keyword_retrieve(query, k)

        q = self.model.encode([query], show_progress_bar=False).astype("float32")
        faiss.normalize_L2(q)
        _, idx = self.index.search(q, min(k, len(self.chunks)))
        return [self.chunks[i] for i in idx[0] if i < len(self.chunks)]

    def _keyword_retrieve(self, query: str, k: int) -> list[str]:
        qw = set(query.lower().split())
        scored = sorted(
            self.chunks,
            key=lambda c: len(qw & set(c.lower().split())),
            reverse=True,
        )
        return scored[:k] or self.chunks[:k]


# ── rule-based answer generator 
def _rule_answer(query: str, chunks: list[str], full: str) -> str:
    """
    Generate a fully grounded answer from context chunks.
    Uses regex + pattern matching to pull specific numbers.
    """
    q = query.lower()
    ctx = "\n".join(chunks) if chunks else full

    def _find(pattern, text=ctx):
        m = re.search(pattern, text, re.IGNORECASE)
        return m.group(1).rstrip(".") if m else None

    # ── feature importance 
    if any(w in q for w in ["feature","important","driver","impact","factor","weight","matter","predict"]):
        top_match = _find(r"Top 3 predictive features?:\s*([^\n]+)")
        imp_matches = re.findall(r"'([^']+)':\s*([\d\.]+)\s*\(([\d\.]+)%\)", ctx)

        if imp_matches:
            top3 = sorted(imp_matches, key=lambda x: float(x[1]), reverse=True)[:3]
            feat_str = "; ".join(f"**{f}** ({p}%)" for f, _, p in top3)
            return (
                f"The top predictive features are: {feat_str}. "
                f"These variables carry the most weight in the model's decision logic. "
                f"**Business implication:** Controlling or optimizing these features will have "
                f"the highest leverage on outcomes — prioritize them in strategy planning."
            )
        if top_match:
            return (
                f"The most important features are: **{top_match}**. "
                f"These drive the majority of predictive power. "
                f"Business strategy should focus resources on understanding and optimizing these variables."
            )
        return "Feature importance is shown in the Modeling tab. Run model training to see ranked feature contributions."

    # ── model performance 
    if any(w in q for w in ["model","accuracy","performance","r2","r²","rmse","f1","score","reliable","best model"]):
        best_m  = _find(r"Best model:\s*([^\.\n]+)")
        acc     = _find(r"Accuracy \(%\)=([\d\.]+)")
        f1      = _find(r"F1-Score \(%\)=([\d\.]+)")
        r2      = _find(r"R² \(%\)=([\d\.]+)")
        gap     = _find(r"Overfit Gap=([\d\.\-]+)")

        if best_m and (acc or r2):
            metric_str = f"accuracy {acc}%, F1 {f1}%" if acc else f"R²={r2}%"
            gap_str = (f" Train-test gap of {gap}% detected — slight overfitting risk."
                       if gap and float(gap) > 10 else " Generalization is healthy.")
            return (
                f"The best-performing model is **{best_m}** with {metric_str}.{gap_str} "
                f"{'This accuracy level is suitable for automated decision support.' if acc and float(acc) >= 75 else 'Performance needs improvement — consider adding features or collecting more data.'}"
            )
        return "Model performance details are in the Modeling tab. Train the model to see accuracy, F1, and R² metrics."

    # ── growth / trend 
    if any(w in q for w in ["growth","trend","increase","decrease","decline","drop","rise","change","yoy","year"]):
        yoy   = _find(r"Year-over-year.*?([+-]?[\d\.]+)%\s*change")
        col   = _find(r"change in '([^']+)'")
        latest= _find(r"to (\d{4})\.")
        prev  = _find(r"from (\d{4}) to")
        trend_info = _find(r"trended (up|down)[^\.]*\(([+-]?[\d\.]+)%")

        if yoy and col:
            direction = "grew" if float(yoy) > 0 else "declined"
            return (
                f"**`{col}` {direction} {abs(float(yoy)):.1f}% year-over-year** "
                f"(from {prev or 'prior year'} to {latest or 'latest year'}). "
                f"{'This positive trajectory suggests the current strategy is working — continue investing in top segments.' if float(yoy) > 0 else 'This decline requires investigation. Review segment-level data, pricing strategy, and competitive dynamics.'}"
            )
        if trend_info:
            return f"Latest trend: {trend_info}. Check the EDA tab for detailed period-by-period breakdown."
        return "No time-series data detected. Growth analysis requires a date column in the dataset."

    # ── segment / category 
    if any(w in q for w in ["segment","region","category","top","best","worst","performing","breakdown"]):
        top_seg   = _find(r"Top segment: '([^']+)'")
        top_col   = _find(r"in '([^']+)'\s*=")
        top_share = _find(r"=\s*([\d\.]+)%\s*of total")
        bottom_info = re.findall(r"Underperforming[^:]+:\s*([^\-]+)", ctx)

        parts = []
        if top_seg and top_share:
            parts.append(
                f"**Top performing segment:** '{top_seg}' in `{top_col}` contributes "
                f"{top_share}% of total — it is the primary revenue driver."
            )
        if bottom_info:
            parts.append(
                f"**Underperformers identified:** {bottom_info[0].strip()[:120]}. "
                f"Targeted action here offers the highest incremental gain."
            )
        if parts:
            return " ".join(parts) + " See EDA and Decisions tabs for full segment breakdowns."
        return "Segment analysis is available in the EDA tab — categorical breakdowns show performance by region, category, and other dimensions."

    # ── missing / data quality 
    if any(w in q for w in ["missing","null","empty","quality","complete","gap","incomplete"]):
        miss_info = re.findall(r"Column '([^']+)' is missing ([\d\.]+)% of values", ctx)
        no_miss   = "No significant missing values" in ctx or "Dataset is complete" in ctx

        if no_miss:
            return "✅ The dataset has no significant missing values. It is clean and ready for modeling."
        if miss_info:
            parts = [f"**`{col}`**: {pct}% missing" for col, pct in miss_info[:3]]
            return (
                "Data quality issues detected: " + "; ".join(parts) + ". "
                "High missingness in key columns reduces model reliability. "
                "Implement data capture validation in source systems to resolve this."
            )
        dup_match = _find(r"(\d[\d,]+) duplicate records")
        if dup_match:
            return f"**{dup_match} duplicate records** detected. This inflates metrics and must be resolved upstream."
        return "Dataset quality analysis is in the Overview tab. Run the EDA for a detailed missing-value breakdown."

    # ── risk 
    if any(w in q for w in ["risk","alert","danger","warning","concern","threat"]):
        risks = re.findall(r"\[(\w+)\]\s*([^\:]+):\s*([^\n]{30,120})", ctx)
        risk_lines = [r for r in risks if r[0] in ("High","Critical","Medium")][:4]
        if risk_lines:
            parts = [f"**[{sev}] {title}** — {body[:100]}" for sev, title, body in risk_lines]
            return "Active risk alerts from the analysis:\n\n" + "\n\n".join(parts) + "\n\nSee Decisions tab for full details."
        return "Risk analysis is available in the Decisions tab after running the full pipeline."

    # ── recommendation 
    if any(w in q for w in ["recommend","suggest","should","action","what to do","next step","strategy","improve","optimize"]):
        recs = re.findall(r"\[High\]\s*([^\:]+):\s*([^\n]{40,200})", ctx)[:3]
        if recs:
            parts = [f"**{title}:** {body[:120]}" for title, body in recs]
            return "Top business recommendations from the analysis:\n\n" + "\n\n".join(parts) + "\n\nFor full details, see the Decisions tab."
        return "Business recommendations are generated in the Decisions tab after data processing and modeling."

    # ── rows / size / shape 
    if any(w in q for w in ["row","size","record","large","how many","dataset size","shape"]):
        orig  = _find(r"Original dataset:\s*([\d,]+)\s*rows")
        cols  = _find(r"rows\s*×\s*(\d+)\s*columns")
        agg_r = _find(r"After aggregation:\s*([\d,]+)\s*summary rows")
        red   = _find(r"reduction:\s*([\d\.]+)%")

        if orig:
            return (
                f"The dataset contains **{orig} rows × {cols or '?'} columns**. "
                f"After aggregation: **{agg_r} summary rows** ({red}% reduction). "
                f"The system processed the full dataset efficiently using group-by summarization "
                f"before analysis — ensuring fast performance without losing statistical integrity."
            )
        return "Dataset size is shown on the Overview tab."

    # ── correlation 
    if any(w in q for w in ["correlation","relate","relationship","together","move","linear"]):
        corr_matches = re.findall(r"'([^']+)' and '([^']+)' have correlation\s*([\-\d\.]+)", ctx)
        if corr_matches:
            top = sorted(corr_matches, key=lambda x: abs(float(x[2].rstrip("."))), reverse=True)[:3]
            parts = []
            for c1, c2, val in top:
                v = float(val.rstrip("."))
                direction = "positive" if v > 0 else "negative"
                parts.append(f"**`{c1}`** ↔ **`{c2}`**: r={v:.3f} ({direction})")
            return (
                "Key correlations found in the dataset: " + "; ".join(parts) + ". "
                "Strong correlations indicate features that move together — "
                "these can be used for joint optimization or flagged for multicollinearity checks."
            )
        return "Correlation analysis is in the EDA tab. Run EDA to see the correlation heatmap."

    # ── generic grounded fallback 
    numbers = re.findall(r"[\w\s]+?(?:=|:)\s*([\d\.\,]+(?:%|k|M)?)", ctx)[:5]
    if numbers:
        return (
            f"Based on the dataset analysis, here are key statistics: "
            + "; ".join(numbers[:4]) + ". "
            "For specific questions try asking about: features, model accuracy, "
            "growth trends, segment performance, risks, or recommendations."
        )

    return (
        "I can answer questions grounded in your dataset. Try asking:\n"
        "• 'Which features impact predictions most?'\n"
        "• 'What is the model accuracy?'\n"
        "• 'Why did [metric] change?'\n"
        "• 'What are the top risks?'\n"
        "• 'What should the business do?'"
    )


# ── main chatbot class 
class BusinessCopilotChatbot:
    """
    RAG chatbot grounded exclusively in aggregated context.
    Never accesses raw data directly.
    """

    def __init__(self):
        self.index        = VectorIndex()
        self.full_context = ""
        self.is_ready     = False
        self.history      = []   # list of (role, message)

    def build(self, *context_parts: str):
        """Combine all context pieces and build the FAISS index."""
        self.full_context = "\n\n".join(p for p in context_parts if p)
        self.index.build(self.full_context)
        self.is_ready = True

    def chat(self, question: str, api_key: Optional[str] = None) -> str:
        if not self.is_ready:
            return "⚠️ Please complete data processing before using the chatbot."

        chunks = self.index.retrieve(question, k=7)
        answer = _rule_answer(question, chunks, self.full_context)

        # Optionally enhance via Anthropic API
        if api_key:
            try:
                import requests
                context_str = "\n\n".join(chunks) or self.full_context[:2500]
                payload = {
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 500,
                    "system": SYSTEM_PROMPT,
                    "messages": [{"role": "user",
                                  "content": f"CONTEXT:\n{context_str}\n\nQUESTION: {question}"}],
                }
                resp = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"x-api-key": api_key,
                             "anthropic-version": "2023-06-01",
                             "content-type": "application/json"},
                    json=payload, timeout=15,
                )
                if resp.status_code == 200:
                    answer = resp.json()["content"][0]["text"]
            except Exception:
                pass   # fall through to rule-based answer

        self.history.append(("user", question))
        self.history.append(("bot", answer))
        return answer

    def clear(self):
        self.history = []

    @staticmethod
    def suggested_questions() -> list[str]:
        return [
            "Which features impact predictions most?",
            "What is the best model and its accuracy?",
            "Why did the primary metric change year-over-year?",
            "Which segments are underperforming?",
            "What are the top business risks?",
            "What actions should the business take?",
            "Are there any data quality issues?",
            "What correlations exist in the dataset?",
        ]
        
        

if __name__ == "__main__":

    # 🔹 Dummy context (IMPORTANT)
    context = """
    Original dataset: 5,000 rows × 5 columns.
    Year-over-year performance: +12.5% change in 'sales' from 2022 to 2023.
    Top segment: 'Electronics' in 'category' = 62.0% of total sales.

    Best model: Random Forest.
    Accuracy (%)=82.5
    Overfit Gap=5.2

    [High] Increase Sales in South Region: South region is underperforming by 18%.
    """

    # 🔹 Create chatbot
    bot = BusinessCopilotChatbot()

    # 🔹 Build context
    bot.build(context)

    print("\n🤖 CHATBOT READY!\n")

    # 🔹 Loop for questions
    while True:
        question = input("Ask question (type 'exit' to stop): ")

        if question.lower() == "exit":
            print("👋 Exiting chatbot...")
            break

        answer = bot.chat(question)

        print("\n💡 Answer:")
        print(answer)
        print("-" * 50)        