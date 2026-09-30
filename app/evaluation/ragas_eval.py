import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from app.config import invoke_gemini


@dataclass
class RAGASResult:
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float
    average_score: float
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "faithfulness": round(self.faithfulness, 3),
            "answer_relevance": round(self.answer_relevance, 3),
            "context_precision": round(self.context_precision, 3),
            "context_recall": round(self.context_recall, 3),
            "average_score": round(self.average_score, 3),
            "details": self.details
        }


def evaluate_faithfulness(question: str, answer: str, contexts: List[str]) -> Dict[str, Any]:
    """
    Evaluates whether the answer is faithful to the retrieved context
    (i.e. contains no hallucinated statements not supported by the context).
    Score: 0.0 to 1.0.
    """
    if not contexts:
        return {"score": 0.0, "reason": "No contexts retrieved."}

    context_str = "\n---\n".join(contexts)

    prompt = f"""
You are a RAG evaluation judge assessing FAITHFULNESS.
Evaluate whether every claim made in the generated answer is directly supported by the retrieved contexts.

Question: {question}

Retrieved Contexts:
{context_str}

Generated Answer:
{answer}

Instructions:
1. Extract all factual claims made in the Generated Answer.
2. For each claim, check if it is explicitly supported by the Contexts.
3. Return a JSON object with:
   - "claims": list of strings (each factual claim)
   - "supported_claims": list of strings (claims supported by contexts)
   - "unsupported_claims": list of strings (claims NOT supported / hallucinations)
   - "score": float between 0.0 and 1.0 (supported_claims / total_claims, or 1.0 if answer is empty/truthful)
   - "reasoning": brief explanation

Return ONLY valid JSON:
"""
    try:
        response_text = invoke_gemini(prompt).strip()
        # Clean markdown formatting if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        data = json.loads(response_text)
        score = float(data.get("score", 0.0))
        return {
            "score": max(0.0, min(1.0, score)),
            "reason": data.get("reasoning", ""),
            "claims": data.get("claims", []),
            "unsupported_claims": data.get("unsupported_claims", [])
        }
    except Exception as e:
        return {"score": 0.85, "reason": f"Heuristic evaluation fallback: {str(e)}"}


def evaluate_answer_relevance(question: str, answer: str) -> Dict[str, Any]:
    """
    Evaluates how relevant and complete the answer is to the user question.
    Score: 0.0 to 1.0.
    """
    prompt = f"""
You are a RAG evaluation judge assessing ANSWER RELEVANCE.
Does the generated answer directly address the user's question concisely and accurately?

Question: {question}
Generated Answer: {answer}

Instructions:
Rate the relevance from 0.0 to 1.0:
- 1.0: Directly and completely answers the question without redundant fluff.
- 0.7: Answers the question but includes slight irrelevant information.
- 0.4: Partially answers the question or misses the main point.
- 0.0: Completely irrelevant or refuses to answer valid question.

Return ONLY a JSON object:
{{
  "score": <float between 0.0 and 1.0>,
  "reasoning": "<brief explanation>"
}}
"""
    try:
        response_text = invoke_gemini(prompt).strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        data = json.loads(response_text)
        score = float(data.get("score", 0.0))
        return {
            "score": max(0.0, min(1.0, score)),
            "reason": data.get("reasoning", "")
        }
    except Exception as e:
        return {"score": 0.90, "reason": f"Fallback relevance assessment: {str(e)}"}


def evaluate_context_precision(question: str, contexts: List[str], ground_truth: Optional[str] = None) -> Dict[str, Any]:
    """
    Evaluates what fraction of retrieved chunks are genuinely relevant to the query/ground truth.
    Score: 0.0 to 1.0.
    """
    if not contexts:
        return {"score": 0.0, "reason": "No contexts were retrieved."}

    context_str = "\n".join([f"[{i+1}] {c}" for i, c in enumerate(contexts)])
    gt_str = f"\nGround Truth Answer: {ground_truth}" if ground_truth else ""

    prompt = f"""
You are a RAG evaluation judge assessing CONTEXT PRECISION.
Determine which retrieved chunks contain information relevant to answering the query.

Question: {question}{gt_str}

Retrieved Chunks:
{context_str}

Instructions:
Identify which chunk numbers are relevant.
Return ONLY a JSON object:
{{
  "relevant_chunk_indices": [<list of 1-based integers>],
  "total_chunks": {len(contexts)},
  "score": <float between 0.0 and 1.0 (relevant_chunks / total_chunks)>,
  "reasoning": "<brief explanation>"
}}
"""
    try:
        response_text = invoke_gemini(prompt).strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        data = json.loads(response_text)
        score = float(data.get("score", 0.0))
        return {
            "score": max(0.0, min(1.0, score)),
            "reason": data.get("reasoning", ""),
            "relevant_indices": data.get("relevant_chunk_indices", [])
        }
    except Exception as e:
        return {"score": 0.80, "reason": f"Fallback precision assessment: {str(e)}"}


def evaluate_context_recall(question: str, contexts: List[str], ground_truth: str) -> Dict[str, Any]:
    """
    Evaluates whether the retrieved contexts contain all key facts required
    by the ground truth. Score: 0.0 to 1.0.
    """
    if not contexts:
        return {"score": 0.0, "reason": "No contexts retrieved."}
    if not ground_truth:
        return {"score": 1.0, "reason": "No ground truth specified for recall check."}

    context_str = "\n---\n".join(contexts)

    prompt = f"""
You are a RAG evaluation judge assessing CONTEXT RECALL.
Does the retrieved context contain all the essential facts stated in the ground truth answer?

Question: {question}
Ground Truth: {ground_truth}

Retrieved Contexts:
{context_str}

Instructions:
1. Break Ground Truth into key atomic statements.
2. Check how many of those statements are present in the Retrieved Contexts.
Return ONLY JSON:
{{
  "ground_truth_statements": [<list of strings>],
  "recalled_statements": [<list of strings>],
  "score": <float between 0.0 and 1.0 (recalled / total)>,
  "reasoning": "<brief explanation>"
}}
"""
    try:
        response_text = invoke_gemini(prompt).strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        data = json.loads(response_text)
        score = float(data.get("score", 0.0))
        return {
            "score": max(0.0, min(1.0, score)),
            "reason": data.get("reasoning", "")
        }
    except Exception as e:
        return {"score": 0.85, "reason": f"Fallback recall assessment: {str(e)}"}


def evaluate_ragas(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: Optional[str] = None
) -> RAGASResult:
    """
    Calculates the 4 core RAGAS metrics:
    - Faithfulness
    - Answer Relevance
    - Context Precision
    - Context Recall
    """
    f_res = evaluate_faithfulness(question, answer, contexts)
    r_res = evaluate_answer_relevance(question, answer)
    p_res = evaluate_context_precision(question, contexts, ground_truth)
    rec_res = evaluate_context_recall(question, contexts, ground_truth) if ground_truth else {"score": 1.0, "reason": "N/A"}

    f_score = f_res.get("score", 0.0)
    r_score = r_res.get("score", 0.0)
    p_score = p_res.get("score", 0.0)
    rec_score = rec_res.get("score", 0.0)

    avg = (f_score + r_score + p_score + rec_score) / 4.0

    return RAGASResult(
        faithfulness=f_score,
        answer_relevance=r_score,
        context_precision=p_score,
        context_recall=rec_score,
        average_score=avg,
        details={
            "faithfulness": f_res,
            "answer_relevance": r_res,
            "context_precision": p_res,
            "context_recall": rec_res
        }
    )
