import json
from pathlib import Path
from typing import List, Dict, Any, Optional

DATASET_PATH = Path(__file__).resolve().parent / "data" / "benchmark_dataset.json"


def load_evaluation_dataset(category: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Loads benchmark evaluation questions.
    Can filter by category: 'RAG', 'CALCULATOR', 'WEB', 'HYBRID', 'GUARDRAIL_INJECTION', 'GUARDRAIL_PII'.
    """
    if not DATASET_PATH.exists():
        return []

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if category:
        cat_upper = category.upper()
        return [item for item in data if item.get("category", "").upper() == cat_upper]

    return data


def get_dataset_stats() -> Dict[str, Any]:
    """
    Returns summary statistics of the evaluation dataset.
    """
    dataset = load_evaluation_dataset()
    categories = {}
    for item in dataset:
        cat = item.get("category", "OTHER")
        categories[cat] = categories.get(cat, 0) + 1

    return {
        "total_items": len(dataset),
        "categories": categories
    }
