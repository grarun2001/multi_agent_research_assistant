from typing import List, Dict, Any, Set
from dataclasses import dataclass
from app.supervisor.supervisor_agent import choose_agents
from app.agents.calculator_agent import calculator_agent
from app.agents.web_agent import web_agent
from app.agents.rag_agent import rag_agent


@dataclass
class SupervisorEvalResult:
    query: str
    selected_agents: List[str]
    expected_agents: List[str]
    is_exact_match: bool
    precision: float
    recall: float
    f1_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "selected_agents": self.selected_agents,
            "expected_agents": self.expected_agents,
            "is_exact_match": self.is_exact_match,
            "precision": round(self.precision, 3),
            "recall": round(self.recall, 3),
            "f1_score": round(self.f1_score, 3)
        }


def evaluate_supervisor_routing(query: str, expected_agents: List[str]) -> SupervisorEvalResult:
    """
    Evaluates supervisor agent routing decisions against expected ground truth agents.
    Calculates precision, recall, and F1 score.
    """
    selected = choose_agents(query)
    sel_set: Set[str] = set(selected)
    exp_set: Set[str] = set(expected_agents)

    is_exact = sel_set == exp_set

    # If both are empty (e.g. guardrail blocked queries)
    if not exp_set and not sel_set:
        return SupervisorEvalResult(query, selected, expected_agents, True, 1.0, 1.0, 1.0)

    tp = len(sel_set.intersection(exp_set))
    fp = len(sel_set - exp_set)
    fn = len(exp_set - sel_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return SupervisorEvalResult(
        query=query,
        selected_agents=selected,
        expected_agents=expected_agents,
        is_exact_match=is_exact,
        precision=precision,
        recall=recall,
        f1_score=f1
    )


def evaluate_calculator_accuracy(query: str, expected_num: float) -> Dict[str, Any]:
    """
    Evaluates whether the calculator agent extracts and calculates the correct numerical answer.
    """
    res_str = calculator_agent(query)
    import re
    # Extract numerical result from response
    matches = re.findall(r"[-+]?\d*\.?\d+", res_str.replace("Calculator Result:", ""))
    if not matches:
        return {
            "passed": False,
            "raw_output": res_str,
            "extracted_val": None,
            "expected_val": expected_num,
            "error": "No number extracted"
        }

    try:
        val = float(matches[-1])
        passed = abs(val - expected_num) < 1e-3
        return {
            "passed": passed,
            "raw_output": res_str,
            "extracted_val": val,
            "expected_val": expected_num
        }
    except Exception as e:
        return {
            "passed": False,
            "raw_output": res_str,
            "extracted_val": None,
            "expected_val": expected_num,
            "error": str(e)
        }
