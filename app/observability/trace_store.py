import json
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import deque
from dataclasses import dataclass, asdict

LOGS_DIR = Path(__file__).resolve().parents[2] / "logs"
LOG_FILE = LOGS_DIR / "traces.jsonl"


@dataclass
class TraceRecord:
    trace_id: str
    query: str
    total_duration_ms: float
    total_tokens: int
    total_cost_usd: float
    status: str
    selected_agents: List[str]
    guardrails_triggered: List[str]
    final_answer: str
    spans: List[Dict[str, Any]]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TraceStore:
    def __init__(self, max_in_memory: int = 150):
        self._lock = threading.Lock()
        self._traces: deque[Dict[str, Any]] = deque(maxlen=max_in_memory)
        self._ensure_log_file()
        self._load_recent_logs()

    def _ensure_log_file(self):
        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            if not LOG_FILE.exists():
                LOG_FILE.touch()
        except Exception:
            pass

    def _load_recent_logs(self):
        if not LOG_FILE.exists():
            return
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()[-50:]  # load last 50
                for line in lines:
                    line_clean = line.strip()
                    if line_clean:
                        self._traces.append(json.loads(line_clean))
        except Exception:
            pass

    def add_trace(self, trace_dict: Dict[str, Any]):
        import datetime
        trace_dict["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            self._traces.append(trace_dict)
            try:
                with open(LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(trace_dict) + "\n")
            except Exception:
                pass

    def get_traces(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._traces)[-limit:][::-1]  # reverse to get newest first

    def get_metrics_summary(self) -> Dict[str, Any]:
        with self._lock:
            traces = list(self._traces)

        if not traces:
            return {
                "total_queries": 0,
                "avg_latency_ms": 0.0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "guardrail_violations_count": 0,
                "agent_distribution": {},
                "success_rate": 100.0
            }

        total_q = len(traces)
        total_lat = sum(t.get("total_duration_ms", 0.0) for t in traces)
        total_tok = sum(t.get("total_tokens", 0) for t in traces)
        total_cost = sum(t.get("total_cost_usd", 0.0) for t in traces)
        guardrail_viols = sum(1 for t in traces if t.get("guardrails_triggered"))
        successful = sum(1 for t in traces if t.get("status") == "SUCCESS")

        agent_dist: Dict[str, int] = {}
        for t in traces:
            for ag in t.get("selected_agents", []):
                agent_dist[ag] = agent_dist.get(ag, 0) + 1

        return {
            "total_queries": total_q,
            "avg_latency_ms": round(total_lat / total_q, 1) if total_q > 0 else 0.0,
            "total_tokens": total_tok,
            "total_cost_usd": round(total_cost, 6),
            "guardrail_violations_count": guardrail_viols,
            "agent_distribution": agent_dist,
            "success_rate": round((successful / total_q) * 100.0, 1) if total_q > 0 else 100.0
        }

    def clear(self):
        with self._lock:
            self._traces.clear()
            try:
                if LOG_FILE.exists():
                    LOG_FILE.write_text("")
            except Exception:
                pass


# Global singleton store
trace_store = TraceStore()
