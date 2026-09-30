import time
import uuid
import contextvars
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager
from app.observability.cost_tracker import get_token_usage_summary


@dataclass
class Span:
    name: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    parent_span_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_ms: float = 0.0
    status: str = "RUNNING"  # "SUCCESS", "FAILED"
    input_data: Optional[Any] = None
    output_data: Optional[Any] = None
    tokens: int = 0
    cost_usd: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def finish(self, status: str = "SUCCESS", output_data: Optional[Any] = None):
        self.end_time = time.time()
        self.duration_ms = round((self.end_time - self.start_time) * 1000.0, 2)
        self.status = status
        if output_data is not None:
            self.output_data = str(output_data)[:500]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "span_id": self.span_id,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "input_data": str(self.input_data)[:300] if self.input_data is not None else None,
            "output_data": str(self.output_data)[:300] if self.output_data is not None else None,
            "tokens": self.tokens,
            "cost_usd": self.cost_usd,
            "metadata": self.metadata
        }


@dataclass
class TraceContext:
    trace_id: str = field(default_factory=lambda: f"trc-{uuid.uuid4().hex[:10]}")
    query: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_duration_ms: float = 0.0
    spans: List[Span] = field(default_factory=list)
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    status: str = "RUNNING"
    guardrails_triggered: List[str] = field(default_factory=list)
    final_answer: str = ""

    def add_span(self, span: Span):
        self.spans.append(span)

    def record_llm_call(
        self,
        prompt: str,
        response: str,
        model: str,
        duration_ms: float,
        usage_metadata: Optional[Dict[str, Any]] = None
    ):
        usage = get_token_usage_summary(prompt, response, usage_metadata)
        self.total_tokens += usage["total_tokens"]
        self.total_cost_usd += usage["cost_usd"]

        span = Span(
            name=f"llm_call:{model}",
            start_time=time.time() - (duration_ms / 1000.0),
            end_time=time.time(),
            duration_ms=round(duration_ms, 2),
            status="SUCCESS",
            input_data=prompt[:200],
            output_data=response[:200],
            tokens=usage["total_tokens"],
            cost_usd=usage["cost_usd"],
            metadata={"model": model, "usage": usage}
        )
        self.spans.append(span)

    def finish(self, final_answer: str = "", status: str = "SUCCESS"):
        self.end_time = time.time()
        self.total_duration_ms = round((self.end_time - self.start_time) * 1000.0, 2)
        self.status = status
        self.final_answer = final_answer
        self.total_cost_usd = round(self.total_cost_usd, 6)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "query": self.query,
            "total_duration_ms": self.total_duration_ms,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "status": self.status,
            "guardrails_triggered": self.guardrails_triggered,
            "spans_count": len(self.spans),
            "spans": [s.to_dict() for s in self.spans],
            "final_answer": self.final_answer[:300] if self.final_answer else ""
        }


import threading

_active_trace: contextvars.ContextVar[Optional[TraceContext]] = contextvars.ContextVar(
    "_active_trace", default=None
)
_registry_lock = threading.Lock()
_active_traces_registry: Dict[str, TraceContext] = {}
_latest_active_trace: Optional[TraceContext] = None


def get_current_trace(trace_id: Optional[str] = None) -> Optional[TraceContext]:
    if trace_id:
        with _registry_lock:
            if trace_id in _active_traces_registry:
                return _active_traces_registry[trace_id]
    ctx_trace = _active_trace.get()
    if ctx_trace is not None:
        return ctx_trace
    return _latest_active_trace


def start_trace(query: str = "") -> TraceContext:
    global _latest_active_trace
    trace = TraceContext(query=query)
    _active_trace.set(trace)
    with _registry_lock:
        _active_traces_registry[trace.trace_id] = trace
        _latest_active_trace = trace
    return trace


def end_trace(trace_id: Optional[str] = None, final_answer: str = "", status: str = "SUCCESS") -> Optional[TraceContext]:
    global _latest_active_trace
    trace = get_current_trace(trace_id)
    if trace:
        trace.finish(final_answer=final_answer, status=status)
        _active_trace.set(None)
        with _registry_lock:
            _active_traces_registry.pop(trace.trace_id, None)
            if _latest_active_trace == trace:
                _latest_active_trace = None
    return trace



@contextmanager
def trace_span(name: str, input_data: Optional[Any] = None, metadata: Optional[Dict[str, Any]] = None):
    """
    Context manager to easily trace an operation span.
    """
    trace = get_current_trace()
    span = Span(
        name=name,
        input_data=input_data,
        metadata=metadata or {}
    )
    if trace:
        trace.add_span(span)

    try:
        yield span
        span.finish(status="SUCCESS")
    except Exception as e:
        span.finish(status="FAILED", output_data=str(e))
        raise
