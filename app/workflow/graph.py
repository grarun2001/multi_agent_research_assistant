import time
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END

from app.supervisor.supervisor_agent import choose_agents
from app.agents.rag_agent import rag_agent, RAGOutput
from app.agents.web_agent import web_agent
from app.agents.calculator_agent import calculator_agent
from app.agents.final_compiler import final_compiler

from app.guardrails import validate_input, validate_output
from app.observability import start_trace, end_trace, trace_span, trace_store


class AgentState(TypedDict, total=False):
    query: str
    sanitized_query: str
    selected_agents: List[str]
    agent_outputs: Dict[str, Any]
    results: List[str]
    final_answer: str
    rag_contexts: List[str]
    rag_sources: List[str]
    guardrail_input_passed: bool
    guardrail_violations: List[str]
    guardrail_output_passed: bool
    trace_id: str
    latency_ms: Dict[str, float]
    total_tokens: int
    total_cost_usd: float


def input_guardrail_node(state: AgentState) -> Dict[str, Any]:
    """
    Validates user query for length, character repetition, PII, and prompt injection.
    """
    raw_query = state.get("query", "")
    trace = start_trace(query=raw_query)

    with trace_span("input_guardrails", input_data=raw_query) as span:
        guard_res = validate_input(raw_query)
        span.finish(
            status="SUCCESS" if guard_res.passed else "BLOCKED",
            output_data=guard_res.violations
        )

    if not guard_res.passed:
        trace.guardrails_triggered.extend(guard_res.violations)
        end_trace(final_answer=guard_res.error_message or "Blocked by guardrail", status="BLOCKED")
        # Save blocked trace
        trace_store.add_trace(trace.to_dict())

        return {
            "sanitized_query": guard_res.sanitized_query,
            "guardrail_input_passed": False,
            "guardrail_violations": guard_res.violations,
            "selected_agents": [],
            "agent_outputs": {},
            "results": [],
            "final_answer": guard_res.error_message or "Query blocked by security guardrails.",
            "trace_id": trace.trace_id,
            "total_tokens": trace.total_tokens,
            "total_cost_usd": trace.total_cost_usd,
        }

    if guard_res.violations:
        trace.guardrails_triggered.extend(guard_res.violations)

    return {
        "sanitized_query": guard_res.sanitized_query,
        "guardrail_input_passed": True,
        "guardrail_violations": guard_res.violations,
        "trace_id": trace.trace_id
    }


def route_after_guardrail(state: AgentState) -> str:
    if state.get("guardrail_input_passed", False):
        return "supervisor"
    return END


def supervisor_node(state: AgentState) -> Dict[str, Any]:
    """
    Supervisor selects required agents and executes them.
    Captures latency and structured outputs (contexts, sources) for each agent.
    """
    query = state.get("sanitized_query") or state.get("query", "")

    # 1. Supervisor agent routing
    with trace_span("supervisor_routing", input_data=query) as span:
        agents = choose_agents(query)
        span.finish(output_data=agents)

    print(f"\nSupervisor selected: {', '.join(agents)}")

    results = []
    agent_outputs = {}
    rag_contexts = []
    rag_sources = []
    latencies = {}

    # 2. Execute selected agents with tracing
    if "RAG" in agents:
        t0 = time.time()
        with trace_span("rag_agent", input_data=query) as span:
            rag_out = rag_agent(query)
            span.finish(output_data=str(rag_out)[:200])
        latencies["RAG"] = round((time.time() - t0) * 1000.0, 2)
        results.append(str(rag_out))
        agent_outputs["RAG"] = str(rag_out)

        if hasattr(rag_out, "contexts"):
            rag_contexts = rag_out.contexts
        if hasattr(rag_out, "sources"):
            rag_sources = rag_out.sources

    if "WEB" in agents:
        t0 = time.time()
        with trace_span("web_agent", input_data=query) as span:
            web_out = web_agent(query)
            span.finish(output_data=str(web_out)[:200])
        latencies["WEB"] = round((time.time() - t0) * 1000.0, 2)
        results.append(str(web_out))
        agent_outputs["WEB"] = str(web_out)

    if "CALCULATOR" in agents:
        t0 = time.time()
        with trace_span("calculator_agent", input_data=query) as span:
            calc_out = calculator_agent(query)
            span.finish(output_data=str(calc_out)[:200])
        latencies["CALCULATOR"] = round((time.time() - t0) * 1000.0, 2)
        results.append(str(calc_out))
        agent_outputs["CALCULATOR"] = str(calc_out)

    return {
        "selected_agents": agents,
        "agent_outputs": agent_outputs,
        "results": results,
        "rag_contexts": rag_contexts,
        "rag_sources": rag_sources,
        "latency_ms": latencies
    }


def compiler_node(state: AgentState) -> Dict[str, Any]:
    """
    Compiles agent findings into unified answer, then verifies output guardrails.
    """
    query = state.get("sanitized_query") or state.get("query", "")
    results = state.get("results", [])

    # 1. Final compiler
    with trace_span("final_compiler", input_data={"query": query, "results_count": len(results)}) as span:
        raw_final_answer = final_compiler(query, results)
        span.finish(output_data=raw_final_answer[:200])

    # 2. Output Guardrails
    with trace_span("output_guardrails", input_data=raw_final_answer[:200]) as span:
        out_guard = validate_output(raw_final_answer)
        span.finish(status="SUCCESS" if out_guard.passed else "BLOCKED", output_data=out_guard.violations)

    final_answer = out_guard.sanitized_output

    # 3. Finalize trace
    trace = end_trace(trace_id=state.get("trace_id"), final_answer=final_answer, status="SUCCESS")

    if trace:
        if out_guard.violations:
            trace.guardrails_triggered.extend(out_guard.violations)
        trace_data = trace.to_dict()
        trace_data["selected_agents"] = state.get("selected_agents", [])
        trace_store.add_trace(trace_data)
        tokens = trace.total_tokens
        cost = trace.total_cost_usd
        trace_id = trace.trace_id
    else:
        tokens = 0
        cost = 0.0
        trace_id = state.get("trace_id", "")


    return {
        "final_answer": final_answer,
        "guardrail_output_passed": out_guard.passed,
        "trace_id": trace_id,
        "total_tokens": tokens,
        "total_cost_usd": cost
    }


def build_graph():
    """
    Builds the multi-agent workflow graph:
    START -> input_guardrail -> supervisor -> compiler -> END
    (or input_guardrail -> END if blocked)
    """
    graph = StateGraph(AgentState)

    graph.add_node("input_guardrail", input_guardrail_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("compiler", compiler_node)

    graph.add_edge(START, "input_guardrail")
    graph.add_conditional_edges(
        "input_guardrail",
        route_after_guardrail,
        {
            "supervisor": "supervisor",
            END: END
        }
    )
    graph.add_edge("supervisor", "compiler")
    graph.add_edge("compiler", END)

    return graph.compile()