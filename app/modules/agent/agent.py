from langgraph.graph import StateGraph, START, END
from .utils.state import AgentState
from .utils.nodes import guardrail, llm_call, tool_node, should_continue, planner, reasoning

def route_after_guardrail(state: AgentState):
    guardrail_verdict = state.get("guardrail_verdict")
    if guardrail_verdict == "JAILBREAK" or guardrail_verdict == "OFF_TOPIC":
        return END
    return "planner"

def route_after_planner(state: AgentState):
    plan = state.get("plan", [])
    # If plan is empty, it means planner refused (secondary defense)
    if not plan or len(plan) == 0:
        return END
    return "reasoning"

builder = StateGraph(AgentState) \
    .add_node("guardrail", guardrail) \
    .add_node("planner", planner) \
    .add_node("reasoning", reasoning) \
    .add_node("llm_call", llm_call) \
    .add_node("tool_node", tool_node) \
    .add_edge(START, "guardrail") \
    .add_conditional_edges("guardrail", route_after_guardrail, ["planner", END]) \
    .add_conditional_edges("planner", route_after_planner, ["reasoning", END]) \
    .add_edge("reasoning", "llm_call") \
    .add_conditional_edges("llm_call", should_continue, ["tool_node", END]) \
    .add_edge("tool_node", "reasoning")

agent = builder.compile()
