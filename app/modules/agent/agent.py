from langgraph.graph import StateGraph, START, END
from .utils.state import AgentState
from .utils.nodes import guardrail, llm_call, tool_node, should_continue, planner, reasoning

def route_after_guardrail(state: AgentState):
    last_message = state["messages"][-1] if state["messages"] else None
    print(last_message.type if last_message else None)

    if (last_message and
        last_message.type == "ai" and
        isinstance(last_message.content, str) and
        "can't assist" in last_message.content):
        return END
    return "planner"

builder = StateGraph(AgentState) \
    .add_node("guardrail", guardrail) \
    .add_node("planner", planner) \
    .add_node("reasoning", reasoning) \
    .add_node("llm_call", llm_call) \
    .add_node("tool_node", tool_node) \
    .add_edge(START, "guardrail") \
    .add_conditional_edges("guardrail", route_after_guardrail, ["planner", END]) \
    .add_edge("planner", "reasoning") \
    .add_edge("reasoning", "llm_call") \
    .add_conditional_edges("llm_call", should_continue, ["tool_node", END]) \
    .add_edge("tool_node", "reasoning")

agent = builder.compile()
