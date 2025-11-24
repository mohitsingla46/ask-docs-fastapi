from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from .utils.state import AgentState
from .utils.nodes import guardrail, llm_call, tool_node, should_continue

def route_after_guardrail(state: AgentState):
    last_message = state["messages"][-1] if state["messages"] else None
    print(last_message.type if last_message else None)

    if (last_message and
        last_message.type == "ai" and
        isinstance(last_message.content, str) and
        "can't assist" in last_message.content):
        return END
    return "llm_call"

builder = StateGraph(AgentState) \
    .add_node("guardrail", guardrail) \
    .add_node("llm_call", llm_call) \
    .add_node("tool_node", tool_node) \
    .add_edge(START, "guardrail") \
    .add_conditional_edges("guardrail", route_after_guardrail, ["llm_call", END]) \
    .add_conditional_edges("llm_call", should_continue, ["tool_node", END]) \
    .add_edge("tool_node", "llm_call")

checkpointer = MemorySaver()
agent = builder.compile(checkpointer=checkpointer)
