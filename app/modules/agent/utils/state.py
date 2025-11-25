from typing_extensions import TypedDict, Annotated
from langchain.messages import AnyMessage
import operator

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    user_id: str
    plan: list[str]
    reasoning_trace: list[str]