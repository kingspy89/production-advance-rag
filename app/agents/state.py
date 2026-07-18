from typing import TypedDict, List, Annotated, Optional
import operator


class AgentState(TypedDict):
    # Using Annotated with operator.add ensures that messages 
    # are appended to the history rather than replaced.
    messages: Annotated[List[dict], operator.add]
    current_query: str
    documents: List[str]
    plan: List[str]
    status: str
    final_answer: str
    api_key: Optional[str]
    gemini_api_key: Optional[str]

