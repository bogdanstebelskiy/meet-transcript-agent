from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from meet_transcript_agent.domain import Chunk


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    search_query: str
    retrieved: list[Chunk]
    attempts: int
    route: str
    answer: str
    citations: list[str]


class RouteDecision(BaseModel):
    route: str


@dataclass
class AgentDeps:
    llm: object
    vector_store: object

    @staticmethod
    def default() -> AgentDeps:
        from meet_transcript_agent.models import get_llm
        from meet_transcript_agent.store.vector import VectorStore

        return AgentDeps(llm=get_llm(), vector_store=VectorStore.default())