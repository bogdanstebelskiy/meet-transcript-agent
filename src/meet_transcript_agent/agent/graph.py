from functools import partial

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from meet_transcript_agent.agent.nodes import (
    generate,
    reset_turn,
    respond_direct,
    retrieve,
    rewrite_query,
    route,
)
from meet_transcript_agent.agent.state import AgentDeps, AgentState


def route_decision(state: AgentState):
    return state.get("route", "knowledge")


def retrieve_or_generate(state: AgentState):
    if state["retrieved"]:
        return "generate"
    if state.get("attempts", 0) < 2:
        return "rewrite_query"
    return "generate"


def build_graph(
    checkpointer: BaseCheckpointSaver | None = None,
    deps: AgentDeps | None = None,
):
    deps = deps or AgentDeps.default()
    builder = StateGraph(AgentState)

    builder.add_node("reset_turn", partial(reset_turn, deps=deps))
    builder.add_node("route", partial(route, deps=deps))
    builder.add_node("retrieve", partial(retrieve, deps=deps))
    builder.add_node("generate", partial(generate, deps=deps))
    builder.add_node("respond_direct", partial(respond_direct, deps=deps))
    builder.add_node("rewrite_query", partial(rewrite_query, deps=deps))

    builder.add_edge(START, "reset_turn")
    builder.add_edge("reset_turn", "route")
    builder.add_conditional_edges("route", route_decision, {
        "knowledge": "retrieve",
        "chitchat": "respond_direct",
    })
    builder.add_conditional_edges("retrieve", retrieve_or_generate)
    builder.add_edge("rewrite_query", "retrieve")
    builder.add_edge("generate", END)
    builder.add_edge("respond_direct", END)

    return builder.compile(checkpointer=checkpointer)
