import logging
import re

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

from meet_transcript_agent.agent.prompts import GENERATION_PROMPT
from meet_transcript_agent.agent.state import AgentDeps, AgentState, RouteDecision

logger = logging.getLogger(__name__)

MAX_DISTANCE = 1.2


def _format_timestamp(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:02d}:{secs:02d}"


def _source_label(chunk) -> str:
    speaker = chunk.metadata.get("speaker", "unknown")
    start = chunk.metadata.get("start_ts", 0.0)
    end = chunk.metadata.get("end_ts", 0.0)
    return f"{speaker}, {_format_timestamp(start)}-{_format_timestamp(end)}"


def retrieve(state: AgentState, *, deps: AgentDeps) -> dict:
    query = state["search_query"]
    logger.info("retrieve: query=%r", query)
    results = deps.vector_store.search(query, k=5)
    seen = set()
    chunks = []
    for chunk, score in results:
        if chunk.chunk_id not in seen and score <= MAX_DISTANCE:
            seen.add(chunk.chunk_id)
            chunks.append(chunk)
            logger.info("retrieve: kept chunk %s (score=%.3f)", chunk.chunk_id, score)
        elif chunk.chunk_id not in seen:
            logger.info("retrieve: dropped chunk %s (score=%.3f)", chunk.chunk_id, score)
    return {"retrieved": chunks}


def generate(state: AgentState, *, deps: AgentDeps) -> dict:
    logger.info("generate: %d chunks", len(state["retrieved"]))
    chunks = state["retrieved"]
    context = "\n\n".join(
        f"[{i + 1}] ({_source_label(chunk)})\n{chunk.text}"
        for i, chunk in enumerate(chunks)
    )
    prompt = GENERATION_PROMPT.format(
        question=state["search_query"],
        context=context,
    )

    response = deps.llm.invoke(prompt)
    answer_text = response.content

    cited_indices = {int(m) for m in re.findall(r"\[(\d+)\]", answer_text)}
    citations = [
        f"[{i}] {_source_label(chunks[i - 1])}"
        for i in sorted(cited_indices)
        if 1 <= i <= len(chunks)
    ]

    return {
        "messages": [
            HumanMessage(content=state["question"]),
            AIMessage(content=answer_text),
        ],
        "answer": answer_text,
        "citations": citations,
    }


def route(state: AgentState, *, deps: AgentDeps) -> dict:
    logger.info("route: question=%r", state["question"])
    router = deps.llm.with_structured_output(RouteDecision)
    try:
        result = router.invoke(
            "Classify this user message as either 'knowledge' or 'chitchat'.\n"
            "- 'knowledge': needs information lookup (questions, requests for facts)\n"
            "- 'chitchat': greetings, thanks, small talk, acknowledgements\n\n"
            f"Message: {state['question']}"
        )
        decision = result.route if result.route in ("knowledge", "chitchat") else "knowledge"
    except (OutputParserException, ValidationError):
        decision = "knowledge"
    logger.info("route: decision=%s", decision)
    return {"route": decision}


def respond_direct(state: AgentState, *, deps: AgentDeps) -> dict:
    logger.info("respond_direct: question=%r", state["question"])
    recent_messages = state.get("messages", [])[-6:]
    if recent_messages:
        history = "\n".join(f"{m.type}: {m.content}" for m in recent_messages)
        prompt = (
            f"Conversation so far:\n{history}\n\n"
            f"User: {state['question']}\n"
            "Respond naturally and briefly."
        )
    else:
        prompt = f"User: {state['question']}\nRespond naturally and briefly."

    response = deps.llm.invoke(prompt).content
    return {
        "messages": [
            HumanMessage(content=state["question"]),
            AIMessage(content=response),
        ],
        "answer": response,
    }


def rewrite_query(state: AgentState, *, deps: AgentDeps) -> dict:
    logger.info("rewrite_query: attempts=%d, question=%r", state.get("attempts", 0), state["question"])

    recent_messages = state.get("messages", [])[-6:]
    if recent_messages:
        history = "\n".join(f"{m.type}: {m.content}" for m in recent_messages)
        prompt = (
            f"Given this conversation history:\n{history}\n\n"
            f"Rephrase this question into a standalone search query "
            f"that resolves any pronouns or references: {state['question']}"
        )
    else:
        prompt = f"Rephrase this question for better search: {state['question']}"

    updated_query = deps.llm.invoke(prompt).content
    updated_attempts = state.get("attempts", 0) + 1

    return {
        "attempts": updated_attempts,
        "search_query": updated_query,
    }


def reset_turn(state: AgentState, *, deps: AgentDeps) -> dict:
    question = state["question"]
    recent_messages = state.get("messages", [])[-6:]
    if recent_messages:
        history = "\n".join(f"{m.type}: {m.content}" for m in recent_messages)
        search_query = deps.llm.invoke(
            "Given this conversation history:\n"
            f"{history}\n\n"
            "The user now asks a follow-up question. Rewrite it as a "
            "standalone search query by replacing pronouns (it, this, "
            "that, they, etc.) with the specific subject from the "
            "conversation. The subject is usually the main topic being "
            "discussed, not a sub-concept.\n\n"
            "Examples:\n"
            "- History about a standup meeting, follow-up "
            "\"What did they say about it?\" -> "
            "\"What did the team say about the deadline?\"\n"
            "- History about API discussion, follow-up "
            "\"Who volunteered?\" -> \"Who volunteered to work on the API?\"\n\n"
            f"Follow-up: {question}\n"
            "Standalone query:"
        ).content.strip()
    else:
        search_query = question

    return {
        "retrieved": [],
        "search_query": search_query,
        "attempts": 0,
        "answer": "",
        "citations": [],
    }
