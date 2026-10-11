from unittest.mock import MagicMock

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage

from meet_transcript_agent.agent.graph import build_graph
from meet_transcript_agent.agent.nodes import MAX_DISTANCE
from meet_transcript_agent.agent.state import AgentDeps, RouteDecision
from meet_transcript_agent.domain import Chunk


def _fake_chunks():
    return [
        Chunk(
            chunk_id="m1-alice:0",
            doc_id="m1-alice",
            index=0,
            text="[00:05] We need to push the deadline to Friday.",
            metadata={
                "doc_id": "m1-alice",
                "index": 0,
                "source_uri": "alice.wav",
                "speaker": "alice",
                "start_ts": 5.0,
                "end_ts": 35.0,
                "meeting_id": "m1",
            },
        ),
        Chunk(
            chunk_id="m1-bob:0",
            doc_id="m1-bob",
            index=0,
            text="[00:40] I can help with the API work tomorrow.",
            metadata={
                "doc_id": "m1-bob",
                "index": 0,
                "source_uri": "bob.wav",
                "speaker": "bob",
                "start_ts": 40.0,
                "end_ts": 70.0,
                "meeting_id": "m1",
            },
        ),
    ]


def _fake_store(chunks, score=0.5):
    store = MagicMock()
    if callable(score):
        store.search.side_effect = score
    else:
        store.search.return_value = [(c, score) for c in chunks]
    return store


def _fake_llm(invoke_results, route_to="knowledge"):
    llm = MagicMock()
    if isinstance(invoke_results, str):
        llm.invoke.return_value = AIMessage(content=invoke_results)
    else:
        llm.invoke.side_effect = [AIMessage(content=t) for t in invoke_results]

    router = MagicMock()
    router.invoke.return_value = RouteDecision(route=route_to)
    llm.with_structured_output.return_value = router

    return llm


def _invoke(question, deps):
    graph = build_graph(deps=deps)
    return graph.invoke({
        "question": question,
        "search_query": question,
        "messages": [],
    })


def test_graph_produces_answer_and_citations():
    chunks = _fake_chunks()
    deps = AgentDeps(
        llm=_fake_llm("Alice said the deadline needs to move to Friday [1]. Bob offered to help with the API [2]."),
        vector_store=_fake_store(chunks),
    )

    result = _invoke("What was discussed?", deps)

    assert "answer" in result
    assert "citations" in result
    assert len(result["retrieved"]) == 2
    assert result["citations"] == [
        "[1] alice, 00:05-00:35",
        "[2] bob, 00:40-01:10",
    ]


def test_graph_citations_only_include_referenced_sources():
    chunks = _fake_chunks()
    deps = AgentDeps(
        llm=_fake_llm("Alice said the deadline needs to move [1]."),
        vector_store=_fake_store(chunks),
    )

    result = _invoke("What did Alice say?", deps)

    assert result["citations"] == ["[1] alice, 00:05-00:35"]


def test_graph_no_citations_when_none_referenced():
    chunks = _fake_chunks()
    deps = AgentDeps(
        llm=_fake_llm("The transcript excerpts don't mention that topic."),
        vector_store=_fake_store(chunks),
    )

    result = _invoke("What about quantum physics?", deps)

    assert result["citations"] == []


def test_good_scores_go_straight_to_generate():
    chunks = _fake_chunks()
    deps = AgentDeps(
        llm=_fake_llm("Alice mentioned the deadline [1]."),
        vector_store=_fake_store(chunks, score=0.4),
    )

    result = _invoke("What about the deadline?", deps)

    assert len(result["retrieved"]) == 2
    assert result.get("attempts", 0) == 0


def test_bad_scores_trigger_rewrite_then_succeed():
    chunks = _fake_chunks()
    call_count = 0

    def search_with_improving_scores(query, k=5):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return [(c, MAX_DISTANCE + 0.5) for c in chunks]
        return [(c, 0.4) for c in chunks]

    deps = AgentDeps(
        llm=_fake_llm(invoke_results=["better query", "The answer is here [1]."]),
        vector_store=_fake_store(chunks, score=search_with_improving_scores),
    )

    result = _invoke("vague question", deps)

    assert result["attempts"] == 1
    assert len(result["retrieved"]) == 2


def test_exhausts_retries_then_generates():
    chunks = _fake_chunks()
    deps = AgentDeps(
        llm=_fake_llm(invoke_results=["rewrite 1", "rewrite 2", "Nothing found."]),
        vector_store=_fake_store(chunks, score=MAX_DISTANCE + 0.5),
    )

    result = _invoke("something unrelated", deps)

    assert result["attempts"] == 2
    assert result["retrieved"] == []
    assert "answer" in result


def test_score_threshold_filters_chunks():
    chunks = _fake_chunks()
    deps = AgentDeps(
        llm=_fake_llm(invoke_results=["rewrite 1", "rewrite 2", "Not found."]),
        vector_store=_fake_store(chunks, score=MAX_DISTANCE + 0.1),
    )

    result = _invoke("irrelevant query", deps)

    assert result["retrieved"] == []


def test_chunks_at_threshold_are_kept():
    chunks = _fake_chunks()
    deps = AgentDeps(
        llm=_fake_llm("Found it [1]."),
        vector_store=_fake_store(chunks, score=MAX_DISTANCE),
    )

    result = _invoke("borderline query", deps)

    assert len(result["retrieved"]) == 2


def test_chitchat_skips_retrieval():
    store = _fake_store([])
    deps = AgentDeps(
        llm=_fake_llm("You're welcome!", route_to="chitchat"),
        vector_store=store,
    )

    result = _invoke("thanks!", deps)

    assert result["answer"] == "You're welcome!"
    assert result["retrieved"] == []
    store.search.assert_not_called()


def test_route_parse_failure_defaults_to_knowledge():
    chunks = _fake_chunks()
    llm = _fake_llm("Alice mentioned something [1].")
    llm.with_structured_output.return_value.invoke.side_effect = OutputParserException("bad")
    deps = AgentDeps(
        llm=llm,
        vector_store=_fake_store(chunks),
    )

    result = _invoke("What did Alice say?", deps)

    assert len(result["retrieved"]) == 2
    assert "answer" in result
