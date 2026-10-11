import shutil
import uuid
from datetime import UTC
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from meet_transcript_agent.config import settings
from meet_transcript_agent.models import get_embeddings, get_llm

app = typer.Typer()
console = Console()
error_console = Console(stderr=True)


@app.command()
def stats() -> None:
    llm = get_llm()

    try:
        response = llm.invoke("ping")
        console.print(f"[bold blue]{llm.model}: {response.content}[/bold blue]")
    except Exception as e:  # noqa: BLE001
        error_console.print(
            f"[bold red]Error occurred while invoking LLM: {e}[/bold red]"
        )
        raise typer.Exit(1)

    emb = get_embeddings()

    try:
        vector = emb.embed_query("ping")
        console.print(f"[bold blue]{emb.model}: {len(vector)}[/bold blue]")
    except Exception as e:  # noqa: BLE001
        error_console.print(
            f"[bold red]Error occurred while invoking Embeddings: {e}[/bold red]"
        )
        raise typer.Exit(1)


@app.command()
def transcribe(
    audio_file: Path = typer.Argument(help="Audio file to transcribe"),
) -> None:
    from meet_transcript_agent.transcribe import transcribe as run_transcribe

    if not audio_file.exists():
        error_console.print(f"[bold red]File not found: {audio_file}[/bold red]")
        raise typer.Exit(1)

    segments = run_transcribe(audio_file, settings)

    for seg in segments:
        minutes, secs = divmod(int(seg.start), 60)
        console.print(f"[dim][{minutes:02d}:{secs:02d}][/dim] {seg.text}")


@app.command()
def ingest(
    meeting_dir: Path = typer.Argument(help="Meeting directory to ingest"),
    rebuild: bool = typer.Option(False, "--rebuild", help="Clear the index before ingesting"),
) -> None:
    from meet_transcript_agent.ingest.pipeline import IngestPipeline

    if rebuild:
        chroma_dir = settings.data_dir / "chroma"
        if chroma_dir.exists():
            shutil.rmtree(chroma_dir)
        manifest_db = settings.data_dir / "document_agent.sqlite"
        if manifest_db.exists():
            manifest_db.unlink()

    report = IngestPipeline().ingest(meeting_dir)

    table = Table(title="Ingestion Report")
    table.add_column("added", style="green", no_wrap=True, justify="center")
    table.add_column("updated", style="blue", no_wrap=True, justify="center")
    table.add_column("skipped", style="white", no_wrap=True, justify="center")
    table.add_row(str(report.added), str(report.updated), str(report.skipped))
    console.print(table)


def _make_checkpointer():
    import sqlite3

    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from langgraph.checkpoint.sqlite import SqliteSaver

    from meet_transcript_agent.domain import Chunk

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    serde = JsonPlusSerializer(allowed_msgpack_modules=[Chunk])
    return SqliteSaver(sqlite3.connect(settings.data_dir / "checkpoints.sqlite", check_same_thread=False), serde=serde)


@app.command()
def ask(
    question: str = typer.Argument(help="Question to ask about your meetings"),
    thread: str = typer.Option(None, "--thread", help="Thread ID for conversation memory"),
) -> None:
    from langchain_core.messages import AIMessageChunk

    from meet_transcript_agent.agent.graph import build_graph

    thread_id = thread or str(uuid.uuid4())
    saver = _make_checkpointer()
    graph = build_graph(checkpointer=saver)
    citations = []

    console.print()
    for mode, chunk in graph.stream(
        {"question": question, "search_query": question, "messages": []},
        stream_mode=["messages", "values"],
        config={"configurable": {"thread_id": thread_id}},
    ):
        if mode == "messages":
            msg_chunk, metadata = chunk
            if isinstance(msg_chunk, AIMessageChunk) and msg_chunk.content and metadata.get("langgraph_node") in ("generate", "respond_direct"):
                print(msg_chunk.content, end="", flush=True)
        elif mode == "values" and "citations" in chunk:
            citations = chunk["citations"]

    print("\n")
    if citations:
        console.print("[dim]Sources:[/dim]")
        for source in citations:
            console.print(f"  - {source}")


@app.command()
def threads() -> None:
    from datetime import datetime

    saver = _make_checkpointer()
    seen = set()
    table = Table(title="Threads")
    table.add_column("thread", style="bold")
    table.add_column("messages", justify="right")
    table.add_column("last active", style="dim")

    for checkpoint_tuple in saver.list(None):
        thread_id = checkpoint_tuple.config["configurable"]["thread_id"]
        if thread_id in seen:
            continue
        seen.add(thread_id)

        channel_values = checkpoint_tuple.checkpoint.get("channel_values", {})
        messages = channel_values.get("messages", [])
        ts = checkpoint_tuple.checkpoint.get("ts", "")
        if ts:
            dt = datetime.fromisoformat(ts).astimezone(UTC)
            last_active = dt.strftime("%Y-%m-%d %H:%M UTC")
        else:
            last_active = "?"

        table.add_row(thread_id, str(len(messages)), last_active)

    if not seen:
        console.print("[dim]No threads found.[/dim]")
    else:
        console.print(table)
