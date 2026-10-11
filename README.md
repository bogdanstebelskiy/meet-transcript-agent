# meet-transcript-agent

CLI tool for transcribing meeting audio with whisper.cpp and asking questions about what was said. Uses RAG with a LangGraph agent that retrieves relevant transcript chunks, rewrites queries when results are weak, and cites speakers with timestamps in answers.

Each meeting is a directory with per-speaker audio files and a `meeting.json` metadata file. Speaker identity comes from the audio filename.

## Stack

- **Transcription**: whisper.cpp (called via subprocess, you compile for your hardware)
- **LLM/embeddings**: Ollama (default models: `qwen2.5:7b-instruct`, `nomic-embed-text`)
- **Vector store**: ChromaDB
- **Agent framework**: LangGraph
- **CLI**: Typer + Rich

## Usage

```bash
# Transcribe a single audio file (test whisper.cpp setup)
meet-transcript-agent transcribe recording.wav

# Ingest a meeting directory
meet-transcript-agent ingest ./meetings/2026-01-15-standup

# Ask questions about ingested meetings
meet-transcript-agent ask "What did Alice say about the deadline?"
meet-transcript-agent ask "Summarize the discussion" --thread <thread-id>

# List conversation threads
meet-transcript-agent threads

# Check Ollama connectivity
meet-transcript-agent stats
```

## Meeting directory format

```
meetings/
  2026-01-15-standup/
    meeting.json
    alice.wav
    bob.wav
```

`meeting.json`:
```json
{
  "meeting_id": "2026-01-15-standup",
  "title": "Daily standup",
  "date": "2026-01-15",
  "participants": ["alice", "bob"]
}
```

Audio files can be `.wav`, `.mp3`, `.ogg`, `.flac`, or `.m4a`. The filename stem becomes the speaker name.

## Environment variables

All prefixed with `DA_`, configured via `.env` or shell.

| Variable | Default | Description |
|---|---|---|
| `DA_WHISPER_BIN` | `whisper-cli` | Path to compiled whisper.cpp binary |
| `DA_WHISPER_MODEL` | `models/ggml-base.bin` | Path to GGML model file |
| `DA_WHISPER_LANGUAGE` | `en` | Transcription language |
| `DA_LLM_MODEL` | `qwen2.5:7b-instruct` | Ollama chat model |
| `DA_EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `DA_DATA_DIR` | `.data` | Directory for ChromaDB, manifests, checkpoints |
| `DA_CHUNK_DURATION` | `30.0` | Seconds per transcript chunk window |
| `DA_CHUNK_OVERLAP_DURATION` | `5.0` | Seconds of overlap between chunks |

## How it works

1. **Transcribe**: whisper.cpp processes each speaker's audio file into timestamped segments
2. **Chunk**: Segments are grouped into time windows (default 30s with 5s overlap)
3. **Embed**: Chunks are embedded with `nomic-embed-text` and stored in ChromaDB with speaker/timestamp metadata
4. **Query**: Questions go through a LangGraph agent that retrieves relevant chunks, retries with rewritten queries if needed, and generates answers with speaker-attributed citations like `[alice, 02:15-02:45]`

Re-ingesting a meeting with unchanged audio is a no-op (hash-based dedup). Changed audio triggers re-transcription of that speaker's track only.
