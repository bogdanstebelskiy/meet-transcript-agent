import json
import subprocess
import tempfile
from pathlib import Path

from pydantic import BaseModel

from meet_transcript_agent.config import Settings
from meet_transcript_agent.config import settings as default_settings


class Segment(BaseModel):
    start: float
    end: float
    text: str


def transcribe(
    audio_path: Path,
    settings: Settings = default_settings,
) -> list[Segment]:
    resolved_audio = Path(audio_path).resolve()

    with tempfile.TemporaryDirectory() as tmp:
        output_prefix = Path(tmp) / "out"
        cmd = [
            str(settings.whisper_bin),
            "-m", str(settings.whisper_model),
            "-l", settings.whisper_language,
            "-oj",
            "-of", str(output_prefix),
            str(resolved_audio),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"whisper-cli exited with code {result.returncode}: {result.stderr.strip()}"
            )

        json_path = output_prefix.with_suffix(".json")
        if not json_path.exists():
            raise FileNotFoundError(
                f"whisper-cli did not produce expected output at {json_path}"
            )
        data = json.loads(json_path.read_text(encoding="utf-8"))

    segments = []
    for item in data.get("transcription", []):
        offsets = item["offsets"]
        text = item["text"].strip()
        if text:
            segments.append(Segment(
                start=offsets["from"] / 1000.0,
                end=offsets["to"] / 1000.0,
                text=text,
            ))

    return segments
