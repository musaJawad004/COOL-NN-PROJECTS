"""Import a balanced 50-file human-speech subset of RAVDESS.

RAVDESS is distributed under CC BY-NC-SA 4.0. This script downloads files from
the Hugging Face dataset mirror and converts them to the app's 16 kHz format.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import tempfile
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emotion_ai.audio import load_wav, save_wav  # noqa: E402


SOURCE = "https://huggingface.co/datasets/birgermoell/ravdess/resolve/main"
EMOTION_CODES = {"calm": "02", "happy": "03", "sad": "04", "angry": "05", "nervous": "06"}


def filename(
    emotion_code: str,
    actor: int,
    intensity: int,
    statement: int,
    repetition: int,
) -> str:
    return f"03-01-{emotion_code}-{intensity:02d}-{statement:02d}-{repetition:02d}-{actor:02d}.wav"


def download_one(
    emotion: str,
    code: str,
    actor: int,
    intensity: int,
    statement: int,
    repetition: int,
) -> Path:
    name = filename(code, actor, intensity, statement, repetition)
    url = f"{SOURCE}/Actor_{actor:02d}/{name}?download=true"
    destination = ROOT / "data" / emotion / f"ravdess_{name}"
    if destination.exists():
        return destination
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with urlopen(url, timeout=90) as response, tempfile.NamedTemporaryFile(suffix=".wav") as temporary:
                temporary.write(response.read())
                temporary.flush()
                save_wav(destination, load_wav(Path(temporary.name)))
            return destination
        except Exception as error:
            last_error = error
            time.sleep(1.5 * (attempt + 1))
    assert last_error is not None
    raise last_error


def main() -> None:
    jobs = [
        (emotion, code, actor, intensity, statement, repetition)
        for emotion, code in EMOTION_CODES.items()
        for actor in range(1, 25)
        for intensity, statement, repetition in ((1, 1, 1), (2, 2, 2))
    ]
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda job: download_one(*job), jobs))
    print(f"Verified {len(results)} balanced RAVDESS speech samples.")


if __name__ == "__main__":
    main()
