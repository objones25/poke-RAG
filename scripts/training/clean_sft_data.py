from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_BAD_ANSWER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bprovided (text|context|chunk)\b", re.IGNORECASE),
    re.compile(r"\bdoes not (contain|include|provide|offer)\b", re.IGNORECASE),
    re.compile(r"\bnot enough information\b", re.IGNORECASE),
    re.compile(r"\bno (further |additional )?information\b", re.IGNORECASE),
    re.compile(r"\bonly (the )?(title|name)\b", re.IGNORECASE),
    re.compile(r"\bidentifies .{0,40} as the title\b", re.IGNORECASE),
    re.compile(r"\bno (descriptive|further|additional) content\b", re.IGNORECASE),
    re.compile(r"\bnot (included|provided)\b", re.IGNORECASE),
    re.compile(r"\bno further detail\b", re.IGNORECASE),
    re.compile(r"\bno (content|data|details?) (is |are )?provided\b", re.IGNORECASE),
    re.compile(r"\bcannot (be |)answered\b", re.IGNORECASE),
    re.compile(r"\binsufficient (information|data|context)\b", re.IGNORECASE),
]

_BARE_POKEMON_RE = re.compile(r"^[\w][\w\s\-']+\s+is an? [Pp]ok[eé]mon\.?\s*$")
_VAR_RE = re.compile(r"\[VAR\s*\(", re.IGNORECASE)
_DB_ID_RE = re.compile(r"\b\w+\d{3,}\b")
_DYNAMAX_CRYSTAL_RE = re.compile(r"Dynamax Crystal", re.IGNORECASE)
_MIN_ANSWER_LEN = 40


def _is_bad_text(text: str) -> bool:
    if _VAR_RE.search(text):
        return True
    if _DB_ID_RE.search(text):
        return True
    return bool(_DYNAMAX_CRYSTAL_RE.search(text))


def _is_bad_answer(answer: str) -> bool:
    stripped = answer.strip()
    if len(stripped) < _MIN_ANSWER_LEN:
        return True
    if _BARE_POKEMON_RE.match(stripped):
        return True
    return any(pat.search(answer) for pat in _BAD_ANSWER_PATTERNS)


def _normalize_messages(messages: list[dict[str, str]]) -> list[dict[str, str]] | None:
    roles = [m.get("role") for m in messages]
    if roles == ["system", "user", "assistant"]:
        return [messages[1], messages[2]]
    if roles == ["user", "assistant"]:
        return messages
    return None


def clean(input_path: Path, output_path: Path) -> tuple[int, int, int]:
    total = 0
    kept = 0
    removed = 0

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", delete=False, suffix=".jsonl"
    ) as tmp:
        tmp_path = Path(tmp.name)
        try:
            with open(input_path, encoding="utf-8") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    total += 1
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        logger.warning("Skipping invalid JSON line.")
                        removed += 1
                        continue

                    messages = data.get("messages", [])
                    normalized = _normalize_messages(messages)
                    if normalized is None:
                        logger.debug("Removing entry with unexpected message structure.")
                        removed += 1
                        continue

                    user_content = next(
                        (m["content"] for m in normalized if m.get("role") == "user"), ""
                    )
                    assistant_content = next(
                        (m["content"] for m in normalized if m.get("role") == "assistant"), ""
                    )

                    if _is_bad_text(user_content) or _is_bad_text(assistant_content):
                        removed += 1
                        continue

                    if _is_bad_answer(assistant_content):
                        removed += 1
                        continue

                    data["messages"] = normalized
                    tmp.write(json.dumps(data) + "\n")
                    kept += 1

            tmp.flush()
            os.replace(tmp_path, output_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    return total, kept, removed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean an SFT JSONL file: strip system messages, remove bad Q&A pairs."
    )
    parser.add_argument("input", type=Path, help="Input JSONL file")
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--inplace", action="store_true", help="Overwrite the input file")
    output_group.add_argument(
        "--output", type=Path, default=None, help="Write cleaned data to this path"
    )
    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"Input file not found: {args.input}")

    if args.inplace:
        output = args.input
    elif args.output is not None:
        output = args.output
        output.parent.mkdir(parents=True, exist_ok=True)
    else:
        parser.error("Specify --inplace or --output <path>.")

    total, kept, removed = clean(args.input, output)
    logger.info(
        "Done. Total=%d, Kept=%d, Removed=%d (%.1f%% kept).",
        total,
        kept,
        removed,
        100.0 * kept / total if total else 0.0,
    )


if __name__ == "__main__":
    main()
