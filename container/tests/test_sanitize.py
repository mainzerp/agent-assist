"""FLOW-MED-4: canonical sanitizer corpus.

The container's :func:`app.agents.sanitize.strip_markdown` is the
canonical implementation. The HA custom_component's
``_strip_markdown`` in ``custom_components/ha_agenthub/conversation.py``
MUST produce identical output for every input in
``tests/data/sanitize_corpus.txt``. This test locks the container
side; HA parity is verified manually from the PR description.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.agents.sanitize import strip_markdown

CORPUS_PATH = Path(__file__).parent / "data" / "sanitize_corpus.txt"


def _load_cases() -> list[tuple[str, str]]:
    """Parse the corpus into (input, expected) pairs.

    Format:
      # comment lines start with '#'
      INPUT_LINE(s)
      <blank line>
      EXPECTED_LINE(s)
      ---
    """
    raw = CORPUS_PATH.read_text(encoding="utf-8")
    # Strip leading comment header
    cleaned_lines = []
    for line in raw.splitlines():
        if line.startswith("#"):
            continue
        cleaned_lines.append(line)
    body = "\n".join(cleaned_lines).strip("\n")

    cases: list[tuple[str, str]] = []
    for block in body.split("\n---\n"):
        block = block.strip("\n")
        if not block:
            continue
        parts = block.split("\n\n", 1)
        if len(parts) != 2:
            continue
        input_text, expected_text = parts[0], parts[1]
        cases.append((input_text, expected_text))
    return cases


CASES = _load_cases()


@pytest.mark.parametrize("input_text,expected_text", CASES)
def test_strip_markdown_corpus(input_text: str, expected_text: str) -> None:
    assert strip_markdown(input_text) == expected_text


def test_corpus_is_nonempty() -> None:
    assert len(CASES) >= 10, "corpus should cover the main markdown constructs"
