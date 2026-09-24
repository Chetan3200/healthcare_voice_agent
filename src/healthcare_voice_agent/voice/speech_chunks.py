"""Lossless, punctuation-guided splitting for earlier first audio.

At most one boundary is chosen for the first TTS request in a context. The
remainder stays intact so a short tail can never become a third native request.
This is a conservative punctuation heuristic, not a grammatical parser or a
promise of seamless prosody. Without a suitable boundary, prefer intact speech
rather than forcing a cut to meet a character budget.
"""

from __future__ import annotations

from collections.abc import Iterator

# Do not treat arbitrary punctuation (hyphens, slashes, apostrophes, underscores,
# etc.) as a spoken pause.
_PAUSE_MARKS = frozenset(",;:.!?।॥…")
_OPENERS = {"(": ")", "[": "]", "{": "}", "“": "”", "‘": "’", "«": "»"}
_CLOSERS = frozenset(")]}”’»\"'")
_ABBREVIATIONS = frozenset({
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "no", "vs", "etc",
    "approx", "dept", "fig", "inc", "ltd", "डॉ", "प्रो",
})


def _valid_target(target_chars: int) -> None:
    if isinstance(target_chars, bool) or not isinstance(target_chars, int):
        raise ValueError("First-audio chunk characters must be an integer")
    if target_chars and not 32 <= target_chars <= 256:
        raise ValueError("First-audio chunk characters must be 0 or 32 to 256")


def _is_pause(text: str, space_start: int, space_end: int) -> bool:
    index = space_start - 1
    # A completed quotation or parenthesis stays with the punctuation before it.
    while index >= 0 and text[index] in _CLOSERS:
        index -= 1
    if index < 0 or text[index] not in _PAUSE_MARKS:
        return False
    mark = text[index]
    if mark == ".":
        token = text[:index + 1].rsplit(None, 1)[-1].lstrip("([{\"'“‘«")
        stem = token[:-1]
        # Titles, initials, dotted abbreviations and numbered items are not
        # sentence endings. Missing some real endings is safer than cutting names.
        if ("." in stem or stem.casefold() in _ABBREVIATIONS or stem.isdigit()
                or (len(stem) == 1 and stem.isalpha())):
            return False
    if mark in ",:." and index > 0 and text[index - 1].isdigit():
        if space_end < len(text) and text[space_end].isdigit():
            return False  # Spaced numeric/time notation: 1, 000 / 10: 30.
    return True


def _boundaries(text: str) -> Iterator[int]:
    """Yield punctuation-plus-whitespace cuts outside quotes and parentheses."""
    stack: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char.isspace():
            end = index + 1
            while end < len(text) and text[end].isspace():
                end += 1
            if not stack and end < len(text) and _is_pause(text, index, end):
                yield end
            index = end
            continue
        # Apostrophes inside words must not open/close a quotation.
        apostrophe = char in "'’" and 0 < index < len(text) - 1 and (
            text[index - 1].isalnum() and text[index + 1].isalnum()
        )
        if not apostrophe:
            if stack and char == stack[-1]:
                stack.pop()
            elif char in _OPENERS:
                stack.append(_OPENERS[char])
            elif char in "\"'":
                stack.append(char)
        index += 1


def _substantial(text: str, min_chars: int) -> bool:
    words = sum(any(char.isalnum() for char in token) for token in text.split())
    return len(text.strip()) >= min_chars and words >= 3


def split_for_first_audio(text: str, target_chars: int) -> tuple[str, ...]:
    """Return intact text or two lossless pieces at a conservative pause.

    ``target_chars`` is a preference, never a hard length limit. Look for a
    punctuation boundary within twice that target, nearest the target, with at
    least three word-like tokens and 16 to 24 characters on each side. Ignore cuts
    inside quotes/parentheses, abbreviations and numeric notation. Never fall
    back to ordinary whitespace. Keep the entire remainder in one request.

    Existing punctuation and following whitespace stay on the first piece.
    Zero disables splitting; whitespace-only text produces no synthesis piece.
    """
    _valid_target(target_chars)
    if not text or not text.strip():
        return ()
    if target_chars == 0 or len(text) <= target_chars:
        return (text,)

    min_chars = max(16, min(24, target_chars // 2))
    candidates = []
    for cut in _boundaries(text):
        if cut > 2 * target_chars:
            break
        if _substantial(text[:cut], min_chars) and _substantial(text[cut:], min_chars):
            candidates.append(cut)
    if not candidates:
        return (text,)

    cut = min(candidates, key=lambda value: (abs(value - target_chars), value))
    pieces = (text[:cut], text[cut:])
    assert "".join(pieces) == text and all(piece.strip() for piece in pieces)
    return pieces
