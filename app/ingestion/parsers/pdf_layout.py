import math
import re
import unicodedata
from collections import Counter

JOINERS = set(",.:/+-@_&=?#")  # neutrals that stay inside a Latin word or a number
COLUMN_GAP = 1.0  # a gap wider than this many font sizes separates table cells
WORD_GAP = 0.2  # a gap wider than this many font sizes is a space


def _is_arabic_char(c: str) -> bool:
    return "\u0600" <= c <= "\u06ff" or "\ufb50" <= c <= "\ufdff" or "\ufe70" <= c <= "\ufeff"


def _kind(text: str) -> str:
    """A = Arabic letter, L = left-to-right (Latin letters and digits), N = neutral."""
    c = text[0]
    if c.isspace():
        return "N"
    if c.isdigit() or c in "%$\u00b0":  # digits (also Arabic-Indic) and signs glued to numbers
        return "L"
    if _is_arabic_char(c):
        return "A"
    if c.isalpha():
        return "L"
    return "N"


def group_lines(chars: list[dict]) -> list[list[dict]]:
    """Group characters into visual lines by their vertical position."""

    def mid(c: dict) -> float:
        return (c["top"] + c["bottom"]) / 2

    lines: list[list[dict]] = []
    current: list[dict] = []
    for c in sorted(chars, key=mid):
        if current and abs(mid(c) - sum(map(mid, current)) / len(current)) > 0.5 * c["size"]:
            lines.append(current)
            current = []
        current.append(c)
    if current:
        lines.append(current)
    return lines


def _is_latin_letter(token: tuple[str, str]) -> bool:
    return token[0] == "L" and token[1][0].isascii() and token[1][0].isalpha()


def _joins_latin_words(tokens: list[tuple[str, str]], j: int) -> bool:
    """A space keeps two words in one left-to-right run only between two Latin words."""
    return 0 < j < len(tokens) - 1 and _is_latin_letter(tokens[j - 1]) and _is_latin_letter(tokens[j + 1])


def _reverse_ltr_runs(tokens: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Read right to left, a Latin word or a number comes out reversed: flip each run back."""
    out: list[tuple[str, str]] = []
    i, n = 0, len(tokens)
    while i < n:
        if tokens[i][0] != "L":
            out.append(tokens[i])
            i += 1
            continue
        j, last = i, i
        while j < n and tokens[j][0] in ("L", "N"):
            if tokens[j][1] == " " and not _joins_latin_words(tokens, j):
                break
            if tokens[j][0] == "N" and tokens[j][1] != " " and tokens[j][1] not in JOINERS:
                break
            if tokens[j][0] == "L":
                last = j
            j += 1
        out.extend(reversed(tokens[i : last + 1]))
        i = last + 1
    return out


BRACKETS = ("()", "[]", "{}", "\u00ab\u00bb")


def _unmirror(text: str) -> str:
    """Some PDFs store brackets as their mirrored shape: a closer that comes before its opener is swapped back."""
    for opener, closer in BRACKETS:
        first_open, first_close = text.find(opener), text.find(closer)
        if first_open != -1 and first_close != -1 and first_close < first_open:
            text = text.translate(str.maketrans({opener: closer, closer: opener}))
    return text


def line_text(chars: list[dict], page_rtl: bool = False) -> str:
    """Rebuild one visual line in reading order (right to left for Arabic lines).

    `page_rtl` says the page is mostly Arabic, so a line with a long English phrase still reads right to left.
    """
    letters = [c for c in chars if unicodedata.category(c["text"][0]) != "Mn"]  # drop diacritics
    arabic = sum(1 for c in letters if _kind(c["text"]) == "A")
    latin = sum(1 for c in letters if _is_latin_letter((_kind(c["text"]), c["text"])))
    rtl = arabic > 0 and (page_rtl or arabic >= latin)
    letters.sort(key=(lambda c: -c["x1"]) if rtl else (lambda c: c["x0"]))
    tokens: list[tuple[str, str]] = []
    prev = None
    for c in letters:
        if prev is not None:
            gap = (prev["x0"] - c["x1"]) if rtl else (c["x0"] - prev["x1"])
            if gap > COLUMN_GAP * c["size"]:
                tokens.append(("S", " | "))
            elif gap > WORD_GAP * c["size"] and (not tokens or tokens[-1][1] != " "):
                tokens.append(("N", " "))
        prev = c
        if c["text"].isspace():
            if tokens and tokens[-1][0] != "S" and tokens[-1][1] != " ":
                tokens.append(("N", " "))
            continue
        tokens.append((_kind(c["text"]), c["text"]))
    if rtl:
        tokens = _reverse_ltr_runs(tokens)
    result = "".join(t for _, t in tokens).strip()
    return _unmirror(result) if rtl else result


def page_text(page) -> str:
    """Text of a pdfplumber page, with Arabic read in the right order."""
    chars = page.dedupe_chars().chars if hasattr(page, "dedupe_chars") else page.chars
    arabic = sum(1 for c in chars if _kind(c["text"]) == "A")
    latin = sum(1 for c in chars if _is_latin_letter((_kind(c["text"]), c["text"])))
    page_rtl = arabic > 0 and arabic >= latin
    lines = (line_text(line, page_rtl) for line in group_lines(chars))
    return "\n".join(t for t in lines if t)




_PAGE_COUNTER = re.compile(r"\d+\s*/\s*\d+|(?:\u0635\u0641\u062d\u0629|page)\s*\d+(?:\s*(?:\u0645\u0646|of)\s*\d+)?", re.IGNORECASE)


def _line_key(line: str) -> str:
    """Page counters ("4 / 1", "Page 3") change from page to page, so they are ignored when comparing lines."""
    return _PAGE_COUNTER.sub("#", line).strip()


def remove_repeated_lines(pages: list[str], edge: int = 2, share: float = 0.6) -> list[str]:
    """Drop running headers and footers: lines at the top or bottom edge of most pages.

    The first page that has such a line keeps it, so no information is lost, only repeated.
    """
    split = [page.splitlines() for page in pages]
    counts: Counter[str] = Counter()
    for lines in split:
        edges = lines[:edge] + lines[-edge:]
        counts.update({_line_key(line) for line in edges if line.strip()})
    threshold = max(2, math.ceil(share * len(pages)))
    repeated = {key for key, n in counts.items() if n >= threshold}

    seen: set[str] = set()
    cleaned: list[str] = []
    for lines in split:
        kept: list[str] = []
        for index, line in enumerate(lines):
            key = _line_key(line)
            at_edge = index < edge or index >= len(lines) - edge
            if at_edge and key in repeated:
                if key not in seen:
                    seen.add(key)
                    kept.append(line)
                continue
            kept.append(line)
        cleaned.append("\n".join(kept))
    return cleaned