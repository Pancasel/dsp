"""Small text helpers for the Markdown reports."""


def build_markdown_table(rows: list[dict[str, str]], columns: list[tuple[str, str]]) -> str:
    """columns: (key in the row, column title), in display order."""
    lines = [
        "| " + " | ".join(title for _, title in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |")
    return "\n".join(lines)


def format_as_percentage(value: str | float) -> str:
    """0.123 -> "12.3%"; text that is not a number is returned unchanged."""
    try:
        return f"{float(value):.1%}"
    except ValueError:
        return value
