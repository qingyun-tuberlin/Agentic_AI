"""Web search tool for MLE-STAR agents.

Restores the search capability that the upstream Google implementation
provided via `google.adk.tools.google_search_tool.google_search` (Gemini's
built-in Google Search tool), which is unavailable on the GWDG OpenAI-
compatible endpoint. Uses DuckDuckGo via the `ddgs` package — no API key
required.
"""

from __future__ import annotations

from google.adk.tools import FunctionTool


def web_search(query: str, num_results: int = 5) -> str:
    """Search the web and return short text snippets.

    Use this when you need recent information that may not be in your
    training data, such as state-of-the-art ML models for a specific
    task, or to look up an error message while debugging.

    Args:
        query: The search query string.
        num_results: Number of results to return (1-10).

    Returns:
        A newline-delimited string of "[N] Title — URL\\n    snippet" entries,
        or an explanatory error message if the search failed.
    """
    num_results = max(1, min(int(num_results), 10))
    try:
        from ddgs import DDGS
    except ImportError:
        return "web_search unavailable: install the `ddgs` package."

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))
    except Exception as exc:
        return f"web_search failed: {exc!r}"

    if not results:
        return f"web_search returned no results for query: {query!r}"

    lines = []
    for i, r in enumerate(results, start=1):
        title = r.get("title", "").strip()
        href = r.get("href") or r.get("url") or ""
        body = (r.get("body") or "").strip().replace("\n", " ")
        lines.append(f"[{i}] {title} — {href}\n    {body}")
    return "\n".join(lines)


web_search_tool = FunctionTool(func=web_search)
