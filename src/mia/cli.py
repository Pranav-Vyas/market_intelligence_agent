"""Command line: `uv run mia --help`."""

from __future__ import annotations

import logging

import typer

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Market Gap Scout: competitor intelligence and market-gap reports.",
)


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Show debug logs.")) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


@app.command()
def run(
    market: str = typer.Argument(..., help='Market to analyze, e.g. "AI meeting assistants".'),
    offline: bool = typer.Option(
        False, "--offline", help="Use only cached pages and LLM responses (no network)."
    ),
) -> None:
    """Run the full pipeline and write report.md + run.json."""
    from mia import pipeline

    result = pipeline.run(market, offline=offline)
    stats = result.report.stats
    typer.echo(f"Report: {result.run_dir / 'report.md'}")
    typer.echo(f"LLM: {stats['llm']['requests']} requests, {stats['llm']['cache_hits']} cache hits")
    if stats["stubs"]:
        typer.echo(f"Still stubbed ({len(stats['stubs'])}): {', '.join(stats['stubs'])}")


@app.command()
def models() -> None:
    """List the Gemini model IDs your API key can use (for MIA_MODEL_BULK / MIA_MODEL_SMART)."""
    from mia.config import get_settings
    from mia.llm import GeminiBackend, LLMConfigError

    try:
        backend = GeminiBackend(get_settings().gemini_api_key)
    except LLMConfigError as err:
        typer.echo(str(err), err=True)
        raise typer.Exit(1) from err
    for name in backend.list_models():
        typer.echo(f"gemini/{name}")


@app.command("search-web")
def search_web(
    query: str = typer.Argument(..., help="Search query."),
    k: int = typer.Option(10, "-k", help="Number of results."),
) -> None:
    """Search the web (Tavily if TAVILY_API_KEY is set, otherwise ddgs)."""
    from mia.collectors.search import search

    for i, r in enumerate(search(query, k), 1):
        typer.echo(f"{i:>2}. {r.title}\n    {r.url}\n    {r.snippet[:160]}")


@app.command("check-access")
def check_access(
    markdown: bool = typer.Option(False, "--markdown", help="Print a Markdown table."),
) -> None:
    """Check which data sources work from this machine (no cache)."""
    from mia.collectors.access import as_markdown, check_all

    results = check_all()
    if markdown:
        typer.echo(as_markdown(results))
        return
    for r in results:
        typer.echo(f"{r.status:>8}  {r.source}: {r.detail}")
