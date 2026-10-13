"""Command line: `uv run mia --help`."""

from __future__ import annotations

import logging
from typing import Annotated

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


@app.command()
def discover(
    market: str = typer.Argument(..., help='Market to analyze, e.g. "AI meeting assistants".'),
    k: int = typer.Option(10, "-k", help="Number of competitors to return."),
) -> None:
    """Discover competitors and compare them with the gold list, if one exists."""
    import json

    from mia import runtime
    from mia.config import REPO_ROOT
    from mia.discovery import discover_competitors, name_key

    runtime.configure()
    found = discover_competitors(market, k=k)
    for i, c in enumerate(found, 1):
        typer.echo(f"{i:>2}. {c.name:<22} {c.mention_count} sources  {c.website}")
        typer.echo(f"    {c.one_liner}")
    usage = runtime.llm().usage
    typer.echo(f"LLM: {usage.requests} requests, {usage.cache_hits} cache hits")

    slug = "_".join(market.lower().split())
    gold_path = REPO_ROOT / "eval" / "gold" / f"competitors_{slug}.json"
    if gold_path.exists() and found:
        gold = {
            name_key(n) for n in json.loads(gold_path.read_text(encoding="utf-8"))["competitors"]
        }
        hits = [c.name for c in found if name_key(c.name) in gold]
        typer.echo(
            f"Gold list: {len(hits)}/{len(found)} found competitors are on it "
            f"(precision {len(hits) / len(found):.0%}); "
            f"{len(hits)}/{len(gold)} of the gold list found (recall {len(hits) / len(gold):.0%})"
        )


@app.command()
def index(
    market: str = typer.Argument(..., help='Market to index, e.g. "AI meeting assistants".'),
    offline: bool = typer.Option(
        False, "--offline", help="Use only cached pages and LLM responses (no network)."
    ),
) -> None:
    """Discover competitors, collect their pages and complaints, and save the search index."""
    from collections import Counter

    from mia import pipeline, runtime

    runtime.configure(offline=offline)
    competitors = pipeline.discover_competitors(market, k=runtime.settings().max_competitors)
    docs = pipeline.collect(market, competitors)
    built = pipeline.build_index(market, docs)

    counts = Counter((d.company or "(market)", d.source_type) for d in docs)
    sources = sorted({source for _, source in counts})
    typer.echo(f"{'Company':<22}" + "".join(f"{s:>10}" for s in sources))
    for company in dict.fromkeys(c for c, _ in counts):
        typer.echo(f"{company:<22}" + "".join(f"{counts[company, s]:>10}" for s in sources))
    typer.echo(
        f"Indexed {len(docs)} documents as {len(built.chunks)} chunks: {pipeline.index_dir(market)}"
    )


@app.command()
def search(
    query: str = typer.Argument(..., help="What to look for."),
    market: str | None = typer.Option(
        None, "--market", "-m", help="Market whose saved index to search (default: the latest)."
    ),
    company: str | None = typer.Option(
        None, "--company", "-c", help='Only this competitor, e.g. "Otter".'
    ),
    source: Annotated[
        list[str] | None,
        typer.Option("--source", "-s", help="Only this source type (repeatable): pricing, hn, ..."),
    ] = None,
    k: int = typer.Option(10, "-k", help="Number of results."),
    no_rerank: bool = typer.Option(False, "--no-rerank", help="Skip the cross-encoder (faster)."),
) -> None:
    """Search the saved index of a market (build it with `mia index` or `mia run`)."""
    from typing import get_args

    from mia import pipeline
    from mia.discovery import name_key
    from mia.schemas import SourceType

    try:
        slug, idx = pipeline.load_index(market)
    except FileNotFoundError as err:
        typer.echo(str(err), err=True)
        raise typer.Exit(1) from err

    if company:
        companies = sorted({ch.company for ch in idx.chunks if ch.company})
        match = [c for c in companies if name_key(c) == name_key(company)]
        match = match or [c for c in companies if company.lower() in c.lower()]
        if len(match) != 1:
            typer.echo(f"Unknown company {company!r}. Indexed: {', '.join(companies)}", err=True)
            raise typer.Exit(1)
        company = match[0]
    unknown = [s for s in source or [] if s not in get_args(SourceType)]
    if unknown:
        allowed = ", ".join(get_args(SourceType))
        typer.echo(f"Unknown source {unknown[0]!r}. Use: {allowed}", err=True)
        raise typer.Exit(1)

    hits = idx.search(
        query, k=k, company=company, source_types=source or None, rerank=not no_rerank
    )
    typer.echo(f"Index: {slug} ({len(idx.chunks)} chunks)")
    if not hits:
        typer.echo("No matching chunks.")
    for i, h in enumerate(hits, 1):
        ch = h.chunk
        body = " ".join(ch.text.split("\n", 1)[-1].split())
        where = f"{ch.company or '(market)'} · {ch.source_type} · {ch.url}"
        typer.echo(f"{i:>2}. {h.score:.3f}  {where}")
        typer.echo(f"    {body[:240]}")
