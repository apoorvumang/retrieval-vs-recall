#!/usr/bin/env python3
"""Standalone WideSearch runner for staged deep-search loops.

A small, self-contained harness for reproducing WideSearch runs on any
OpenAI-compatible model. It supports two execution paths:

- ``endpoint`` posts requests to a running ``/search`` endpoint.
- ``python-loop`` runs a lightweight staged loop: plan search queries, call the
  Exa Search API, accumulate stable sources, then produce a final answer from
  the evidence. Pass ``--closed-book`` to skip search and answer from memory.

Grade the resulting ``results.jsonl`` with ``lib/official_grade.py`` (the
unchanged official ByteDance grader).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import re
import shlex
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

try:
    import aiohttp
except ImportError:  # pragma: no cover - exercised by external minimal installs.
    aiohttp = None  # type: ignore[assignment]

URL_ARRAY_SCHEMA: dict[str, Any] = {"type": "array", "items": {"type": "string"}, "minItems": 1}
ENTITY_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["entities"],
    "additionalProperties": False,
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "name",
                    "entity_identity",
                    "entity_identity_urls",
                    "entity_identity_evidence",
                    "qualifying_criteria",
                    "qualifying_criteria_urls",
                    "qualifying_criteria_evidence",
                ],
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "entity_identity": {"type": "string"},
                    "entity_identity_urls": URL_ARRAY_SCHEMA,
                    "entity_identity_evidence": {"type": "string"},
                    "qualifying_criteria": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "qualifying_criteria_urls": URL_ARRAY_SCHEMA,
                    "qualifying_criteria_evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                },
            },
        },
    },
}

DEFAULT_STRUCTURED_DATASET = "widesearch-rubric-structured-entities-qualifying-criteria-v1-100"
DEFAULT_TABLE_DATASET = "widesearch-en"
PUBLIC_WIDESEARCH_REPO = "ByteDance-Seed/WideSearch"
DEFAULT_EXA_BASE_URL = "https://api.exa.ai"
DEFAULT_ENDPOINT_URL = "http://127.0.0.1:3337/search"
STANDALONE_STAGE = "standalone-deep-widesearch"


@dataclass
class Query:
    """Small local query record with the fields the loop needs."""

    query: str
    source: str
    expected: str | dict | list | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class HarnessConfig:
    """One Deep-style searcher configuration."""

    name: str = "mercury2_r5_f2"
    variant: str = "deep_staged_v2"
    model: str = "mercury-2"
    model_route: str = "inception"
    reasoning_effort: str | None = "medium"
    search_type: str = "fast"
    strategy: str = "fixed"
    max_searches: int = 5
    fanout: int = 2
    num_results: int = 10
    highlight_max_characters: int = 2_000
    contents_max_characters: int | None = None
    max_contents: int | None = 0
    supersnippets: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceRecord:
    """Stable source identity for a retrieved URL."""

    source_id: int
    url: str
    title: str
    first_seen_round: int
    document: dict[str, Any]


@dataclass
class EvidenceEntry:
    """One source occurrence under one search query."""

    round_number: int
    query: str
    source_id: int
    url: str
    title: str
    published_date: str | None
    result_rank: int
    highlights: list[str]


@dataclass
class DeepLoopState:
    """Mutable accumulated evidence for the Python Deep-style loop."""

    original_query: str
    initial_transcript: str
    prompt_transcript: str
    sources_by_url: dict[str, SourceRecord] = field(default_factory=dict)
    ordered_urls: list[str] = field(default_factory=list)
    evidence_entries: list[EvidenceEntry] = field(default_factory=list)


def json_dumps(value: Any) -> str:
    """Serialize compact JSONL-safe data with stable Unicode handling."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def json_dumps_pretty(value: Any) -> str:
    """Serialize human-readable JSON without escaping non-ASCII evidence text."""
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def parse_json_object(text: str) -> Any:
    """Parse a JSON object from raw model text, with fenced/block fallbacks."""
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty JSON response")

    candidates = [stripped]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.append(fenced.group(1).strip())

    first_obj = stripped.find("{")
    last_obj = stripped.rfind("}")
    if first_obj >= 0 and last_obj > first_obj:
        candidates.append(stripped[first_obj : last_obj + 1])

    first_arr = stripped.find("[")
    last_arr = stripped.rfind("]")
    if first_arr >= 0 and last_arr > first_arr:
        candidates.append(stripped[first_arr : last_arr + 1])

    errors = []
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(str(exc))
    raise ValueError(f"could not parse JSON response: {errors[-1] if errors else 'unknown'}")


def clean_highlights(result: dict[str, Any]) -> list[str]:
    """Return non-empty deduped highlights, falling back to result text."""
    raw = result.get("highlights")
    candidates = raw if isinstance(raw, list) and raw else [result.get("text") or ""]
    seen: set[str] = set()
    highlights: list[str] = []
    for candidate in candidates:
        text = str(candidate).replace("\r", " ").replace("\n", " ")
        value = re.sub(r"\s+", " ", text).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        highlights.append(value)
    return highlights


def result_text(result: dict[str, Any]) -> str:
    """Pick the best citation text from an Exa or endpoint result."""
    text = result.get("text") or result.get("summary")
    if isinstance(text, str) and text.strip():
        return text
    highlights = result.get("highlights")
    if isinstance(highlights, list):
        return "\n".join(str(item) for item in highlights if str(item).strip())
    if highlights:
        return str(highlights)
    return ""


def citations_from_results(raw_results: list[Any]) -> list[dict[str, str]]:
    """Convert endpoint/raw search results into grader citation records."""
    citations: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        citations.append(
            {
                "url": url,
                "title": str(item.get("title") or ""),
                "text": result_text(item),
            }
        )
    return citations


def response_answer_for_grader(answer: Any) -> Any:
    """Preserve structured answers and stringify scalar endpoint answers."""
    if isinstance(answer, dict | list):
        return answer
    if answer is None:
        return ""
    return str(answer)


def shaped_grade_results(answer: Any, citations: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Return the single-answer result shape expected by the answer graders."""
    return [{"answer": response_answer_for_grader(answer), "citations": citations}]


def source_citations(state: DeepLoopState) -> list[dict[str, str]]:
    """Return citation records for all stable sources in first-seen order."""
    citations = []
    for url in state.ordered_urls:
        source = state.sources_by_url[url]
        citations.append(
            {
                "url": source.url,
                "title": source.title,
                "text": result_text(source.document),
            }
        )
    return citations


def required_columns_from_metadata(metadata: dict[str, Any]) -> list[str]:
    """Read legacy WideSearch table columns from query metadata."""
    direct = metadata.get("required_columns")
    if isinstance(direct, list):
        columns = [str(item).strip() for item in direct if str(item).strip()]
        if columns:
            return columns
    evaluation = metadata.get("evaluation")
    if isinstance(evaluation, dict):
        required = evaluation.get("required")
        if isinstance(required, list):
            return [str(item).strip() for item in required if str(item).strip()]
    return []


def norm_column(column: str) -> str:
    """Normalize WideSearch column names the same way as the official grader."""
    return str(column).strip().lower().replace(" ", "")


def load_gold_csv_rows(
    repo_id: str,
    instance_id: str,
    required_columns: list[str],
    *,
    answer_root: str = "widesearch_gold",
) -> tuple[list[dict[str, str]], str] | None:
    """Load one public WideSearch gold CSV from the HuggingFace cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError as exc:
        raise RuntimeError(
            "Public WideSearch loading needs huggingface_hub. Install with: "
            "pip install datasets huggingface_hub"
        ) from exc

    cache_path = try_to_load_from_cache(
        repo_id=repo_id,
        filename=f"{answer_root}/{instance_id}.csv",
        repo_type="dataset",
    )
    if cache_path is None:
        return None

    rows: list[dict[str, str]] = []
    # WideSearch gold CSVs are UTF-8 with a BOM; utf-8-sig strips it so the
    # first header (e.g. "Subject") normalizes correctly instead of "﻿subject",
    # which would mark the first required column missing and drop every row.
    with open(cache_path, encoding="utf-8-sig", newline="") as handle:
        raw_text = handle.read()
        handle.seek(0)
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return None
        column_map = {norm_column(column): column for column in reader.fieldnames}
        missing = [column for column in required_columns if column not in column_map]
        if missing:
            return None
        for row in reader:
            rows.append(
                {
                    required: str(row.get(column_map[required]) or "")
                    for required in required_columns
                }
            )
    return rows, raw_text


async def load_public_widesearch_queries(args: argparse.Namespace) -> list[Query]:
    """Load public ByteDance-Seed/WideSearch rows from Hugging Face."""
    try:
        from datasets import load_dataset
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Public WideSearch loading needs datasets and huggingface_hub. Install with: "
            "pip install datasets huggingface_hub"
        ) from exc

    repo_id = args.public_widesearch_repo
    snapshot_download(repo_id=repo_id, repo_type="dataset")
    dataset = load_dataset(repo_id)["full"]

    queries: list[Query] = []
    for item in dataset:
        if not isinstance(item, dict):
            continue
        language = str(item.get("language") or "")
        if args.language and language != args.language:
            continue

        evaluation = item.get("evaluation")
        if isinstance(evaluation, str):
            evaluation = json.loads(evaluation)
        if not isinstance(evaluation, dict):
            continue

        required_columns = [norm_column(column) for column in evaluation.get("required", [])]
        instance_id = str(item.get("instance_id") or "")
        gold = load_gold_csv_rows(repo_id, instance_id, required_columns)
        if gold is None:
            continue
        gold_rows, gold_csv = gold

        queries.append(
            Query(
                query=str(item["query"]),
                source=f"public-widesearch-{language or 'all'}",
                expected=json.dumps(gold_rows, ensure_ascii=False),
                metadata={
                    "instance_id": instance_id,
                    "language": language,
                    "evaluation": evaluation,
                    "gold_csv": gold_csv,
                    "required_columns": required_columns,
                    "unique_columns": [
                        norm_column(column) for column in evaluation.get("unique_columns", [])
                    ],
                    "eval_pipeline": {
                        norm_column(key): value
                        for key, value in evaluation.get("eval_pipeline", {}).items()
                    },
                },
                tags=[f"lang:{language}"] if language else [],
            )
        )
        if args.limit > 0 and len(queries) >= args.limit:
            break
    return queries


def format_highlights(highlights: list[str]) -> str:
    """Format highlights for the evidence transcript."""
    if not highlights:
        return "- (no highlights returned)"
    return "\n".join(f"- {highlight}" for highlight in highlights)


def format_evidence_append(round_number: int, entries: list[EvidenceEntry]) -> str:
    """Format a Deep staged-v2-like evidence append."""
    if not entries:
        return f"## Search Round {round_number}\n\nNo results returned."

    grouped: dict[str, list[EvidenceEntry]] = defaultdict(list)
    for entry in entries:
        grouped[entry.query].append(entry)

    sections = []
    for query, query_entries in grouped.items():
        formatted_entries = []
        for entry in query_entries:
            date = f"\nDate: {entry.published_date}" if entry.published_date else ""
            formatted_entries.append(
                f"""#### Source [{entry.source_id}]
URL: {entry.url}
Title: {entry.title or "(untitled)"}{date}
Highlights:
{format_highlights(entry.highlights)}"""
            )
        sections.append(f"### Query: {query}\n\n" + "\n\n".join(formatted_entries))
    return f"## Search Round {round_number}\n\n" + "\n\n".join(sections)


def rebuild_transcript(state: DeepLoopState) -> str:
    """Rebuild the transcript from the stable initial prompt plus evidence."""
    grouped: dict[int, list[EvidenceEntry]] = defaultdict(list)
    for entry in state.evidence_entries:
        grouped[entry.round_number].append(entry)
    if not grouped:
        return state.initial_transcript
    appends = [
        format_evidence_append(round_number, grouped[round_number])
        for round_number in sorted(grouped)
    ]
    return f"{state.initial_transcript}\n\n" + "\n\n".join(appends)


def build_table_instructions(metadata: dict[str, Any]) -> str:
    """Return legacy WideSearch table output guidance when columns are present."""
    required_columns = required_columns_from_metadata(metadata)
    if not required_columns:
        return ""
    return f"""## WideSearch Table Output

This is a WideSearch table task. The final answer content must be a markdown table.

Required table columns, in order: {" | ".join(required_columns)}

- Use the required column headers exactly, preserving spelling and order.
- Return no prose before or after the table content.
- Do not include citations, footnotes, source labels, or URLs inside cells unless the task specifically asks for URLs.
- If a required value is not supported by the accumulated evidence, leave that cell blank rather than guessing."""


def build_initial_transcript(
    query: Query,
    config: HarnessConfig,
    *,
    answer_shape: Literal["structured", "table"],
) -> str:
    """Build the staged-v2-inspired initial prompt transcript."""
    current_date = datetime.now(timezone.utc).date().isoformat()
    output_section = (
        """## Structured WideSearch Output

The final answer must be raw JSON only, matching the schema supplied to the final answer step.
Each entity must include URLs from retrieved sources that support identity and qualifying criteria.
Do not use outside knowledge to fill unsupported fields."""
        if answer_shape == "structured"
        else build_table_instructions(query.metadata)
    )
    return f"""You are Exa's deep search agent. You answer broad enumeration questions by iteratively searching the web, analyzing results, and refining searches until you have source-backed evidence.

## Workflow

Optimize for speed: batch queries aggressively and finish as soon as the accumulated evidence can support an answer.

1. SEARCH - Call batch_search with up to {config.fanout} queries covering the question from different angles.
2. ANALYZE - Check what is supported, what is missing, and whether the answer is ready.
3. REFINE - Only search again for genuinely missing criteria, disambiguation, or verification.
4. FINISH - Finish when evidence is sufficient or when search rounds are exhausted.

## Search Strategy

- Start with varied candidate-enumeration queries, not just one broad query.
- For WideSearch, enumerate candidates first, then verify the constraints for each candidate.
- Prefer official, primary, or list/table sources when available.
- Do not repeat exact query strings. Avoid synonym-only rewrites.
- If a source does not explicitly support a required criterion, treat that criterion as missing.

## Evidence Rules

- Base your answer only on what retrieved sources explicitly state.
- If a query asks for a name, count, date, or qualification, find direct evidence.
- Preserve uncertainty and exclusions; do not rescue missing evidence with prior knowledge.

{output_section}

## Structured Action Contract

Each planner turn must return exactly one JSON object:

- {{"action":"batch_search","queries":["query one","query two"],"reason":"why these searches are needed"}}
- {{"action":"finish","reason":"why the evidence is sufficient"}}

Original query:
{query.query}

Today's date: {current_date}

## Evidence Transcript

No searches have been run yet."""


def build_planner_prompt(
    state: DeepLoopState,
    *,
    round_number: int,
    remaining_rounds: int,
    fanout: int,
) -> tuple[str, str]:
    """Build the planner system/user prompt."""
    system = "Choose the next Deep search action. Return exactly one JSON object."
    prompt = f"""{state.prompt_transcript}

## Next Action

Search rounds used: {round_number - 1}
Search rounds remaining: {remaining_rounds}
Maximum queries in the next batch_search action: {fanout}

Return exactly one JSON object action now. Do not include prose outside JSON."""
    return system, prompt


def build_final_prompt(
    query: Query,
    state: DeepLoopState,
    *,
    answer_shape: Literal["structured", "table"],
    finish_reason: str | None,
    table_format: str = "markdown",
    closed_book: bool = False,
) -> tuple[str, str]:
    """Build the source-grounded final answer prompt (or own-knowledge if closed_book)."""
    evidence = format_evidence_append(0, state.evidence_entries).replace(
        "## Search Round 0",
        "## Accumulated Evidence",
    )
    source_list = "\n".join(
        f"[{state.sources_by_url[url].source_id}] {state.sources_by_url[url].title or '(untitled)'}: {url}"
        for url in state.ordered_urls
    )
    finish_section = (
        ""
        if not finish_reason
        else f"\n\n## Research Conclusion From Search Phase\n\n{finish_reason.strip()}"
    )
    if answer_shape == "structured":
        instructions = f"""Return raw JSON only. The JSON must match this schema:
{json_dumps_pretty(ENTITY_OUTPUT_SCHEMA)}

Rules:
- Include only entities supported by retrieved sources.
- entity_identity_urls and qualifying_criteria_urls must contain source URLs from the stable source list.
- entity_identity_evidence and qualifying_criteria_evidence must be source snippets or close paraphrases.
- qualifying_criteria should list concrete criteria the entity satisfies.
- Do not include markdown, comments, citations in bracket form, or wrapper keys outside the schema."""
    elif table_format == "json":
        columns = required_columns_from_metadata(query.metadata)
        instructions = f"""Return raw JSON only, of the form: {{"rows": [{{"column": "value", ...}}, ...]}}

- Each row object must use exactly these keys, in this order: {" | ".join(columns)}
- Emit one row object per item the query asks to enumerate; do not stop at a partial list.
- Include every key in every row; use "" when a value is not supported by the evidence (do not omit keys or rows).
- Use only the accumulated evidence. Do not include markdown, prose, or any keys outside the schema."""
    else:
        table_instructions = build_table_instructions(query.metadata)
        instructions = f"""{table_instructions}

Use only the accumulated evidence. Return the markdown table only."""

    if closed_book:
        # No retrieval: permit answering from the model's own/parametric knowledge.
        instructions = (
            instructions
            .replace("Use only the accumulated evidence.",
                     "Answer from your own knowledge — no sources are provided. Give your best known value rather than leaving blanks.")
            .replace('use "" when a value is not supported by the evidence (do not omit keys or rows)',
                     "give your best known value (do not omit keys or rows)")
            .replace("- Include only entities supported by retrieved sources.",
                     "- Include entities from your own knowledge; no sources are provided.")
        )
    system = (
        "You are a meticulous data assistant. Produce the requested table from your own knowledge; "
        "no documents are provided. Fill every required column for every row you can recall."
        if closed_book else
        "You are the final answer model for Deep search. Answer only from the accumulated evidence."
    )
    prompt = f"""Original query:
{query.query}

{evidence}
{finish_section}

## Stable Source List

{source_list or "No sources were retrieved."}

## Final Answer Instructions

{instructions}"""
    return system, prompt


def normalized_config_item(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize camelCase endpoint config keys to Python dataclass names."""
    aliases = {
        "modelRoute": "model_route",
        "reasoningEffort": "reasoning_effort",
        "searchType": "search_type",
        "maxSearches": "max_searches",
        "numResults": "num_results",
        "highlightMaxCharacters": "highlight_max_characters",
        "contentsMaxCharacters": "contents_max_characters",
        "maxContents": "max_contents",
    }
    normalized: dict[str, Any] = {}
    for key, value in raw.items():
        target_key = aliases.get(key, key)
        if target_key == "reasoning" and isinstance(value, dict):
            normalized["reasoning_effort"] = value.get("effort")
            continue
        if target_key == "search" and isinstance(value, dict):
            normalized.update(normalized_config_item(value))
            continue
        normalized[target_key] = value
    return normalized


def parse_configs(args: argparse.Namespace) -> list[HarnessConfig]:
    """Read one or more harness configs from CLI defaults plus optional JSON."""
    base = HarnessConfig(
        name=args.name,
        variant=args.variant,
        model=args.model,
        model_route=args.model_route,
        reasoning_effort=args.reasoning_effort,
        search_type=args.search_type,
        strategy=args.strategy,
        max_searches=args.max_searches,
        fanout=args.fanout,
        num_results=args.num_results,
        highlight_max_characters=args.highlight_max_characters,
        contents_max_characters=args.contents_max_characters,
        max_contents=args.max_contents,
        supersnippets=args.supersnippets,
        metadata={},
    )
    if not args.configs_json:
        return [base]

    configs_arg = args.configs_json.strip()
    if configs_arg.startswith("["):
        raw_text = configs_arg
    else:
        raw_text = Path(configs_arg).read_text(encoding="utf-8")
    parsed = json.loads(raw_text)
    if not isinstance(parsed, list):
        raise ValueError("--configs-json must be a JSON list or a path to one")

    configs = []
    valid_fields = set(HarnessConfig.__dataclass_fields__)
    for item in parsed:
        if not isinstance(item, dict):
            raise ValueError("each --configs-json entry must be an object")
        normalized = normalized_config_item(item)
        unknown = sorted(set(normalized) - valid_fields)
        if unknown:
            raise ValueError(f"unknown config fields: {unknown}")
        configs.append(replace(base, **normalized))
    return configs


def query_from_json_record(record: dict[str, Any], *, default_source: str) -> Query:
    """Create a Query from a local JSONL record."""
    query_text = record.get("query")
    if not isinstance(query_text, str) or not query_text.strip():
        raise ValueError("query JSONL records must contain a non-empty string 'query'")
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    tags = record.get("tags") if isinstance(record.get("tags"), list) else []
    return Query(
        query=query_text,
        source=str(record.get("source") or default_source),
        expected=record.get("expected"),
        metadata=metadata,
        tags=[str(tag) for tag in tags],
    )


async def load_queries(args: argparse.Namespace) -> list[Query]:
    """Load query rows from a local JSONL file or the public WideSearch dataset."""
    if args.source == "jsonl":
        if not args.query_jsonl:
            raise ValueError("--source jsonl requires --query-jsonl")
        queries = []
        with Path(args.query_jsonl).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{args.query_jsonl}:{line_number}: invalid JSON") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"{args.query_jsonl}:{line_number}: expected JSON object")
                queries.append(query_from_json_record(record, default_source=args.dataset))
        return queries[: args.limit] if args.limit > 0 else queries

    if args.source == "public-widesearch":
        return await load_public_widesearch_queries(args)

    raise ValueError(
        f"unsupported source: {args.source!r}. "
        "This release supports --source public-widesearch or --source jsonl."
    )


def endpoint_request_body(
    query: Query,
    config: HarnessConfig,
    *,
    experiment_id: str,
    answer_shape: Literal["structured", "table"],
    pass_expected: bool,
) -> dict[str, Any]:
    """Build an endpoint request body."""
    body: dict[str, Any] = {
        "query": query.query,
        "variant": config.variant,
        "model": config.model,
        "modelRoute": config.model_route,
        "search": {
            "type": config.search_type,
            "strategy": config.strategy,
            "maxSearches": config.max_searches,
            "fanout": config.fanout,
            "numResults": config.num_results,
            "highlightMaxCharacters": config.highlight_max_characters,
            "supersnippets": config.supersnippets,
        },
        "experimentId": experiment_id,
        "metadata": {
            "stage": STANDALONE_STAGE,
            "harness": config.variant,
            "searcher": config.name,
            **config.metadata,
            **query.metadata,
        },
    }
    if config.contents_max_characters is not None:
        body["search"]["contentsMaxCharacters"] = config.contents_max_characters
    if config.max_contents is not None:
        body["search"]["maxContents"] = config.max_contents
    if config.reasoning_effort:
        body["reasoning"] = {"effort": config.reasoning_effort}
    if answer_shape == "structured":
        body["outputSchema"] = ENTITY_OUTPUT_SCHEMA
    if pass_expected and query.expected is not None:
        body["expected"] = query.expected
    return body


async def post_json(
    session: aiohttp.ClientSession,
    url: str,
    *,
    body: dict[str, Any],
    headers: dict[str, str],
) -> tuple[int, Any, int]:
    """POST JSON and return status, parsed payload/raw text, and latency."""
    started = time.perf_counter()
    async with session.post(url, headers=headers, json=body) as response:
        text = await response.text()
        latency_ms = int((time.perf_counter() - started) * 1000)
        try:
            data: Any = json.loads(text)
        except json.JSONDecodeError:
            data = {"raw": text}
        return response.status, data, latency_ms


async def run_endpoint_case(
    session: aiohttp.ClientSession,
    query: Query,
    config: HarnessConfig,
    args: argparse.Namespace,
    *,
    query_index: int,
    experiment_id: str,
) -> dict[str, Any]:
    """Run one query against a search endpoint."""
    body = endpoint_request_body(
        query,
        config,
        experiment_id=experiment_id,
        answer_shape=args.answer_shape,
        pass_expected=args.pass_expected,
    )
    headers = {"Content-Type": "application/json"}
    if args.endpoint_api_key_env and os.environ.get(args.endpoint_api_key_env):
        headers["Authorization"] = f"Bearer {os.environ[args.endpoint_api_key_env]}"

    status, data, latency_ms = await post_json(
        session,
        args.endpoint_url,
        body=body,
        headers=headers,
    )
    base_row = {
        "run_id": experiment_id,
        "runner": "endpoint",
        "searcher": config.name,
        "query_index": query_index,
        "query": query.query,
        "query_source": query.source,
        "query_metadata": query.metadata,
        "query_expected": query.expected,
        "request": body if args.write_requests else None,
        "status": status,
        "latency_ms": latency_ms,
    }
    if status >= 400:
        return {
            **base_row,
            "ok": False,
            "error": data,
            "answer": None,
            "raw_results": [],
            "grade_results": [],
            "search_metadata": {},
            "metrics": {},
        }

    raw_results = data.get("results", []) if isinstance(data, dict) else []
    raw_results = raw_results if isinstance(raw_results, list) else []
    answer = data.get("answer") if isinstance(data, dict) else None
    citations = citations_from_results(raw_results)
    metrics = data.get("metrics", {}) if isinstance(data, dict) else {}
    metrics = metrics if isinstance(metrics, dict) else {}
    search_metadata = {
        "rag_search_results": citations,
        "endpoint_metrics": metrics,
        "http_response": data if args.write_http_response else {"status": status},
    }
    return {
        **base_row,
        "ok": True,
        "error": None,
        "answer": response_answer_for_grader(answer),
        "raw_results": raw_results,
        "grade_results": shaped_grade_results(answer, citations),
        "search_metadata": search_metadata,
        "metrics": metrics,
    }


def exa_search_body(query: str, config: HarnessConfig) -> dict[str, Any]:
    """Build a public Exa /search request body."""
    return {
        "query": query,
        "type": config.search_type,
        "numResults": config.num_results,
        "contents": {
            "highlights": {
                "maxCharacters": config.highlight_max_characters,
            },
        },
    }


def normalize_search_results(raw_results: Any, query: str) -> list[dict[str, Any]]:
    """Normalize search-provider results into the evidence document shape."""
    if not isinstance(raw_results, list):
        return []
    documents = []
    for index, item in enumerate(raw_results):
        if not isinstance(item, dict):
            continue
        documents.append(
            {
                "url": str(item.get("url") or ""),
                "title": str(item.get("title") or ""),
                "author": item.get("author"),
                "text": result_text(item),
                "highlights": clean_highlights(item),
                "score": item.get("score"),
                "publishedDate": item.get("publishedDate") or item.get("published_date"),
                "entities": item.get("entities"),
                "rank": int(item.get("rank") if item.get("rank") is not None else index),
                "sourceQuery": query,
            }
        )
    return documents


async def search_exa(
    session: aiohttp.ClientSession,
    query: str,
    config: HarnessConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run one Exa search request and normalize documents."""
    api_key = os.environ.get(args.exa_api_key_env)
    if not api_key:
        raise RuntimeError(f"{args.exa_api_key_env} is not set")
    # Retry on 429 (rate-limit) / 5xx (transient) with exponential backoff + jitter.
    # BUGFIX (2026-06-24): previously a single 429 failed the whole task -> fast/
    # concurrent instant runs collapsed with ~40-80% "no-answer" that were really
    # Exa rate-limits, not model failures. Retrying makes runs robust to bursts.
    max_retries = 6
    status, data, latency_ms = 0, None, 0
    for attempt in range(max_retries):
        status, data, latency_ms = await post_json(
            session,
            f"{args.exa_base_url.rstrip('/')}/search",
            body=exa_search_body(query, config),
            headers={"Content-Type": "application/json", "x-api-key": api_key},
        )
        if status == 429 or status >= 500:
            if attempt < max_retries - 1:
                await asyncio.sleep(min(30.0, 2.0 ** attempt) + random.uniform(0, 1.5))
                continue
        break
    if status >= 400:
        raise RuntimeError(f"Exa /search failed with status {status}: {json_dumps(data)[:500]}")
    if not isinstance(data, dict):
        raise RuntimeError("Exa /search returned a non-object payload")

    return {
        "documents": normalize_search_results(data.get("results", []), query),
        "latencyMs": latency_ms,
        "costUsd": data.get("costDollars"),
        "requestId": data.get("requestId"),
        "resolvedSearchType": data.get("resolvedSearchType"),
        "searchTime": data.get("searchTime"),
    }


async def search_with_command(query: str, config: HarnessConfig, args: argparse.Namespace) -> dict[str, Any]:
    """Run an external search command that reads request JSON on stdin."""
    if not args.search_command:
        raise RuntimeError("--search-provider command requires --search-command")

    payload = {
        "query": query,
        "type": config.search_type,
        "num_results": config.num_results,
        "highlight_max_characters": config.highlight_max_characters,
    }
    started = time.perf_counter()
    process = await asyncio.create_subprocess_exec(
        *shlex.split(args.search_command),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate(json_dumps(payload).encode("utf-8"))
    latency_ms = int((time.perf_counter() - started) * 1000)
    if process.returncode != 0:
        raise RuntimeError(
            f"search command failed with code {process.returncode}: "
            f"{stderr.decode('utf-8', errors='replace')[:500]}"
        )

    text = stdout.decode("utf-8", errors="replace")
    data = parse_json_object(text)
    raw_results = data.get("results", data) if isinstance(data, dict) else data
    return {
        "documents": normalize_search_results(raw_results, query),
        "latencyMs": latency_ms,
        "costUsd": data.get("cost") if isinstance(data, dict) else None,
        "requestId": data.get("request_id") if isinstance(data, dict) else None,
        "resolvedSearchType": args.search_provider,
        "searchTime": None,
    }


async def run_search(
    session: aiohttp.ClientSession,
    query: str,
    config: HarnessConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Dispatch one search query to Exa or an external command."""
    if args.search_provider == "exa":
        return await search_exa(session, query, config, args)
    if args.search_provider == "command":
        return await search_with_command(query, config, args)
    raise RuntimeError(f"unsupported search provider: {args.search_provider}")


def append_search_outputs(
    state: DeepLoopState,
    *,
    round_number: int,
    outputs: list[dict[str, Any]],
    queries: list[str],
) -> tuple[list[EvidenceEntry], int]:
    """Append search outputs to state using stable source IDs by URL."""
    entries: list[EvidenceEntry] = []
    new_result_count = 0
    next_source_id = len(state.ordered_urls) + 1

    for output_index, output in enumerate(outputs):
        search_query = queries[output_index]
        for document in output["documents"]:
            url = str(document.get("url") or "").strip()
            if not url:
                continue
            source = state.sources_by_url.get(url)
            if source is None:
                source = SourceRecord(
                    source_id=next_source_id,
                    url=url,
                    title=str(document.get("title") or ""),
                    first_seen_round=round_number,
                    document=document,
                )
                state.sources_by_url[url] = source
                state.ordered_urls.append(url)
                next_source_id += 1
                new_result_count += 1
            entries.append(
                EvidenceEntry(
                    round_number=round_number,
                    query=search_query,
                    source_id=source.source_id,
                    url=url,
                    title=str(document.get("title") or source.title),
                    published_date=document.get("publishedDate"),
                    result_rank=int(document.get("rank") or 0),
                    highlights=clean_highlights(document),
                )
            )

    state.evidence_entries.extend(entries)
    state.prompt_transcript = rebuild_transcript(state)
    return entries, new_result_count


def usage_dict(usage: Any) -> dict[str, int]:
    """Normalize OpenAI usage objects/dicts to a small token map."""
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        raw = usage.model_dump()
    elif isinstance(usage, dict):
        raw = usage
    else:
        raw = {}
    output: dict[str, int] = {}
    for source, target in (
        ("prompt_tokens", "input"),
        ("completion_tokens", "output"),
        ("total_tokens", "total"),
    ):
        value = raw.get(source)
        if isinstance(value, int | float) and not isinstance(value, bool):
            output[target] = int(value)
    return output


class PythonDeepLoop:
    """Small Python implementation of the staged Deep search loop."""

    def __init__(self, args: argparse.Namespace):
        """Create an OpenAI-compatible client from CLI/env settings."""
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "--runner python-loop needs openai. Install with: pip install openai"
            ) from exc
        raw_key = os.environ.get(args.llm_api_key_env)
        if not raw_key:
            raise RuntimeError(f"{args.llm_api_key_env} is not set")
        self.args = args
        # Allow a comma-separated list of keys -> one client per key, round-robined
        # per call (spreads load across e.g. Gemini per-key rate limits during a
        # teacher rollout). A single key behaves exactly as before (one client).
        keys = [k.strip() for k in raw_key.split(",") if k.strip()]
        self.clients = []
        for key in keys:
            client_kwargs = {"api_key": key}
            if args.llm_base_url:
                client_kwargs["base_url"] = args.llm_base_url
            self.clients.append(AsyncOpenAI(**client_kwargs))
        self.client = self.clients[0]
        self._client_rr = 0

        # Optional separate client(s) for the synthesis (answer) step.
        self.synth_clients = None
        self._synth_rr = 0
        if getattr(args, "synth_model", None):
            synth_key_env = getattr(args, "synth_api_key_env", None) or args.llm_api_key_env
            synth_raw = os.environ.get(synth_key_env)
            if not synth_raw:
                raise RuntimeError(f"{synth_key_env} is not set (needed for --synth-model)")
            self.synth_clients = []
            for key in [k.strip() for k in synth_raw.split(",") if k.strip()]:
                ckw = {"api_key": key}
                if getattr(args, "synth_base_url", None):
                    ckw["base_url"] = args.synth_base_url
                self.synth_clients.append(AsyncOpenAI(**ckw))

    async def call_json(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        max_tokens: int,
        force_json: bool = True,
        is_synth: bool = False,
    ) -> tuple[Any, dict[str, Any]]:
        """Call an OpenAI-compatible chat model.

        force_json=True (planner, structured answers) requests a JSON object and
        parses it. force_json=False (table answers) returns the raw text: a
        markdown table is free-form text, so forcing response_format=json_object
        and JSON-parsing it would coerce a valid table to an empty/failed answer.
        """
        started = time.perf_counter()
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        # OpenAI reasoning models (gpt-5*, o3*, o4*) reject `max_tokens` and non-default
        # `temperature`: use `max_completion_tokens` and omit temperature for them.
        _is_openai_reasoning = str(model).lower().startswith(("gpt-5", "o3", "o4"))
        if _is_openai_reasoning:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
            if self.args.temperature is not None:
                kwargs["temperature"] = self.args.temperature
        if force_json and not self.args.no_response_format:
            kwargs["response_format"] = {"type": "json_object"}
        # extra_body: merge any user-supplied JSON with reasoning_effort.
        # BUGFIX (2026-06-23): the python-loop runner previously NEVER forwarded
        # reasoning_effort, so the gpt-oss chat template fell back to its hard-coded
        # default ("medium") for EVERY call, regardless of --reasoning-effort. So the
        # "instant" arms were actually served at medium. Forward it now (vLLM/litellm
        # convention in this codebase = top-level `reasoning_effort`, cf. frameworks/
        # api_workers/rest.py). Default "medium" keeps prior medium runs identical.
        extra_body: dict[str, Any] = (
            json.loads(self.args.llm_extra_body_json) if self.args.llm_extra_body_json else {}
        )
        # Synth step uses its own reasoning-effort knob when configured; "" means OFF
        # (send nothing). Falls back to the shared --reasoning-effort otherwise.
        if is_synth and getattr(self.args, "synth_model", None):
            _eff = getattr(self.args, "synth_reasoning_effort", None)
            if _eff:
                extra_body.setdefault("reasoning_effort", _eff)
        elif getattr(self.args, "reasoning_effort", None):
            extra_body.setdefault("reasoning_effort", self.args.reasoning_effort)
        if extra_body:
            kwargs["extra_body"] = extra_body

        # AUDIT: dump the FULL set of call hyperparams once per call-type (json planner
        # vs free-text answer) so we can rigorously verify nothing else is mis-passed.
        if not hasattr(self, "_logged_cfgs"):
            self._logged_cfgs = set()
        if force_json not in self._logged_cfgs:
            self._logged_cfgs.add(force_json)
            _audit = {
                k: (f"[{len(v)} msgs: " + ", ".join(f"{m['role']}={len(m['content'])}ch" for m in v) + "]")
                if k == "messages" else v
                for k, v in kwargs.items()
            }
            print(
                f"[CALL-CONFIG purpose={'planner/json' if force_json else 'answer/table'}] "
                f"base_url={self.args.llm_base_url} {json.dumps(_audit, default=str, sort_keys=True)}",
                file=sys.stderr, flush=True,
            )

        if is_synth and self.synth_clients:
            client = self.synth_clients[self._synth_rr % len(self.synth_clients)]
            self._synth_rr += 1
        else:
            client = self.clients[self._client_rr % len(self.clients)]
            self._client_rr += 1
        response = await client.chat.completions.create(**kwargs)
        latency_ms = int((time.perf_counter() - started) * 1000)
        msg = response.choices[0].message
        text = msg.content or ""
        # Capture the reasoning channel (this serve returns it under `reasoning`; some return
        # `reasoning_content`). Non-standard field -> check attr and model_extra for both names.
        _ex = getattr(msg, "model_extra", None) or {}
        reasoning = (getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
                     or _ex.get("reasoning") or _ex.get("reasoning_content"))
        result = parse_json_object(text) if force_json else text.strip()
        metrics = {
            "latencyMs": latency_ms,
            "tokenUsage": usage_dict(response.usage),
            "rawText": text if self.args.write_llm_raw_text else None,
            "reasoningText": reasoning if self.args.write_llm_raw_text else None,
            # Persist the exact prompt so each llm_call is a complete SFT
            # (system, user, assistant[, reasoning]) example for teacher-rollout datagen.
            "system": system if self.args.write_llm_raw_text else None,
            "userPrompt": prompt if self.args.write_llm_raw_text else None,
        }
        return result, metrics

    async def run(
        self,
        session: aiohttp.ClientSession,
        query: Query,
        config: HarnessConfig,
    ) -> tuple[Any, list[dict[str, str]], dict[str, Any], list[dict[str, Any]]]:
        """Run the staged loop and return answer, citations, metrics, and search results."""
        initial = build_initial_transcript(query, config, answer_shape=self.args.answer_shape)
        state = DeepLoopState(
            original_query=query.query,
            initial_transcript=initial,
            prompt_transcript=initial,
        )
        searched_queries: set[str] = set()
        llm_calls: list[dict[str, Any]] = []
        search_steps: list[dict[str, Any]] = []
        finish_reason: str | None = None

        _max_rounds = 0 if getattr(self.args, "closed_book", False) else config.max_searches
        for round_number in range(1, _max_rounds + 1):
            remaining = config.max_searches - round_number + 1
            system, prompt = build_planner_prompt(
                state,
                round_number=round_number,
                remaining_rounds=remaining,
                fanout=config.fanout,
            )
            action, metrics = await self.call_json(
                model=config.model,
                system=system,
                prompt=prompt,
                max_tokens=self.args.planner_max_tokens,
            )
            llm_calls.append({"purpose": "search_plan", "round": round_number, **metrics})
            if not isinstance(action, dict):
                finish_reason = "planner returned non-object JSON"
                break

            action_name = str(action.get("action") or action.get("type") or "").strip()
            if action_name == "finish":
                finish_reason = str(action.get("reason") or "planner chose finish")
                break
            if action_name != "batch_search":
                finish_reason = f"planner returned unsupported action: {action_name!r}"
                break

            raw_queries = action.get("queries")
            if not isinstance(raw_queries, list):
                finish_reason = "planner returned batch_search without queries"
                break
            deduped = []
            duplicates = []
            for raw_query in raw_queries:
                search_query = str(raw_query).strip()
                if not search_query:
                    continue
                if search_query in searched_queries:
                    duplicates.append(search_query)
                    continue
                searched_queries.add(search_query)
                deduped.append(search_query)
                if len(deduped) >= config.fanout:
                    break

            if not deduped:
                finish_reason = "planner produced no new search queries"
                break

            search_started = time.perf_counter()
            outputs = await asyncio.gather(
                *(run_search(session, search_query, config, self.args) for search_query in deduped)
            )
            entries, new_result_count = append_search_outputs(
                state,
                round_number=round_number,
                outputs=outputs,
                queries=deduped,
            )
            search_latency_ms = int((time.perf_counter() - search_started) * 1000)
            search_steps.append(
                {
                    "stepNumber": round_number,
                    "queries": deduped,
                    "exactDuplicateQueries": duplicates,
                    "latencyMs": search_latency_ms,
                    "resultCount": sum(len(output["documents"]) for output in outputs),
                    "newResultCount": new_result_count,
                    "entryCount": len(entries),
                    "requestIds": [
                        output["requestId"] for output in outputs if output.get("requestId")
                    ],
                }
            )

        system, prompt = build_final_prompt(
            query,
            state,
            answer_shape=self.args.answer_shape,
            finish_reason=finish_reason,
            table_format=self.args.table_format,
            closed_book=getattr(self.args, "closed_book", False),
        )
        # Table answers are markdown free-text (parse as raw); JSON-rows and
        # structured answers are JSON (force + parse).
        force_json = self.args.answer_shape == "structured" or self.args.table_format in ("json", "forcejson")
        synth_model = getattr(self.args, "synth_model", None) or config.model
        answer_alt = None
        if getattr(self.args, "dual_synth", False) and getattr(self.args, "synth_model", None):
            # Synthesis A/B on IDENTICAL evidence+prompt: base model (config.model)
            # writes `answer`; synth model writes `answer_alt`. Isolates the
            # synthesis step — both see exactly the same retrieved evidence.
            answer, answer_metrics = await self.call_json(
                model=config.model, system=system, prompt=prompt,
                max_tokens=self.args.answer_max_tokens, force_json=force_json,
                is_synth=False,
            )
            llm_calls.append({"purpose": "answer", "model": config.model, **answer_metrics})
            answer_alt, alt_metrics = await self.call_json(
                model=synth_model, system=system, prompt=prompt,
                max_tokens=self.args.answer_max_tokens, force_json=force_json,
                is_synth=True,
            )
            llm_calls.append({"purpose": "answer_alt", "model": synth_model, **alt_metrics})
        else:
            answer, answer_metrics = await self.call_json(
                model=synth_model, system=system, prompt=prompt,
                max_tokens=self.args.answer_max_tokens, force_json=force_json,
                is_synth=True,
            )
            llm_calls.append({"purpose": "answer", "model": synth_model, **answer_metrics})

        citations = source_citations(state)
        metrics = {
            "variant": "python_staged_loop",
            "model": config.model,
            "searchRoundCount": len(search_steps),
            "searchCount": sum(len(step["queries"]) for step in search_steps),
            "llmCallCount": len(llm_calls),
            "llmCalls": llm_calls,
            "searchSteps": search_steps,
            "searchCountDeduped": len(searched_queries),
            "finishReason": finish_reason,
            "sourceCount": len(state.ordered_urls),
            "answer_alt": answer_alt,
        }
        raw_results = [source.document for source in state.sources_by_url.values()]
        return answer, citations, metrics, raw_results


async def run_python_loop_case(
    session: aiohttp.ClientSession,
    loop: PythonDeepLoop,
    query: Query,
    config: HarnessConfig,
    args: argparse.Namespace,
    *,
    query_index: int,
    experiment_id: str,
) -> dict[str, Any]:
    """Run one query through the pure Python loop."""
    started = time.perf_counter()
    base_row = {
        "run_id": experiment_id,
        "runner": "python-loop",
        "searcher": config.name,
        "query_index": query_index,
        "query": query.query,
        "query_source": query.source,
        "query_metadata": query.metadata,
        "query_expected": query.expected,
        "request": asdict(config) if args.write_requests else None,
    }
    try:
        answer, citations, metrics, raw_results = await loop.run(session, query, config)
    except Exception as exc:
        return {
            **base_row,
            "ok": False,
            "error": str(exc),
            "answer": None,
            "raw_results": [],
            "grade_results": [],
            "search_metadata": {},
            "metrics": {},
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }

    latency_ms = int((time.perf_counter() - started) * 1000)
    metrics["latencyMs"] = latency_ms
    search_metadata = {
        "rag_search_results": citations,
        "endpoint_metrics": metrics,
        "results": citations,
    }
    return {
        **base_row,
        "ok": True,
        "error": None,
        "answer": answer,
        "raw_results": raw_results,
        "grade_results": shaped_grade_results(answer, citations),
        "search_metadata": search_metadata,
        "metrics": metrics,
        "latency_ms": latency_ms,
    }


async def run_one_case(
    session: aiohttp.ClientSession,
    python_loop: PythonDeepLoop | None,
    query: Query,
    config: HarnessConfig,
    args: argparse.Namespace,
    *,
    query_index: int,
    experiment_id: str,
) -> dict[str, Any]:
    """Dispatch one query/config pair to the selected runner."""
    if args.runner == "endpoint":
        return await run_endpoint_case(
            session,
            query,
            config,
            args,
            query_index=query_index,
            experiment_id=experiment_id,
        )
    if python_loop is None:
        raise RuntimeError("python loop runner was not initialized")
    return await run_python_loop_case(
        session,
        python_loop,
        query,
        config,
        args,
        query_index=query_index,
        experiment_id=experiment_id,
    )


def serializable_row(row: dict[str, Any]) -> dict[str, Any]:
    """Drop internal null request fields before JSONL writing."""
    cleaned = dict(row)
    if cleaned.get("request") is None:
        cleaned.pop("request", None)
    return cleaned


def build_grader(args: argparse.Namespace) -> Any:
    """In-process grading is not bundled in this public release."""
    raise RuntimeError(
        "In-process --grade is not available in this public release. "
        "Run without --grade to produce results.jsonl, then grade it with "
        "lib/official_grade.py (the unchanged official ByteDance grader)."
    )


def to_minos_query(query: Query) -> Any:
    """Not used in this public release (in-process grading is disabled)."""
    raise RuntimeError(
        "In-process grading is disabled; grade results.jsonl with lib/official_grade.py."
    )


async def grade_rows(
    rows: list[dict[str, Any]],
    queries: list[Query],
    args: argparse.Namespace,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Grade successful rows and write grades.jsonl."""
    grader = build_grader(args)
    grades: list[dict[str, Any]] = []
    grade_path = output_dir / "grades.jsonl"
    with grade_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            query = to_minos_query(queries[int(row["query_index"])])
            grade_row = {
                "run_id": row["run_id"],
                "searcher": row["searcher"],
                "query_index": row["query_index"],
                "query": row["query"],
                "ok": False,
                "scores": {},
                "cost": 0.0,
                "error": None,
            }
            if not row.get("ok"):
                grade_row["error"] = row.get("error")
            else:
                try:
                    if args.grader == "structured":
                        scores, cost = await grader.grade(
                            query,
                            row["grade_results"],
                            search_metadata=row.get("search_metadata") or {},
                        )
                        details = getattr(grader, "_last_entity_details", [])
                    else:
                        scores, cost = await grader.grade(query, row["grade_results"])
                        details = None
                    grade_row.update(
                        {
                            "ok": True,
                            "scores": scores,
                            "cost": cost,
                            "details": details,
                        }
                    )
                except Exception as exc:
                    grade_row["error"] = str(exc)
            grades.append(grade_row)
            handle.write(json_dumps(grade_row) + "\n")
            handle.flush()
    return grades


def percentile(values: list[float], pct: float) -> float | None:
    """Compute a nearest-rank percentile for small standalone summaries."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil((pct / 100.0) * len(ordered)))
    return ordered[min(rank - 1, len(ordered) - 1)]


def average(values: list[float]) -> float | None:
    """Return an average or None for empty input."""
    return sum(values) / len(values) if values else None


def score_summary(grades: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    """Average numeric scores by searcher."""
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for grade in grades:
        if not grade.get("ok"):
            continue
        searcher = str(grade["searcher"])
        scores = grade.get("scores") or {}
        if not isinstance(scores, dict):
            continue
        for key, value in scores.items():
            if isinstance(value, int | float) and not isinstance(value, bool):
                grouped[searcher][key].append(float(value))
    return {
        searcher: {metric: average(values) for metric, values in metrics.items()}
        for searcher, metrics in grouped.items()
    }


def summarize_rows(
    rows: list[dict[str, Any]],
    grades: list[dict[str, Any]],
    *,
    experiment_id: str,
) -> dict[str, Any]:
    """Build a compact run summary by searcher."""
    by_searcher: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_searcher[str(row["searcher"])].append(row)

    score_averages = score_summary(grades)
    searcher_summaries = {}
    for searcher, searcher_rows in by_searcher.items():
        ok_rows = [row for row in searcher_rows if row.get("ok")]
        latencies = [
            float(row["latency_ms"])
            for row in ok_rows
            if isinstance(row.get("latency_ms"), int | float)
        ]
        answers = [row.get("answer") for row in ok_rows]
        empty_answers = sum(
            1
            for answer in answers
            if answer is None or (isinstance(answer, str) and answer.strip() == "")
        )
        json_parse_failures = 0
        for answer in answers:
            if isinstance(answer, str):
                try:
                    parse_json_object(answer)
                except Exception:
                    json_parse_failures += 1
        search_counts = [
            float(row.get("metrics", {}).get("searchCount"))
            for row in ok_rows
            if isinstance(row.get("metrics", {}).get("searchCount"), int | float)
        ]
        round_counts = [
            float(row.get("metrics", {}).get("searchRoundCount"))
            for row in ok_rows
            if isinstance(row.get("metrics", {}).get("searchRoundCount"), int | float)
        ]
        searcher_summaries[searcher] = {
            "total": len(searcher_rows),
            "ok": len(ok_rows),
            "errors": len(searcher_rows) - len(ok_rows),
            "empty_answers": empty_answers,
            "json_parse_failures": json_parse_failures,
            "latency_ms_avg": average(latencies),
            "latency_ms_p50": percentile(latencies, 50),
            "latency_ms_p90": percentile(latencies, 90),
            "latency_ms_p99": percentile(latencies, 99),
            "avg_searches": average(search_counts),
            "avg_rounds": average(round_counts),
            "scores": score_averages.get(searcher, {}),
        }
    return {
        "run_id": experiment_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "searchers": searcher_summaries,
    }


def default_output_dir() -> Path:
    """Return a timestamped output directory under standalone-runs/."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("standalone-runs") / f"deep-widesearch-{stamp}"


def auto_grader(args: argparse.Namespace) -> str:
    """Resolve the grader family from the selected answer shape."""
    if args.grader != "auto":
        return args.grader
    return "structured" if args.answer_shape == "structured" else "table"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register CLI arguments."""
    parser.add_argument("--runner", choices=["endpoint", "python-loop"], default="endpoint")
    parser.add_argument("--answer-shape", choices=["structured", "table"], default="table")
    parser.add_argument(
        "--table-format",
        choices=["markdown", "json", "forcejson"],
        default="markdown",
        help="Final answer format for table tasks: markdown (free text), json (rows {'rows':[{col:val}]}), "
        "or forcejson (markdown prompt but force response_format=json_object + JSON-parse — reproduces the "
        "original codex harness's {} finalize bug, for ablation)",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "public-widesearch", "jsonl"],
        default="auto",
        help="Where to load queries from. External users should use public-widesearch or jsonl.",
    )
    parser.add_argument("--dataset", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--public-widesearch-repo", default=PUBLIC_WIDESEARCH_REPO)
    parser.add_argument("--language", default="en", help="Public WideSearch language filter")
    parser.add_argument("--query-jsonl", help="Local JSONL query file with query/expected/metadata")
    parser.add_argument("--expected-column", default="expected")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--experiment-id", default=None)

    parser.add_argument("--name", default="mercury2_r5_f2")
    parser.add_argument("--configs-json", help="JSON list or path with per-searcher config overrides")
    parser.add_argument("--variant", default="deep_staged_v2")
    parser.add_argument("--model", default="mercury-2")
    parser.add_argument("--model-route", default="inception")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--search-type", default="fast")
    parser.add_argument("--strategy", default="fixed")
    parser.add_argument("--max-searches", type=int, default=5)
    parser.add_argument("--fanout", type=int, default=2)
    parser.add_argument("--num-results", type=int, default=10)
    parser.add_argument("--highlight-max-characters", type=int, default=2_000)
    parser.add_argument("--contents-max-characters", type=int)
    parser.add_argument("--max-contents", type=int, default=0)
    parser.add_argument("--supersnippets", action="store_true")

    parser.add_argument("--endpoint-url", default=DEFAULT_ENDPOINT_URL)
    parser.add_argument("--endpoint-api-key-env")
    parser.add_argument("--pass-expected", action="store_true")

    parser.add_argument("--exa-base-url", default=DEFAULT_EXA_BASE_URL)
    parser.add_argument("--exa-api-key-env", default="EXA_API_KEY")
    parser.add_argument("--search-provider", choices=["exa", "command"], default="exa")
    parser.add_argument(
        "--search-command",
        help=(
            "External search adapter command. It receives JSON on stdin and returns "
            '{"results":[{"url","title","highlights","text"}]} on stdout.'
        ),
    )
    parser.add_argument("--llm-base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--llm-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--llm-extra-body-json")
    # Hybrid: route the final SYNTHESIS (answer) call to a different model/endpoint
    # than the planner/search calls. If --synth-model is unset, synthesis uses the
    # same model/client as the planner (default, unchanged behavior).
    parser.add_argument("--synth-model")
    parser.add_argument("--synth-base-url")
    parser.add_argument("--synth-api-key-env")
    parser.add_argument("--synth-reasoning-effort")
    # Dual synthesis: at the answer step, call BOTH the base model and --synth-model
    # on the identical evidence/prompt; store base in `answer`, synth in
    # metrics.answer_alt. Isolates the synthesis step for A/B comparison.
    parser.add_argument("--dual-synth", action="store_true")
    parser.add_argument("--planner-max-tokens", type=int, default=1_200)
    parser.add_argument("--answer-max-tokens", type=int, default=12_000)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--no-response-format", action="store_true")
    parser.add_argument("--closed-book", action="store_true",
                        help="No retrieval: skip search, answer the final table from the model's own knowledge.")

    parser.add_argument("--grade", action="store_true")
    parser.add_argument("--grader", choices=["auto", "structured", "table"], default="auto")
    parser.add_argument("--grader-model", default="openai/gpt-5-mini")
    parser.add_argument("--grader-base-url")
    parser.add_argument("--grader-api-key-env")
    parser.add_argument("--grader-qps", type=float, default=50.0)
    parser.add_argument("--grader-concurrency", type=int, default=200)

    parser.add_argument("--write-requests", action="store_true")
    parser.add_argument("--write-http-response", action="store_true")
    parser.add_argument("--write-llm-raw-text", action="store_true")


def parse_args() -> argparse.Namespace:
    """Parse and normalize CLI args."""
    parser = argparse.ArgumentParser(
        description="Run a standalone Deep-style WideSearch harness.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_arguments(parser)
    args = parser.parse_args()

    if args.source == "auto":
        args.source = "jsonl" if args.query_jsonl else "public-widesearch"
    if args.query_jsonl and args.source != "jsonl":
        parser.error("--query-jsonl can only be used with --source auto/jsonl")
    if args.source == "jsonl" and not args.query_jsonl:
        parser.error("--source jsonl requires --query-jsonl")
    if args.source == "public-widesearch" and args.answer_shape == "structured":
        parser.error(
            "the public WideSearch dataset is table-shaped; use --answer-shape table, "
            "or provide a structured-rubric JSONL with --source jsonl"
        )
    if args.runner == "python-loop" and args.search_provider == "command" and not args.search_command:
        parser.error("--search-provider command requires --search-command")

    if args.dataset is None:
        args.dataset = DEFAULT_STRUCTURED_DATASET if args.answer_shape == "structured" else DEFAULT_TABLE_DATASET
    if args.answer_shape == "table" and args.grader_model == "openai/gpt-5-mini":
        args.grader_model = "openai/gpt-5.4-mini"
    args.grader = auto_grader(args)
    return args


async def run(args: argparse.Namespace) -> int:
    """Run the harness, optional grader, and summary writing."""
    if aiohttp is None:
        raise RuntimeError("This script needs aiohttp. Install with: pip install aiohttp")

    experiment_id = args.experiment_id or f"standalone-widesearch-{uuid4()}"
    output_dir = args.output_dir or default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = parse_configs(args)
    queries = await load_queries(args)
    if not queries:
        raise RuntimeError("no queries loaded")

    (output_dir / "run_config.json").write_text(
        json_dumps_pretty(
            {
                "run_id": experiment_id,
                "runner": args.runner,
                "answer_shape": args.answer_shape,
                "dataset": args.dataset,
                "limit": args.limit,
                "configs": [asdict(config) for config in configs],
                "grade": args.grade,
                "grader": args.grader if args.grade else None,
                "grader_model": args.grader_model if args.grade else None,
            }
        ),
        encoding="utf-8",
    )

    timeout = aiohttp.ClientTimeout(total=args.timeout, connect=20.0)
    semaphore = asyncio.Semaphore(args.concurrency)
    python_loop = PythonDeepLoop(args) if args.runner == "python-loop" else None
    rows: list[dict[str, Any]] = []
    results_path = output_dir / "results.jsonl"

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def bounded_case(query_index: int, query: Query, config: HarnessConfig):
            async with semaphore:
                return await run_one_case(
                    session,
                    python_loop,
                    query,
                    config,
                    args,
                    query_index=query_index,
                    experiment_id=experiment_id,
                )

        tasks = [
            asyncio.create_task(bounded_case(query_index, query, config))
            for config in configs
            for query_index, query in enumerate(queries)
        ]

        with results_path.open("w", encoding="utf-8") as handle:
            for task in asyncio.as_completed(tasks):
                row = await task
                rows.append(row)
                handle.write(json_dumps(serializable_row(row)) + "\n")
                handle.flush()
                status = "ok" if row.get("ok") else "error"
                print(f"{status} {row['searcher']} q{row['query_index']}: {row['query'][:100]}")

    grades: list[dict[str, Any]] = []
    if args.grade:
        grades = await grade_rows(rows, queries, args, output_dir)

    summary = summarize_rows(rows, grades, experiment_id=experiment_id)
    (output_dir / "summary.json").write_text(json_dumps_pretty(summary), encoding="utf-8")
    print(json_dumps_pretty({"output_dir": str(output_dir), **summary}))
    return 0


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    try:
        exit_code = asyncio.run(run(args))
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
