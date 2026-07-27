#!/usr/bin/env python3
"""
Build a protein-missense LoF/GoF table from public MaveDB score sets.

Important interpretation rule
-----------------------------
MaveDB provides assay scores and, for calibrated score sets, functional classes
(`normal`, `abnormal`, `indeterminate`). An `abnormal` MAVE result is not by
itself a universal loss-of-function (LoF) or gain-of-function (GoF) label.
This script assigns LoF/GoF only when directionality is supported by one of:

1. An explicit variant-level LoF/GoF annotation in the downloaded score table.
2. A user-curated score-set mapping supplied with --assay-map-csv.
3. User-curated LoF/GoF score ranges supplied in that mapping.
4. An explicit high-precision LoF/GoF phrase in score-set metadata, combined
   with a calibrated/author-provided `abnormal` functional classification.

It never infers biological direction from the sign of a score alone.

Recommended workflow
--------------------
First create a score-set catalog for review:

python mavedb_directionality_table.py \
  --all-published \
  --catalog-only \
  --score-set-catalog-csv data/mavedb_score_sets.csv \
  --out data/unused.parquet

Then curate a mapping CSV using the supplied template and build the variant table:

python mavedb_directionality_table.py \
  --all-published \
  --assay-map-csv mavedb_assay_directionality_template.csv \
  --out data/mavedb_directionality.parquet \
  --workdir mavedb_cache \
  --missense-only \
  --labeled-only \
  --summary-json data/mavedb_summary.json \
  --summary-csv data/mavedb_summary.csv \
  --score-set-catalog-csv data/mavedb_score_sets.csv \
  --errors-csv data/mavedb_errors.csv

For a smaller targeted run, repeat --score-set-urn or --search-text:

python mavedb_directionality_table.py \
  --score-set-urn urn:mavedb:00000003-a-1 \
  --score-set-urn urn:mavedb:00000005-a-1 \
  --assay-map-csv mavedb_assay_directionality_template.csv \
  --out data/mavedb_selected.parquet \
  --missense-only --labeled-only

Assay-map CSV columns
---------------------
score_set_urn               Required.
include                     Optional boolean; false excludes the score set.
abnormal_directionality     Optional: lof, gof, exclude, ambiguous, or blank.
score_column                Optional score column for threshold rules; default score.
lof_min, lof_max             Optional inclusive LoF score bounds.
gof_min, gof_max             Optional inclusive GoF score bounds.
notes                       Optional provenance/curation note.

A one-sided threshold is allowed. For example, lof_max=-1 and gof_min=1.
If both LoF and GoF ranges match the same value, the row is left ambiguous.

Requirements
------------
pip install pandas requests pyarrow
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


DEFAULT_API_BASE = "https://api.mavedb.org/api/v1"
DEFAULT_SEARCH_TERMS = (
    "gain of function",
    "loss of function",
    "activating",
    "inactivating",
    "hypermorphic",
    "hypomorphic",
)

AA3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}
AA1 = set("ACDEFGHIKLMNPQRSTVWY")

PROTEIN_SUB_RE = re.compile(
    r"p\.\(?"
    r"(?P<ref>[A-Z][a-z]{2}|[ACDEFGHIKLMNPQRSTVWY])"
    r"(?P<pos>[1-9][0-9]*)"
    r"(?P<alt>[A-Z][a-z]{2}|[ACDEFGHIKLMNPQRSTVWY])"
    r"\)?",
)

GOF_PATTERNS = (
    r"\bgain[\s_-]*of[\s_-]*function\b",
    r"\bgo[-_ ]?f\b",
    r"\bactivating(?: mutation| variant| substitution)?\b",
    r"\bhyperactiv(?:e|ity|ating)\b",
    r"\bhypermorph(?:ic)?\b",
    r"\bconstitutively active\b",
)
LOF_PATTERNS = (
    r"\bloss[\s_-]*of[\s_-]*function\b",
    r"\blo[-_ ]?f\b",
    r"\binactivating(?: mutation| variant| substitution)?\b",
    r"\bnull[- ]like\b",
    r"\bhypomorph(?:ic)?\b",
    r"\bdecreased (?:protein )?function\b",
)

ANNOTATION_COLUMN_TOKENS = (
    "classification", "class", "label", "direction", "annotation",
    "consequence", "effect", "function", "category", "phenotype", "call",
)
DIRECT_DIRECTION_COLUMN_TOKENS = (
    "directionality", "direction", "gof", "lof", "gainoffunction", "lossoffunction",
    "functionalclass", "functionclass", "variantclass", "classification", "label", "call",
)
METADATA_TEXT_KEYS = (
    "title", "description", "abstract", "method", "methods", "summary",
    "shortdescription", "experiment", "keywords", "keyword",
)


@dataclass
class DirectionCall:
    label: str = ""
    source: str = ""
    confidence: str = ""
    evidence: str = ""
    rule: str = ""


class HttpStatusError(RuntimeError):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_colname(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def safe_str_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def parse_bool(value: Any, default: bool = True) -> bool:
    text = safe_str(value).lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "y", "include", "included"}:
        return True
    if text in {"0", "false", "no", "n", "exclude", "excluded"}:
        return False
    raise ValueError(f"Unrecognized boolean value: {value!r}")


def optional_float(value: Any) -> Optional[float]:
    text = safe_str(value)
    if not text:
        return None
    number = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    return None if pd.isna(number) else float(number)


def make_session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": "mavedb-directionality-builder/1.0",
        "Accept": "application/json,text/csv,text/plain,*/*",
    })
    return session


def request(
    session: requests.Session,
    method: str,
    url: str,
    *,
    timeout: int = 120,
    **kwargs: Any,
) -> requests.Response:
    response = session.request(method, url, timeout=timeout, **kwargs)
    if response.status_code >= 400:
        snippet = response.text[:500].replace("\n", " ")
        raise HttpStatusError(
            f"{method} {url} returned HTTP {response.status_code}: {snippet}",
            response.status_code,
        )
    return response


def urn_cache_stem(urn: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", urn)


def _combine_json_values(values: Sequence[Any]) -> Any:
    """Combine multiple top-level JSON values into one API-like payload.

    MaveDB download endpoints may be served as regular JSON, JSON Lines/NDJSON,
    or a sequence of top-level JSON arrays/objects. Multiple arrays are
    flattened into one list; multiple objects become a list of objects.
    """
    if not values:
        return []
    if len(values) == 1:
        return values[0]
    if all(isinstance(value, list) for value in values):
        combined: List[Any] = []
        for value in values:
            combined.extend(value)
        return combined
    return list(values)


def parse_json_or_jsonl(text: str, *, source: str) -> Any:
    """Parse standard JSON, JSON Lines/NDJSON, or concatenated JSON values."""
    cleaned = text.lstrip("\ufeff").strip()
    if not cleaned:
        return []

    # Normal JSON is by far the common case.
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        standard_error_text = str(exc)

    # JSON Lines / NDJSON: one complete JSON value per non-empty line.
    line_values: List[Any] = []
    line_error: Optional[Exception] = None
    for line_number, line in enumerate(cleaned.splitlines(), start=1):
        candidate = line.strip().lstrip("\x1e").strip()
        if not candidate:
            continue
        try:
            line_values.append(json.loads(candidate))
        except json.JSONDecodeError as exc:
            line_error = ValueError(
                f"line {line_number} is not standalone JSON: {exc}"
            )
            line_values = []
            break
    if line_values:
        return _combine_json_values(line_values)

    # Final fallback: decode consecutive top-level values, allowing arbitrary
    # whitespace and RFC 7464 record-separator characters between records.
    decoder = json.JSONDecoder()
    values: List[Any] = []
    position = 0
    length = len(cleaned)
    try:
        while position < length:
            while position < length and (
                cleaned[position].isspace() or cleaned[position] == "\x1e"
            ):
                position += 1
            if position >= length:
                break
            value, position = decoder.raw_decode(cleaned, position)
            values.append(value)
    except json.JSONDecodeError as sequence_error:
        preview = cleaned[:300].replace("\n", "\\n")
        details = f"; JSONL fallback: {line_error}" if line_error else ""
        raise ValueError(
            f"Could not parse JSON response from {source}. "
            f"Standard JSON error: {standard_error_text}; "
            f"sequential JSON error: {sequence_error}{details}. "
            f"Response preview: {preview!r}"
        ) from sequence_error

    if values:
        return _combine_json_values(values)

    raise ValueError(f"Could not parse an empty/invalid JSON response from {source}")


def read_json_cache_or_download(
    session: requests.Session,
    url: str,
    cache_path: Path,
    *,
    force: bool,
    delay: float,
    allow_missing: bool = False,
) -> Any:
    if cache_path.exists() and cache_path.stat().st_size > 0 and not force:
        cached_text = cache_path.read_text(encoding="utf-8", errors="replace")
        payload = parse_json_or_jsonl(cached_text, source=str(cache_path))
        # Rewrite legacy JSONL/concatenated caches as canonical JSON so future
        # runs are deterministic and easier to inspect.
        try:
            json.loads(cached_text.lstrip("\ufeff").strip())
        except json.JSONDecodeError:
            cache_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        return payload

    try:
        response = request(session, "GET", url)
    except HttpStatusError as exc:
        if allow_missing and exc.status_code in {400, 404, 409, 422}:
            return []
        raise

    payload = parse_json_or_jsonl(response.text, source=url)
    ensure_dir(cache_path.parent)
    cache_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if delay > 0:
        time.sleep(delay)
    return payload


def read_csv_cache_or_download(
    session: requests.Session,
    urls: Sequence[Tuple[str, str]],
    cache_dir: Path,
    stem: str,
    *,
    force: bool,
    delay: float,
) -> Tuple[pd.DataFrame, str, Path]:
    errors: List[str] = []
    for endpoint_name, url in urls:
        cache_path = cache_dir / f"{stem}.{endpoint_name}.csv"
        text: Optional[str] = None
        if cache_path.exists() and cache_path.stat().st_size > 0 and not force:
            text = cache_path.read_text(encoding="utf-8", errors="replace")
        else:
            try:
                response = request(session, "GET", url)
                text = response.text
                if not text.strip():
                    raise RuntimeError("empty response")
                ensure_dir(cache_path.parent)
                cache_path.write_text(text, encoding="utf-8")
                if delay > 0:
                    time.sleep(delay)
            except Exception as exc:  # continue to fallback endpoint
                errors.append(f"{endpoint_name}: {exc}")
                continue
        try:
            frame = pd.read_csv(io.StringIO(text), comment="#", low_memory=False)
        except Exception as exc:
            errors.append(f"{endpoint_name}: CSV parse failed: {exc}")
            continue
        if len(frame.columns) == 0:
            errors.append(f"{endpoint_name}: no CSV columns")
            continue
        return frame, endpoint_name, cache_path
    raise RuntimeError("Could not download/parse variant data; " + " | ".join(errors))


def extract_search_items(payload: Any) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)], None
    if not isinstance(payload, dict):
        return [], None

    items: List[Dict[str, Any]] = []
    for key in ("items", "results", "scoreSets", "score_sets", "data", "records"):
        value = payload.get(key)
        if isinstance(value, list):
            items = [x for x in value if isinstance(x, dict)]
            break

    total: Optional[int] = None
    for key in ("total", "count", "totalCount", "total_count"):
        value = payload.get(key)
        if isinstance(value, int):
            total = value
            break
    return items, total


def search_score_sets(
    session: requests.Session,
    api_base: str,
    text: str,
    *,
    page_size: int,
    max_results: Optional[int],
    delay: float,
) -> List[Dict[str, Any]]:
    endpoint = f"{api_base.rstrip('/')}/score-sets/search"
    offset = 0
    results: List[Dict[str, Any]] = []
    seen_urns: set[str] = set()

    while True:
        payload = {"text": text, "limit": page_size, "offset": offset}
        try:
            response = request(session, "POST", endpoint, json=payload)
        except HttpStatusError as exc:
            # Some API revisions model an all-record search by omitting `text`
            # rather than sending an empty string.
            if text == "" and exc.status_code in {400, 422}:
                response = request(
                    session,
                    "POST",
                    endpoint,
                    json={"limit": page_size, "offset": offset},
                )
            else:
                raise
        body = response.json()
        items, total = extract_search_items(body)
        if not items:
            break

        new_items = 0
        for item in items:
            urn = extract_score_set_urn(item)
            if urn and urn not in seen_urns:
                seen_urns.add(urn)
                results.append(item)
                new_items += 1
                if max_results is not None and len(results) >= max_results:
                    return results

        if delay > 0:
            time.sleep(delay)
        if new_items == 0:
            break
        offset += len(items)
        if len(items) < page_size:
            break
        if total is not None and offset >= total:
            break
    return results


def extract_score_set_urn(record: Mapping[str, Any]) -> str:
    for key in ("urn", "scoreSetUrn", "score_set_urn", "accession"):
        value = safe_str(record.get(key))
        if value.startswith("urn:mavedb:"):
            return value
    return ""


def recursively_iter_strings(obj: Any) -> Iterator[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, Mapping):
        for value in obj.values():
            yield from recursively_iter_strings(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from recursively_iter_strings(value)


def collect_metadata_text(meta: Mapping[str, Any]) -> str:
    pieces: List[str] = []

    def visit(obj: Any, key_hint: str = "") -> None:
        if isinstance(obj, Mapping):
            for key, value in obj.items():
                norm = normalize_colname(key)
                if any(token in norm for token in METADATA_TEXT_KEYS):
                    if isinstance(value, str):
                        pieces.append(value)
                    elif isinstance(value, (list, tuple)):
                        pieces.extend(s for s in recursively_iter_strings(value))
                    elif isinstance(value, Mapping):
                        pieces.extend(s for s in recursively_iter_strings(value))
                visit(value, norm)
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                visit(value, key_hint)

    visit(meta)
    return "\n".join(dict.fromkeys(p.strip() for p in pieces if p.strip()))


def pattern_direction(text: str) -> Tuple[str, str]:
    low = text.lower()
    gof_hits = [p for p in GOF_PATTERNS if re.search(p, low, flags=re.I)]
    lof_hits = [p for p in LOF_PATTERNS if re.search(p, low, flags=re.I)]
    if gof_hits and not lof_hits:
        return "gof", gof_hits[0]
    if lof_hits and not gof_hits:
        return "lof", lof_hits[0]
    if gof_hits and lof_hits:
        return "ambiguous", f"GOF={gof_hits[0]}; LOF={lof_hits[0]}"
    return "", ""


def nested_get(obj: Any, *path: str) -> Any:
    current = obj
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def first_nonempty(*values: Any) -> str:
    for value in values:
        text = safe_str(value)
        if text:
            return text
    return ""


def extract_gene_symbols(meta: Mapping[str, Any]) -> List[str]:
    symbols: List[str] = []
    target_lists = []
    for key in ("targetGenes", "target_genes", "targets"):
        value = meta.get(key)
        if isinstance(value, list):
            target_lists.extend(value)
    for target in target_lists:
        if not isinstance(target, Mapping):
            continue
        for key in ("name", "symbol", "gene", "geneSymbol", "gene_symbol"):
            value = safe_str(target.get(key))
            if value:
                symbols.append(value.upper())
        for value in recursively_iter_strings(target.get("targetGene", {})):
            if re.fullmatch(r"[A-Za-z0-9-]{2,20}", value):
                symbols.append(value.upper())
    return list(dict.fromkeys(symbols))


def extract_publications(meta: Mapping[str, Any]) -> str:
    ids: List[str] = []
    for key in ("publicationIdentifiers", "publication_identifiers", "publications"):
        value = meta.get(key)
        if value is None:
            continue
        for text in recursively_iter_strings(value):
            if re.search(r"(?:PMID|doi|10\.)", text, flags=re.I):
                ids.append(text)
    return "; ".join(dict.fromkeys(ids))


def extract_license(meta: Mapping[str, Any]) -> str:
    lic = meta.get("license")
    if isinstance(lic, Mapping):
        return first_nonempty(lic.get("identifier"), lic.get("name"), lic.get("shortName"), lic.get("urn"))
    return safe_str(lic)


def extract_experiment_urn(meta: Mapping[str, Any]) -> str:
    experiment = meta.get("experiment")
    if isinstance(experiment, Mapping):
        value = extract_score_set_urn(experiment)
        if value:
            return value
        value = safe_str(experiment.get("urn"))
        if value:
            return value
    return first_nonempty(meta.get("experimentUrn"), meta.get("experiment_urn"))


def metadata_catalog_row(meta: Mapping[str, Any], urn: str) -> Dict[str, Any]:
    metadata_text = collect_metadata_text(meta)
    inferred, rule = pattern_direction(metadata_text)
    title = first_nonempty(meta.get("title"), nested_get(meta, "scoreSet", "title"))
    genes = extract_gene_symbols(meta)
    return {
        "score_set_urn": urn,
        "experiment_urn": extract_experiment_urn(meta),
        "title": title,
        "gene_symbols": ";".join(genes),
        "num_variants": meta.get("numVariants", meta.get("num_variants")),
        "license": extract_license(meta),
        "publications": extract_publications(meta),
        "metadata_directionality_inferred": inferred,
        "metadata_directionality_rule": rule,
        "metadata_text_excerpt": re.sub(r"\s+", " ", metadata_text)[:1000],
    }


def load_assay_map(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return {}
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    norm_to_real = {normalize_colname(c): c for c in frame.columns}
    urn_col = norm_to_real.get("scoreseturn")
    if urn_col is None:
        raise ValueError(
            f"Assay map must contain score_set_urn. Columns: {list(frame.columns)}"
        )

    mapping: Dict[str, Dict[str, Any]] = {}
    for _, row in frame.iterrows():
        urn = safe_str(row[urn_col])
        if not urn or urn.startswith("#"):
            continue
        normalized: Dict[str, Any] = {}
        for norm, real in norm_to_real.items():
            normalized[norm] = row[real]
        include = parse_bool(normalized.get("include", ""), default=True)
        direction = safe_str(normalized.get("abnormaldirectionality", "")).lower()
        aliases = {
            "loss_of_function": "lof", "loss-of-function": "lof",
            "gain_of_function": "gof", "gain-of-function": "gof",
        }
        direction = aliases.get(direction, direction)
        if direction not in {"", "lof", "gof", "exclude", "ambiguous"}:
            raise ValueError(
                f"Invalid abnormal_directionality={direction!r} for {urn}. "
                "Expected lof, gof, exclude, ambiguous, or blank."
            )
        mapping[urn] = {
            "include": include and direction != "exclude",
            "abnormal_directionality": direction,
            "score_column": safe_str(normalized.get("scorecolumn", "")) or "score",
            "lof_min": optional_float(normalized.get("lofmin", "")),
            "lof_max": optional_float(normalized.get("lofmax", "")),
            "gof_min": optional_float(normalized.get("gofmin", "")),
            "gof_max": optional_float(normalized.get("gofmax", "")),
            "notes": safe_str(normalized.get("notes", "")),
        }
    return mapping


def find_columns(df: pd.DataFrame, exact: Sequence[str] = (), contains: Sequence[str] = ()) -> List[str]:
    exact_norm = {normalize_colname(x) for x in exact}
    out: List[str] = []
    for col in df.columns:
        norm = normalize_colname(col)
        if norm in exact_norm or any(token in norm for token in contains):
            out.append(col)
    return out


def aa_to_one(value: str) -> Optional[str]:
    if value in AA3_TO_1:
        return AA3_TO_1[value]
    if value in AA1:
        return value
    return None


def extract_protein_substitution(value: Any) -> Tuple[str, str]:
    text = safe_str(value)
    if not text:
        return "", ""
    for match in PROTEIN_SUB_RE.finditer(text):
        tail = text[match.end():].lstrip(")")[:12].lower()
        if re.match(r"(?:fs|ter|del|dup|ins|ext|=|\*)", tail):
            continue
        ref = aa_to_one(match.group("ref"))
        alt = aa_to_one(match.group("alt"))
        if ref and alt and ref != alt:
            normalized = f"p.{ref}{match.group('pos')}{alt}"
            return normalized, match.group(0)
    return "", ""


def derive_protein_changes(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    candidate_cols = find_columns(
        df,
        exact=("hgvs_pro", "hgvsp", "protein_change", "variant", "hgvs"),
        contains=("hgvsp", "protein", "hgvs", "variant"),
    )
    if not candidate_cols:
        candidate_cols = [c for c in df.columns if df[c].dtype == object]

    normalized: List[str] = []
    raw_matches: List[str] = []
    source_cols: List[str] = []
    for _, row in df.iterrows():
        found_norm = ""
        found_raw = ""
        found_col = ""
        for col in candidate_cols:
            found_norm, found_raw = extract_protein_substitution(row[col])
            if found_norm:
                found_col = str(col)
                break
        normalized.append(found_norm)
        raw_matches.append(found_raw)
        source_cols.append(found_col)
    return (
        pd.Series(normalized, index=df.index, dtype="string"),
        pd.Series(raw_matches, index=df.index, dtype="string"),
        pd.Series(source_cols, index=df.index, dtype="string"),
    )


def choose_variant_urn_series(df: pd.DataFrame) -> pd.Series:
    candidates = find_columns(
        df,
        exact=("urn", "variant_urn", "mavedb_urn", "accession"),
        contains=("varianturn", "mavedburn"),
    )
    out = pd.Series("", index=df.index, dtype="string")
    for col in candidates:
        values = safe_str_series(df[col])
        mask = out.eq("") & values.str.contains(r"urn:mavedb:.*#\d+", regex=True, na=False)
        out.loc[mask] = values.loc[mask]
    return out


def choose_vrs_id_series(df: pd.DataFrame) -> pd.Series:
    candidates = find_columns(
        df,
        exact=("vrs_id", "vrs_urn", "ga4gh_id"),
        contains=("vrs", "ga4gh"),
    )
    out = pd.Series("", index=df.index, dtype="string")
    for col in candidates:
        values = safe_str_series(df[col])
        extracted = values.str.extract(r"(ga4gh:VA\.[A-Za-z0-9_-]+)", expand=False).fillna("")
        mask = out.eq("") & extracted.ne("")
        out.loc[mask] = extracted.loc[mask]
    return out


def urn_variant_index(value: Any) -> str:
    match = re.search(r"#([^\s#]+)$", safe_str(value))
    return match.group(1) if match else ""


def extract_primary_code(value: Any) -> str:
    if isinstance(value, str):
        return value.strip().lower()
    if not isinstance(value, Mapping):
        return ""
    for path in (
        ("primaryCoding", "code"),
        ("primary_coding", "code"),
        ("code",),
    ):
        found = nested_get(value, *path)
        if isinstance(found, str):
            return found.strip().lower()
    return ""


def extract_mapped_variant_rows(payload: Any) -> pd.DataFrame:
    """Flatten MaveDB mapped-variant/VRS JSON into mergeable identifiers."""
    if isinstance(payload, dict):
        for key in ("items", "results", "data", "mappedVariants", "mapped_variants", "variants"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
        else:
            payload = [payload]
    if not isinstance(payload, list):
        return pd.DataFrame()

    rows: List[Dict[str, str]] = []
    for item in payload:
        if not isinstance(item, Mapping):
            continue
        post = item.get("post_mapped", item.get("postMapped", {}))
        pre = item.get("pre_mapped", item.get("preMapped", {}))
        if not isinstance(post, Mapping):
            post = {}
        if not isinstance(pre, Mapping):
            pre = {}

        variant_urn = ""
        for text in recursively_iter_strings(item):
            match = re.search(r"(urn:mavedb:[^\s\"']+?#\d+)", text)
            if match:
                variant_urn = match.group(1).rstrip(".,;)")
                break

        vrs_id = first_nonempty(post.get("id"), pre.get("id"))
        protein_change = ""
        protein_change_raw = ""
        protein_source = ""
        for source_name, obj in (("post_mapped", post), ("pre_mapped", pre)):
            expressions = obj.get("expressions", []) if isinstance(obj, Mapping) else []
            if not isinstance(expressions, list):
                continue
            for expression in expressions:
                if not isinstance(expression, Mapping):
                    continue
                syntax = safe_str(expression.get("syntax")).lower()
                value = safe_str(expression.get("value"))
                if syntax == "hgvs.p" or "p." in value:
                    protein_change, protein_change_raw = extract_protein_substitution(value)
                    if protein_change:
                        protein_source = source_name
                        break
            if protein_change:
                break
        if not protein_change:
            for text in recursively_iter_strings(item):
                protein_change, protein_change_raw = extract_protein_substitution(text)
                if protein_change:
                    protein_source = "recursive_expression_search"
                    break

        rows.append({
            "variant_urn": variant_urn,
            "variant_index": urn_variant_index(variant_urn),
            "vrs_id": vrs_id,
            "ProteinChange": protein_change,
            "ProteinChange_raw": protein_change_raw,
            "mapped_protein_source": protein_source,
        })
    return pd.DataFrame(rows)


def attach_mapped_variants(scores: pd.DataFrame, mapped: pd.DataFrame) -> pd.DataFrame:
    out = scores.copy()
    if "mapped_variant_match_key" not in out.columns:
        out["mapped_variant_match_key"] = ""
    if "mapped_protein_source" not in out.columns:
        out["mapped_protein_source"] = ""
    if mapped.empty:
        return out

    lookups = {
        "vrs_id": unique_lookup(mapped, "vrs_id"),
        "variant_urn": unique_lookup(mapped, "variant_urn"),
        "variant_index": unique_lookup(mapped, "variant_index"),
        "ProteinChange": unique_lookup(mapped, "ProteinChange"),
    }
    for idx, row in out.iterrows():
        hit: Optional[Dict[str, Any]] = None
        hit_key = ""
        for key in ("variant_urn", "variant_index", "vrs_id", "ProteinChange"):
            value = safe_str(row.get(key))
            if value and value in lookups[key]:
                hit = lookups[key][value]
                hit_key = key
                break
        if hit is None:
            continue
        if not safe_str(row.get("ProteinChange")):
            out.at[idx, "ProteinChange"] = safe_str(hit.get("ProteinChange"))
            out.at[idx, "ProteinChange_raw"] = safe_str(hit.get("ProteinChange_raw"))
            out.at[idx, "protein_change_source_column"] = "MaveDB mapped VRS expression"
        if not safe_str(row.get("vrs_id")):
            out.at[idx, "vrs_id"] = safe_str(hit.get("vrs_id"))
        out.at[idx, "mapped_variant_match_key"] = hit_key
        out.at[idx, "mapped_protein_source"] = safe_str(hit.get("mapped_protein_source"))
    out["is_missense_like"] = out["ProteinChange"].fillna("").astype(str).ne("")
    return out


def extract_statement_rows(payload: Any) -> pd.DataFrame:
    if isinstance(payload, dict):
        for key in ("items", "results", "data", "statements", "annotations"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
        else:
            payload = [payload]
    if not isinstance(payload, list):
        return pd.DataFrame()

    rows: List[Dict[str, str]] = []
    for item in payload:
        if not isinstance(item, Mapping):
            continue
        proposition = item.get("proposition", {})
        subject = proposition.get("subjectVariant", {}) if isinstance(proposition, Mapping) else {}
        classification = extract_primary_code(item.get("classification"))
        direction = safe_str(item.get("direction")).lower()
        vrs_id = safe_str(subject.get("id")) if isinstance(subject, Mapping) else ""
        description = safe_str(item.get("description"))

        all_strings = list(recursively_iter_strings(item))
        variant_urn = ""
        for text in [description, *all_strings]:
            match = re.search(r"(urn:mavedb:[^\s\"']+?#\d+)", text)
            if match:
                variant_urn = match.group(1).rstrip(".,;)")
                break

        protein_change = ""
        for text in all_strings:
            protein_change, _ = extract_protein_substitution(text)
            if protein_change:
                break

        gene = ""
        if isinstance(proposition, Mapping):
            obj_gene = proposition.get("objectGene", {})
            if isinstance(obj_gene, Mapping):
                gene = first_nonempty(obj_gene.get("label"), obj_gene.get("name"))

        calibration_labels: List[str] = []
        evidence_lines = item.get("hasEvidenceLines", [])
        if isinstance(evidence_lines, list):
            for line in evidence_lines:
                if not isinstance(line, Mapping):
                    continue
                specified = line.get("specifiedBy", {})
                if isinstance(specified, Mapping):
                    label = first_nonempty(specified.get("label"), specified.get("description"))
                    if label:
                        calibration_labels.append(label)

        rows.append({
            "mavedb_functional_classification": classification,
            "mavedb_functional_direction": direction,
            "vrs_id": vrs_id,
            "variant_urn": variant_urn,
            "variant_index": urn_variant_index(variant_urn),
            "ProteinChange": protein_change,
            "statement_gene": gene,
            "functional_statement_description": description,
            "functional_statement_calibrations": "; ".join(dict.fromkeys(calibration_labels)),
        })
    return pd.DataFrame(rows)


def unique_lookup(df: pd.DataFrame, key: str) -> Dict[str, Dict[str, Any]]:
    if key not in df.columns or df.empty:
        return {}
    valid = df.loc[df[key].fillna("").astype(str).ne("")].copy()
    if valid.empty:
        return {}
    counts = valid[key].astype(str).value_counts()
    valid_keys = set(counts[counts == 1].index)
    valid = valid.loc[valid[key].astype(str).isin(valid_keys)]
    return {str(row[key]): row.to_dict() for _, row in valid.iterrows()}


def attach_functional_statements(scores: pd.DataFrame, statements: pd.DataFrame) -> pd.DataFrame:
    out = scores.copy()
    columns = (
        "mavedb_functional_classification",
        "mavedb_functional_direction",
        "functional_statement_description",
        "functional_statement_calibrations",
        "functional_statement_match_key",
    )
    for col in columns:
        if col not in out.columns:
            out[col] = ""

    if statements.empty:
        return out

    lookups = {
        "vrs_id": unique_lookup(statements, "vrs_id"),
        "variant_urn": unique_lookup(statements, "variant_urn"),
        "variant_index": unique_lookup(statements, "variant_index"),
        "ProteinChange": unique_lookup(statements, "ProteinChange"),
    }

    for idx, row in out.iterrows():
        hit: Optional[Dict[str, Any]] = None
        hit_key = ""
        for key in ("vrs_id", "variant_urn", "variant_index", "ProteinChange"):
            value = safe_str(row.get(key))
            if value and value in lookups[key]:
                hit = lookups[key][value]
                hit_key = key
                break
        if hit is None:
            continue
        out.at[idx, "mavedb_functional_classification"] = safe_str(
            hit.get("mavedb_functional_classification")
        )
        out.at[idx, "mavedb_functional_direction"] = safe_str(
            hit.get("mavedb_functional_direction")
        )
        out.at[idx, "functional_statement_description"] = safe_str(
            hit.get("functional_statement_description")
        )
        out.at[idx, "functional_statement_calibrations"] = safe_str(
            hit.get("functional_statement_calibrations")
        )
        out.at[idx, "functional_statement_match_key"] = hit_key
    return out


def author_functional_class(row: pd.Series, annotation_cols: Sequence[str]) -> Tuple[str, str]:
    for col in annotation_cols:
        text = safe_str(row.get(col)).lower()
        if not text:
            continue
        if re.search(r"\bindeterminate\b|\bintermediate\b|\buncertain\b", text):
            return "indeterminate", str(col)
        if re.search(r"\babnormal\b|\bnon[- ]?functional\b|\bdamaging\b", text):
            return "abnormal", str(col)
        if re.search(r"\bnormal\b|\bwild[- ]?type[- ]?like\b", text) or text.strip() == "functional":
            return "normal", str(col)
    return "", ""


def direct_variant_direction(row: pd.Series, annotation_cols: Sequence[str]) -> DirectionCall:
    text_pieces: List[str] = []
    used_cols: List[str] = []
    for col in annotation_cols:
        value = safe_str(row.get(col))
        if value:
            text_pieces.append(value)
            used_cols.append(str(col))
    text = " | ".join(text_pieces)
    direction, rule = pattern_direction(text)
    if direction in {"lof", "gof"}:
        return DirectionCall(
            label=direction,
            source="explicit_variant_annotation",
            confidence="high",
            evidence=f"Explicit {direction.upper()} term in variant annotation column(s): {', '.join(used_cols)}",
            rule=rule,
        )
    if direction == "ambiguous":
        return DirectionCall(
            source="explicit_variant_annotation_ambiguous",
            confidence="ambiguous",
            evidence=f"Both LoF and GoF terms found in variant annotation column(s): {', '.join(used_cols)}",
            rule=rule,
        )
    return DirectionCall()


def in_range(value: float, minimum: Optional[float], maximum: Optional[float]) -> bool:
    if minimum is None and maximum is None:
        return False
    if minimum is not None and value < minimum:
        return False
    if maximum is not None and value > maximum:
        return False
    return True


def threshold_direction(row: pd.Series, mapping: Mapping[str, Any]) -> DirectionCall:
    score_column = safe_str(mapping.get("score_column")) or "score"
    if score_column not in row.index:
        # tolerate normalized/case-insensitive column name in the mapping
        target = normalize_colname(score_column)
        matches = [c for c in row.index if normalize_colname(c) == target]
        if matches:
            score_column = str(matches[0])
        else:
            return DirectionCall()
    score = pd.to_numeric(pd.Series([row.get(score_column)]), errors="coerce").iloc[0]
    if pd.isna(score):
        return DirectionCall()
    score_f = float(score)
    lof = in_range(score_f, mapping.get("lof_min"), mapping.get("lof_max"))
    gof = in_range(score_f, mapping.get("gof_min"), mapping.get("gof_max"))
    if lof and not gof:
        return DirectionCall(
            label="lof",
            source="curated_score_threshold",
            confidence="high_curated_rule",
            evidence=f"{score_column}={score_f:g} matched curated LoF range",
            rule=f"lof_min={mapping.get('lof_min')}; lof_max={mapping.get('lof_max')}",
        )
    if gof and not lof:
        return DirectionCall(
            label="gof",
            source="curated_score_threshold",
            confidence="high_curated_rule",
            evidence=f"{score_column}={score_f:g} matched curated GoF range",
            rule=f"gof_min={mapping.get('gof_min')}; gof_max={mapping.get('gof_max')}",
        )
    if lof and gof:
        return DirectionCall(
            source="curated_score_threshold_ambiguous",
            confidence="ambiguous",
            evidence=f"{score_column}={score_f:g} matched both curated LoF and GoF ranges",
        )
    return DirectionCall()


def choose_direction_call(
    row: pd.Series,
    *,
    annotation_cols: Sequence[str],
    direct_direction_cols: Sequence[str],
    assay_mapping: Optional[Mapping[str, Any]],
    metadata_direction: str,
    metadata_rule: str,
    allow_metadata_directionality: bool,
) -> DirectionCall:
    # User-curated score thresholds are the most specific and take precedence over
    # broad free-text annotations.
    if assay_mapping:
        threshold = threshold_direction(row, assay_mapping)
        if threshold.label or threshold.confidence == "ambiguous":
            note = safe_str(assay_mapping.get("notes"))
            if note:
                threshold.evidence += f"; mapping note: {note}"
            return threshold

    functional_class = safe_str(row.get("mavedb_functional_classification")).lower()
    functional_source = "MaveDB calibrated functional statement"
    if not functional_class:
        functional_class = safe_str(row.get("author_functional_classification")).lower()
        functional_source = "author-provided score-table classification"

    if functional_class == "abnormal" and assay_mapping:
        direction = safe_str(assay_mapping.get("abnormal_directionality")).lower()
        if direction in {"lof", "gof"}:
            note = safe_str(assay_mapping.get("notes"))
            evidence = f"{functional_source}='abnormal'; curated score-set mapping says abnormal={direction.upper()}"
            if note:
                evidence += f"; mapping note: {note}"
            return DirectionCall(
                label=direction,
                source="curated_assay_directionality",
                confidence="high_curated_rule",
                evidence=evidence,
                rule=f"abnormal_directionality={direction}",
            )
        if direction in {"ambiguous", "exclude"}:
            return DirectionCall(
                source="curated_assay_directionality_ambiguous",
                confidence="ambiguous",
                evidence=f"{functional_source}='abnormal', but assay map marks directionality {direction}",
            )

    # Explicit variant-level direction is accepted only from direction-oriented
    # columns, not generic phenotype/consequence/function metadata.
    direct = direct_variant_direction(row, direct_direction_cols)
    if direct.label or direct.confidence == "ambiguous":
        return direct

    if functional_class != "abnormal":
        return DirectionCall()

    if allow_metadata_directionality and metadata_direction in {"lof", "gof"}:
        return DirectionCall(
            label=metadata_direction,
            source="explicit_metadata_plus_abnormal_class",
            confidence="medium_metadata_explicit",
            evidence=(
                f"{functional_source}='abnormal'; explicit {metadata_direction.upper()} phrase detected "
                "in score-set metadata"
            ),
            rule=metadata_rule,
        )
    return DirectionCall()

def standardize_score_table(
    raw: pd.DataFrame,
    *,
    urn: str,
    meta: Mapping[str, Any],
    endpoint_name: str,
    statements: pd.DataFrame,
    mapped_variants: pd.DataFrame,
    assay_mapping: Optional[Mapping[str, Any]],
    allow_metadata_directionality: bool,
    preserve_raw_columns: bool,
) -> pd.DataFrame:
    df = raw.copy()
    df = df.dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]

    protein_change, protein_change_raw, protein_source_col = derive_protein_changes(df)
    variant_urn = choose_variant_urn_series(df)
    vrs_id = choose_vrs_id_series(df)
    variant_index = variant_urn.map(urn_variant_index).astype("string")

    score_col = next((c for c in df.columns if normalize_colname(c) == "score"), None)
    score = pd.to_numeric(df[score_col], errors="coerce") if score_col else pd.Series(np.nan, index=df.index)

    genes = extract_gene_symbols(meta)
    title = first_nonempty(meta.get("title"), nested_get(meta, "scoreSet", "title"))
    metadata_text = collect_metadata_text(meta)
    metadata_direction, metadata_rule = pattern_direction(metadata_text)

    out = pd.DataFrame(index=df.index)
    out["source_database"] = "MaveDB"
    out["source_row_index"] = pd.Series(np.arange(len(df)), index=df.index, dtype="int64")
    out["score_set_urn"] = urn
    out["experiment_urn"] = extract_experiment_urn(meta)
    out["score_set_title"] = title
    out["HugoSymbol"] = genes[0] if genes else ""
    out["all_target_genes"] = ";".join(genes)
    out["ProteinChange"] = protein_change
    out["ProteinChange_raw"] = protein_change_raw
    out["protein_change_source_column"] = protein_source_col
    out["variant_urn"] = variant_urn
    out["variant_index"] = variant_index
    out["vrs_id"] = vrs_id
    out["score"] = score
    out["score_source_column"] = score_col or ""
    out["variant_data_endpoint"] = endpoint_name
    out["is_missense_like"] = out["ProteinChange"].ne("")
    out["license"] = extract_license(meta)
    out["publications"] = extract_publications(meta)
    out["metadata_directionality_inferred"] = metadata_direction
    out["metadata_directionality_rule"] = metadata_rule

    # Raw columns are temporarily added before annotation parsing so that direct
    # author labels and user-specified threshold columns remain available.
    raw_name_map: Dict[str, str] = {}
    for col in df.columns:
        safe_name = "mavedb_raw__" + re.sub(r"[^A-Za-z0-9_]+", "_", str(col).strip())
        base = safe_name
        counter = 2
        while safe_name in out.columns:
            safe_name = f"{base}_{counter}"
            counter += 1
        raw_name_map[str(col)] = safe_name
        out[safe_name] = df[col]

    # Copy original columns under their original names into a private working
    # frame, because assay-map score_column refers to MaveDB's column name.
    working = out.copy()
    for col in df.columns:
        if col not in working.columns:
            working[col] = df[col]

    working = attach_mapped_variants(working, mapped_variants)
    working = attach_functional_statements(working, statements)

    annotation_cols_original = [
        c for c in df.columns
        if any(token in normalize_colname(c) for token in ANNOTATION_COLUMN_TOKENS)
    ]
    annotation_cols = annotation_cols_original + [raw_name_map[c] for c in annotation_cols_original]
    annotation_cols = list(dict.fromkeys(c for c in annotation_cols if c in working.columns))
    direct_direction_cols = [
        c for c in annotation_cols
        if any(token in normalize_colname(c) for token in DIRECT_DIRECTION_COLUMN_TOKENS)
    ]

    author_classes: List[str] = []
    author_class_cols: List[str] = []
    calls: List[DirectionCall] = []
    for _, row in working.iterrows():
        author_class, author_col = author_functional_class(row, annotation_cols)
        author_classes.append(author_class)
        author_class_cols.append(author_col)
        row = row.copy()
        row["author_functional_classification"] = author_class
        calls.append(
            choose_direction_call(
                row,
                annotation_cols=annotation_cols,
                direct_direction_cols=direct_direction_cols,
                assay_mapping=assay_mapping,
                metadata_direction=metadata_direction,
                metadata_rule=metadata_rule,
                allow_metadata_directionality=allow_metadata_directionality,
            )
        )

    working["author_functional_classification"] = author_classes
    working["author_functional_class_source_column"] = author_class_cols
    working["directionality_label"] = [c.label for c in calls]
    working["directionality_label_normalized"] = [c.label for c in calls]
    working["binary_label_lof0_gof1"] = pd.Series(
        [0 if c.label == "lof" else 1 if c.label == "gof" else pd.NA for c in calls],
        index=working.index,
        dtype="Int8",
    )
    working["label_confidence"] = [c.confidence for c in calls]
    working["label_source"] = [c.source for c in calls]
    working["label_evidence"] = [c.evidence for c in calls]
    working["label_rule"] = [c.rule for c in calls]

    # Remove duplicate original-name working columns and optionally raw columns.
    for col in df.columns:
        if col in working.columns and col not in out.columns:
            working = working.drop(columns=[col])
    if not preserve_raw_columns:
        working = working.drop(columns=[c for c in working.columns if c.startswith("mavedb_raw__")])

    return working


def make_parquet_safe(df: pd.DataFrame) -> pd.DataFrame:
    safe = df.copy()
    for col in safe.columns:
        if safe[col].dtype == object:
            safe[col] = safe[col].map(
                lambda value: json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list, tuple))
                else value
            ).astype("string")
    return safe


def write_summary(df: pd.DataFrame, summary_json: Optional[Path], summary_csv: Optional[Path]) -> None:
    def counts(col: str) -> Dict[str, int]:
        if col not in df.columns:
            return {}
        return {str(k): int(v) for k, v in df[col].value_counts(dropna=False).items()}

    summary = {
        "n_rows": int(len(df)),
        "n_score_sets": int(df["score_set_urn"].nunique()) if "score_set_urn" in df else 0,
        "n_genes": int(df["HugoSymbol"].replace("", np.nan).nunique(dropna=True)) if "HugoSymbol" in df else 0,
        "n_unique_protein_variants": int(
            df[["HugoSymbol", "ProteinChange"]].drop_duplicates().shape[0]
        ) if {"HugoSymbol", "ProteinChange"}.issubset(df.columns) else 0,
        "directionality_counts": counts("directionality_label_normalized"),
        "label_source_counts": counts("label_source"),
        "confidence_counts": counts("label_confidence"),
        "functional_classification_counts": counts("mavedb_functional_classification"),
        "missense_counts": counts("is_missense_like"),
    }
    if summary_json is not None:
        ensure_dir(summary_json.parent if summary_json.parent != Path("") else Path("."))
        summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if summary_csv is not None:
        rows: List[Dict[str, Any]] = []
        for group in (
            "directionality_counts", "label_source_counts", "confidence_counts",
            "functional_classification_counts", "missense_counts",
        ):
            for value, count in summary[group].items():
                rows.append({"group": group, "value": value, "count": count})
        ensure_dir(summary_csv.parent if summary_csv.parent != Path("") else Path("."))
        pd.DataFrame(rows).to_csv(summary_csv, index=False)
    log("Summary: " + json.dumps(summary, sort_keys=True))


def read_urn_file(path: Optional[Path]) -> List[str]:
    if path is None:
        return []
    urns: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urns.append(line.split(",")[0].strip())
    return urns


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--out", required=True, help="Output Parquet path.")
    parser.add_argument("--workdir", default="mavedb_cache", help="Cache/download directory.")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="MaveDB API base URL.")
    parser.add_argument("--score-set-urn", action="append", default=[], help="Score-set URN; repeatable.")
    parser.add_argument("--urn-file", default=None, help="Text/CSV file with one score-set URN per line.")
    parser.add_argument("--search-text", action="append", default=[], help="MaveDB score-set search text; repeatable.")
    parser.add_argument("--all-published", action="store_true", help="Search all published score sets using an empty search string.")
    parser.add_argument("--max-score-sets", type=int, default=None, help="Optional cap after de-duplication.")
    parser.add_argument("--page-size", type=int, default=100, help="Search pagination size.")
    parser.add_argument("--assay-map-csv", default=None, help="Curated score-set directionality/threshold mapping CSV.")
    parser.add_argument("--missense-only", action="store_true", help="Keep only parsed protein missense substitutions.")
    parser.add_argument("--labeled-only", action="store_true", help="Keep only rows assigned LoF or GoF.")
    parser.add_argument("--calibrated-only", action="store_true", help="Keep only rows matched to a MaveDB calibrated functional statement.")
    parser.add_argument("--no-metadata-directionality", action="store_true", help="Disable explicit metadata directionality inference.")
    parser.add_argument("--no-mapped-variants", action="store_true", help="Do not download MaveDB mapped VRS variants for protein HGVS enrichment.")
    parser.add_argument("--drop-raw-columns", action="store_true", help="Do not preserve downloaded score-table columns.")
    parser.add_argument("--force-download", action="store_true", help="Ignore cached API responses.")
    parser.add_argument("--request-delay", type=float, default=0.1, help="Polite delay between API requests, seconds.")
    parser.add_argument("--summary-json", default=None, help="Optional JSON summary output.")
    parser.add_argument("--summary-csv", default=None, help="Optional CSV summary output.")
    parser.add_argument("--score-set-catalog-csv", default=None, help="Optional score-set metadata/QC catalog.")
    parser.add_argument("--errors-csv", default=None, help="Optional per-score-set errors CSV.")
    parser.add_argument("--catalog-only", action="store_true", help="Write catalog and skip variant downloads/output table.")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    ensure_dir(workdir)
    metadata_cache = workdir / "metadata"
    variants_cache = workdir / "variants"
    statements_cache = workdir / "functional_statements"
    mapped_cache = workdir / "mapped_variants"

    session = make_session()
    assay_map = load_assay_map(Path(args.assay_map_csv) if args.assay_map_csv else None)

    explicit_urns = list(args.score_set_urn) + read_urn_file(Path(args.urn_file) if args.urn_file else None)
    discovered_stubs: Dict[str, Dict[str, Any]] = {}

    search_terms: List[str] = []
    if args.all_published:
        search_terms.append("")
    search_terms.extend(args.search_text)
    if not explicit_urns and not search_terms:
        search_terms.extend(DEFAULT_SEARCH_TERMS)
        log(f"No selectors supplied; using explicit directionality searches: {list(DEFAULT_SEARCH_TERMS)}")

    for term in search_terms:
        display = term if term else "<all published>"
        log(f"Searching MaveDB score sets: {display}")
        results = search_score_sets(
            session,
            args.api_base,
            term,
            page_size=args.page_size,
            max_results=args.max_score_sets,
            delay=args.request_delay,
        )
        log(f"Search {display!r} returned {len(results):,} unique score sets")
        for item in results:
            urn = extract_score_set_urn(item)
            if urn:
                discovered_stubs.setdefault(urn, item)

    for urn in explicit_urns:
        urn = urn.strip()
        if urn:
            discovered_stubs.setdefault(urn, {"urn": urn})

    urns = list(discovered_stubs)
    if args.max_score_sets is not None:
        urns = urns[:args.max_score_sets]
    if not urns:
        raise RuntimeError("No MaveDB score sets were selected or discovered.")
    log(f"Selected {len(urns):,} unique score sets")

    metadata_by_urn: Dict[str, Dict[str, Any]] = {}
    catalog_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    for i, urn in enumerate(urns, start=1):
        try:
            stem = urn_cache_stem(urn)
            url = f"{args.api_base.rstrip('/')}/score-sets/{quote(urn, safe=':')}"
            meta = read_json_cache_or_download(
                session,
                url,
                metadata_cache / f"{stem}.json",
                force=args.force_download,
                delay=args.request_delay,
            )
            if not isinstance(meta, dict):
                raise RuntimeError("Score-set metadata response was not a JSON object")
            metadata_by_urn[urn] = meta
            row = metadata_catalog_row(meta, urn)
            mapping = assay_map.get(urn)
            row["assay_map_present"] = mapping is not None
            row["assay_map_include"] = mapping.get("include") if mapping else ""
            row["assay_map_abnormal_directionality"] = mapping.get("abnormal_directionality") if mapping else ""
            row["assay_map_score_column"] = mapping.get("score_column") if mapping else ""
            row["assay_map_notes"] = mapping.get("notes") if mapping else ""
            catalog_rows.append(row)
        except Exception as exc:
            errors.append({"score_set_urn": urn, "stage": "metadata", "error": str(exc)})
            log(f"[{i}/{len(urns)}] Metadata failed for {urn}: {exc}")

    catalog = pd.DataFrame(catalog_rows)
    if args.score_set_catalog_csv:
        path = Path(args.score_set_catalog_csv)
        ensure_dir(path.parent if path.parent != Path("") else Path("."))
        catalog.to_csv(path, index=False)
        log(f"Wrote score-set catalog: {path}")

    if args.catalog_only:
        if args.errors_csv:
            path = Path(args.errors_csv)
            ensure_dir(path.parent if path.parent != Path("") else Path("."))
            pd.DataFrame(errors, columns=["score_set_urn", "stage", "error"]).to_csv(path, index=False)
        log("Catalog-only run complete.")
        return

    output_frames: List[pd.DataFrame] = []
    for i, urn in enumerate(urns, start=1):
        meta = metadata_by_urn.get(urn)
        if meta is None:
            continue
        mapping = assay_map.get(urn)
        if mapping is not None and not mapping.get("include", True):
            log(f"[{i}/{len(urns)}] Skipping {urn}: excluded by assay map")
            continue
        try:
            log(f"[{i}/{len(urns)}] Downloading/processing {urn}")
            stem = urn_cache_stem(urn)
            urn_url = quote(urn, safe=":")
            raw, endpoint_name, _ = read_csv_cache_or_download(
                session,
                (
                    ("variants_data", f"{args.api_base.rstrip('/')}/score-sets/{urn_url}/variants/data"),
                    ("scores", f"{args.api_base.rstrip('/')}/score-sets/{urn_url}/scores"),
                ),
                variants_cache,
                stem,
                force=args.force_download,
                delay=args.request_delay,
            )

            if args.no_mapped_variants:
                mapped_variants = pd.DataFrame()
            else:
                mapped_payload = read_json_cache_or_download(
                    session,
                    f"{args.api_base.rstrip('/')}/score-sets/{urn_url}/mapped-variants",
                    mapped_cache / f"{stem}.json",
                    force=args.force_download,
                    delay=args.request_delay,
                    allow_missing=True,
                )
                mapped_variants = extract_mapped_variant_rows(mapped_payload)

            statements_payload = read_json_cache_or_download(
                session,
                f"{args.api_base.rstrip('/')}/score-sets/{urn_url}/annotated-variants/functional-statement",
                statements_cache / f"{stem}.json",
                force=args.force_download,
                delay=args.request_delay,
                allow_missing=True,
            )
            statements = extract_statement_rows(statements_payload)

            frame = standardize_score_table(
                raw,
                urn=urn,
                meta=meta,
                endpoint_name=endpoint_name,
                statements=statements,
                mapped_variants=mapped_variants,
                assay_mapping=mapping,
                allow_metadata_directionality=not args.no_metadata_directionality,
                preserve_raw_columns=not args.drop_raw_columns,
            )
            if args.missense_only:
                frame = frame.loc[frame["is_missense_like"]].copy()
            if args.calibrated_only:
                frame = frame.loc[frame["mavedb_functional_classification"].fillna("").ne("")].copy()
            if args.labeled_only:
                frame = frame.loc[frame["directionality_label_normalized"].isin(["lof", "gof"])].copy()
            if not frame.empty:
                output_frames.append(frame)
            log(
                f"[{i}/{len(urns)}] {urn}: raw={len(raw):,}, "
                f"retained={len(frame):,}, mapped_variants={len(mapped_variants):,}, "
                f"calibrated_statements={len(statements):,}"
            )
        except Exception as exc:
            errors.append({"score_set_urn": urn, "stage": "variants", "error": str(exc)})
            log(f"[{i}/{len(urns)}] Failed {urn}: {exc}")

    if output_frames:
        output = pd.concat(output_frames, ignore_index=True, sort=False)
    else:
        output = pd.DataFrame(columns=[
            "source_database", "score_set_urn", "HugoSymbol", "ProteinChange",
            "score", "directionality_label_normalized", "binary_label_lof0_gof1",
            "label_confidence", "label_source", "label_evidence",
        ])

    # Deduplicate only records with a stable variant identifier. Rows lacking all
    # variant identifiers are retained separately using source_row_index.
    id_cols = [c for c in ("variant_urn", "vrs_id", "ProteinChange", "ProteinChange_raw") if c in output.columns]
    stable_id = pd.Series(False, index=output.index)
    for col in id_cols:
        stable_id |= output[col].fillna("").astype(str).str.strip().ne("")
    identified = output.loc[stable_id].copy()
    unidentified = output.loc[~stable_id].copy()
    dedup_cols = [
        c for c in ("score_set_urn", "variant_urn", "vrs_id", "ProteinChange", "ProteinChange_raw", "directionality_label_normalized")
        if c in identified.columns
    ]
    if dedup_cols:
        identified = identified.drop_duplicates(dedup_cols)
    output = pd.concat([identified, unidentified], ignore_index=True, sort=False)

    out_path = Path(args.out)
    ensure_dir(out_path.parent if out_path.parent != Path("") else Path("."))
    log(f"Writing Parquet: {out_path}")
    make_parquet_safe(output).to_parquet(out_path, index=False)

    write_summary(
        output,
        Path(args.summary_json) if args.summary_json else None,
        Path(args.summary_csv) if args.summary_csv else None,
    )

    if args.errors_csv:
        path = Path(args.errors_csv)
        ensure_dir(path.parent if path.parent != Path("") else Path("."))
        pd.DataFrame(errors, columns=["score_set_urn", "stage", "error"]).to_csv(path, index=False)
        log(f"Wrote errors/QC table: {path}")

    if errors:
        log(f"Completed with {len(errors):,} score-set errors; inspect --errors-csv.")
    else:
        log("Done.")


if __name__ == "__main__":
    main()
