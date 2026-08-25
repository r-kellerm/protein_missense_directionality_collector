#!/usr/bin/env python3
"""
Download the OncoKB bulk annotated-variant collection and build a protein
missense GOF/LOF directionality table.

Unlike the earlier version, this script does NOT require --input. It attempts
to retrieve OncoKB's bulk utility endpoint:

    /api/v1/utils/allAnnotatedVariants

and retains explicit missense protein substitutions whose biological effect is:

    Gain-of-function
    Likely Gain-of-function
    Loss-of-function
    Likely Loss-of-function

Important:
- A valid OncoKB API token is normally required.
- OncoKB may restrict or disable bulk access for a given account/license.
  If the bulk endpoint returns 401/403/404, this script stops with a clear
  message. It deliberately does NOT substitute an incomplete external variant
  universe while calling it "all OncoKB".
- The raw bulk JSON is cached under --workdir so the exact downloaded snapshot
  can be retained for reproducibility.

Example:
    set ONCOKB_API_TOKEN=YOUR_TOKEN
    python oncokb_directionality_all_missense.py ^
        --out oncokb_missense_directionality.parquet ^
        --summary-json oncokb_summary.json

PowerShell:
    $env:ONCOKB_API_TOKEN="YOUR_TOKEN"
    python oncokb_directionality_all_missense.py --out oncokb_missense_directionality.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from directionality_common import *


BULK_ENDPOINT = "/api/v1/utils/allAnnotatedVariants"

AA1 = "ACDEFGHIKLMNPQRSTVWY"
AA3 = (
    "Ala", "Arg", "Asn", "Asp", "Cys", "Gln", "Glu", "Gly", "His", "Ile",
    "Leu", "Lys", "Met", "Phe", "Pro", "Ser", "Thr", "Trp", "Tyr", "Val",
)
AA3_RE = "(?:" + "|".join(AA3) + ")"

EXACT_EFFECTS = {
    "gain-of-function": ("gof_like", "high"),
    "gain of function": ("gof_like", "high"),
    "loss-of-function": ("lof", "high"),
    "loss of function": ("lof", "high"),
}
LIKELY_EFFECTS = {
    "likely gain-of-function": ("gof_like", "medium"),
    "likely gain of function": ("gof_like", "medium"),
    "likely loss-of-function": ("lof", "medium"),
    "likely loss of function": ("lof", "medium"),
}


def log(msg: str) -> None:
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}",
        file=sys.stderr,
        flush=True,
    )


def session_with_retry(token: str = "") -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    s = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "oncokb-directionality-bulk-builder/2.0",
        }
    )
    if token:
        s.headers["Authorization"] = f"Bearer {token}"
    return s


def normalize_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(key).lower())


def walk_json(obj: Any) -> Iterable[Tuple[str, Any]]:
    """Yield every dict key/value pair recursively."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k), v
            yield from walk_json(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk_json(v)


def find_values_by_keys(obj: Any, candidates: Sequence[str]) -> List[Any]:
    wanted = {normalize_key(x) for x in candidates}
    vals: List[Any] = []
    for key, value in walk_json(obj):
        if normalize_key(key) in wanted:
            vals.append(value)
    return vals


def scalar_strings(value: Any) -> List[str]:
    """Flatten scalar-ish JSON values to strings without serializing dicts."""
    out: List[str] = []
    if value is None:
        return out
    if isinstance(value, str):
        s = value.strip()
        if s:
            out.append(s)
    elif isinstance(value, (int, float, bool)):
        out.append(str(value))
    elif isinstance(value, list):
        for x in value:
            out.extend(scalar_strings(x))
    elif isinstance(value, dict):
        # Prefer common human-readable fields from nested API objects.
        for key in (
            "hugoSymbol", "symbol", "name", "alteration", "proteinChange",
            "knownEffect", "effect", "value",
        ):
            if key in value:
                out.extend(scalar_strings(value[key]))
    return out


def first_string(obj: Any, candidates: Sequence[str]) -> str:
    for value in find_values_by_keys(obj, candidates):
        vals = scalar_strings(value)
        if vals:
            return vals[0]
    return ""


def extract_gene(record: Dict[str, Any]) -> str:
    gene = first_string(
        record,
        [
            "hugoSymbol",
            "hugo_symbol",
            "geneSymbol",
            "gene_symbol",
        ],
    )
    if not gene:
        # Some versions may expose a nested gene object or a string "gene".
        for value in find_values_by_keys(record, ["gene"]):
            if isinstance(value, dict):
                g = first_string(
                    value,
                    ["hugoSymbol", "symbol", "geneSymbol", "name"],
                )
                if g:
                    gene = g
                    break
            elif isinstance(value, str):
                gene = value.strip()
                break

    gene = gene.strip().upper()
    # Avoid accidentally accepting a long gene description as a symbol.
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9._-]{0,30}", gene):
        return ""
    return gene


def extract_effect(record: Dict[str, Any]) -> str:
    # In normal annotation responses mutationEffect is an object with knownEffect.
    for value in find_values_by_keys(record, ["mutationEffect"]):
        if isinstance(value, dict):
            eff = first_string(
                value,
                ["knownEffect", "effect", "mutationEffect", "biologicalEffect"],
            )
            if eff:
                return eff.strip()
        elif isinstance(value, str) and value.strip():
            return value.strip()

    return first_string(
        record,
        ["knownEffect", "biologicalEffect", "biological_effect"],
    ).strip()


def classify_effect(effect: str) -> Tuple[str, str]:
    key = re.sub(r"\s+", " ", effect.strip().lower())
    if key in EXACT_EFFECTS:
        return EXACT_EFFECTS[key]
    if key in LIKELY_EFFECTS:
        return LIKELY_EFFECTS[key]

    # Defensive handling of minor API punctuation/wording variations.
    compact = re.sub(r"[-_\s]+", " ", key)
    if compact.startswith("likely ") and "gain" in compact and "function" in compact:
        return "gof_like", "medium"
    if compact.startswith("likely ") and "loss" in compact and "function" in compact:
        return "lof", "medium"
    if "gain" in compact and "function" in compact:
        return "gof_like", "high"
    if "loss" in compact and "function" in compact:
        return "lof", "high"
    return "unknown", "low"


def extract_alteration_candidates(record: Dict[str, Any]) -> List[str]:
    """
    Find likely protein-alteration strings from heterogeneous AnnotatedVariant
    JSON structures. Preserve order and de-duplicate.
    """
    values: List[str] = []

    # Prefer explicit protein/alteration fields; "variant" is a fallback.
    for keys in (
        ["proteinChange", "protein_change", "hgvsp", "alteration"],
        ["variantName", "variant_name"],
        ["variant"],
    ):
        for value in find_values_by_keys(record, keys):
            values.extend(scalar_strings(value))
        if values:
            break

    # Clean obvious p. prefix/parentheses but otherwise preserve API notation.
    seen = set()
    cleaned: List[str] = []
    for value in values:
        s = value.strip()
        s = re.sub(r"^p\.\(?", "", s, flags=re.I)
        s = re.sub(r"\)$", "", s)
        if s and s not in seen:
            seen.add(s)
            cleaned.append(s)
    return cleaned


def _exact_one_letter_missense(s: str) -> bool:
    return bool(re.fullmatch(rf"[{AA1}]\d+[{AA1}]", s, flags=re.I))


def _exact_three_letter_missense(s: str) -> bool:
    return bool(re.fullmatch(rf"{AA3_RE}\d+{AA3_RE}", s, flags=re.I))


def expand_explicit_missense_alteration(value: str) -> List[str]:
    """
    Convert an OncoKB alteration string to one or more explicit missense HGVSp-like
    protein substitutions when this can be done without inference.

    Accepted examples:
      V600E
      p.V600E
      Val600Glu
      V600E/K       -> V600E, V600K
      V600E/K/R     -> V600E, V600K, V600R
      V600E, V600K  -> V600E, V600K

    Umbrella/non-specific terms such as "V600 mutations", "Truncating Mutations",
    "Oncogenic Mutations", deletions, frameshifts, fusions, and CNAs are excluded.
    """
    if not value:
        return []

    s = value.strip()
    s = re.sub(r"^p\.\(?", "", s, flags=re.I)
    s = re.sub(r"\)$", "", s)
    s = re.sub(r"\s+", " ", s).strip()

    low = s.lower()
    bad_terms = (
        "mutation", "truncat", "frameshift", "fusion", "amplif", "deletion",
        "insertion", "duplication", "splice", "promoter", "rearrang",
        "wildtype", "wild-type", "exon", "in-frame", "inframe",
    )
    if any(x in low for x in bad_terms):
        return []
    if any(x in s for x in ("*", "=", "?", "_", ">")):
        return []
    if re.search(r"(?:del|dup|ins|fs|ter)$", low):
        return []

    if _exact_one_letter_missense(s) or _exact_three_letter_missense(s):
        return [normalize_protein_change("p." + s)]

    # Several fully specified variants in one record.
    if "," in s or ";" in s:
        pieces = re.split(r"[,;]\s*", s)
        expanded: List[str] = []
        for piece in pieces:
            sub = expand_explicit_missense_alteration(piece)
            if not sub:
                return []
            expanded.extend(sub)
        return list(dict.fromkeys(expanded))

    # Compact one-letter grouped alternatives, e.g. V600E/K/R.
    m = re.fullmatch(
        rf"([{AA1}])(\d+)([{AA1}](?:/[{AA1}])+)",
        s,
        flags=re.I,
    )
    if m:
        ref, pos, alts = m.groups()
        return [
            normalize_protein_change(f"p.{ref.upper()}{pos}{alt.upper()}")
            for alt in alts.split("/")
        ]

    # Compact three-letter grouped alternatives, e.g. Val600Glu/Lys.
    m = re.fullmatch(
        rf"({AA3_RE})(\d+)({AA3_RE}(?:/{AA3_RE})+)",
        s,
        flags=re.I,
    )
    if m:
        ref, pos, alts = m.groups()
        return [
            normalize_protein_change(f"p.{ref}{pos}{alt}")
            for alt in alts.split("/")
        ]

    return []


def extract_pmids(record: Dict[str, Any]) -> List[str]:
    pmids = set()

    for key, value in walk_json(record):
        nk = normalize_key(key)
        if "pmid" in nk or nk in {"pubmed", "pubmedids"}:
            for s in scalar_strings(value):
                for hit in re.findall(r"(?<!\d)(\d{5,9})(?!\d)", s):
                    pmids.add(hit)

    # Citations sometimes contain PMID strings under a generic citations field.
    for value in find_values_by_keys(record, ["citations", "citation"]):
        try:
            text = json.dumps(value, ensure_ascii=False)
        except Exception:
            text = str(value)
        for hit in re.findall(r"(?<!\d)(\d{5,9})(?!\d)", text):
            pmids.add(hit)

    return sorted(pmids, key=lambda x: int(x))


def extract_oncogenicity(record: Dict[str, Any]) -> str:
    return first_string(
        record,
        ["oncogenic", "oncogenicity", "oncogenicEffect", "oncogenic_effect"],
    )


def extract_consequence(record: Dict[str, Any]) -> str:
    return first_string(
        record,
        ["consequence", "variantClassification", "variant_classification", "variantType"],
    )


def extract_description(record: Dict[str, Any]) -> str:
    return first_string(
        record,
        [
            "mutationEffectDescription",
            "mutation_effect_description",
            "variantSummary",
            "description",
        ],
    )


def fetch_json(
    session: requests.Session,
    url: str,
    timeout: int,
) -> Any:
    log(f"GET {url}")
    r = session.get(url, timeout=timeout)

    if r.status_code == 401:
        raise RuntimeError(
            "OncoKB returned HTTP 401 (Unauthorized). Check ONCOKB_API_TOKEN "
            "and confirm that your API access is active."
        )
    if r.status_code == 403:
        raise RuntimeError(
            "OncoKB returned HTTP 403 (Forbidden) for the bulk annotated-variant "
            "endpoint. Your account/license may not permit bulk retrieval. OncoKB "
            "officially warns that bulk download of all annotated variants may not "
            "be supported. This script will not fabricate an 'all OncoKB' dataset "
            "from an incomplete substitute."
        )
    if r.status_code == 404:
        raise RuntimeError(
            "OncoKB returned HTTP 404 for /api/v1/utils/allAnnotatedVariants. "
            "The bulk utility endpoint may have been removed or disabled on this "
            "instance. Exhaustive OncoKB variant discovery cannot be reproduced "
            "with the per-variant annotation endpoint alone."
        )

    r.raise_for_status()

    try:
        return r.json()
    except Exception as exc:
        preview = r.text[:500].replace("\n", " ")
        raise RuntimeError(
            f"OncoKB returned a non-JSON response from {url}: {preview}"
        ) from exc


def load_or_download_bulk(
    session: requests.Session,
    base_url: str,
    workdir: Path,
    timeout: int,
    force: bool,
) -> Tuple[Any, Path]:
    workdir.mkdir(parents=True, exist_ok=True)
    cache = workdir / "allAnnotatedVariants.json"

    if cache.exists() and cache.stat().st_size > 0 and not force:
        log(f"Using cached OncoKB bulk JSON: {cache}")
        return json.loads(cache.read_text(encoding="utf-8")), cache

    url = base_url.rstrip("/") + BULK_ENDPOINT
    payload = fetch_json(session, url, timeout)
    cache.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log(f"Saved raw OncoKB bulk JSON: {cache}")
    return payload, cache


def unwrap_records(payload: Any) -> List[Dict[str, Any]]:
    """
    Support both a bare JSON list and common wrapper forms without depending on a
    single OncoKB software-version serialization shape.
    """
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]

    if isinstance(payload, dict):
        for key in (
            "annotatedVariants",
            "variants",
            "data",
            "results",
            "items",
            "content",
        ):
            value = payload.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]

    raise RuntimeError(
        "Unexpected JSON shape from OncoKB allAnnotatedVariants endpoint. "
        f"Top-level type={type(payload).__name__}; "
        f"keys={list(payload.keys())[:30] if isinstance(payload, dict) else 'n/a'}"
    )


def build_directionality_table(
    records: List[Dict[str, Any]],
    source_url: str,
    include_likely: bool,
    include_raw_json: bool,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    rows: List[Dict[str, Any]] = []
    stats = {
        "bulk_records": len(records),
        "records_missing_gene": 0,
        "records_missing_effect": 0,
        "records_non_gof_lof_effect": 0,
        "records_likely_excluded": 0,
        "records_without_explicit_missense": 0,
        "expanded_output_rows_before_dedup": 0,
    }

    for idx, record in enumerate(records):
        gene = extract_gene(record)
        if not gene:
            stats["records_missing_gene"] += 1
            continue

        effect = extract_effect(record)
        if not effect:
            stats["records_missing_effect"] += 1
            continue

        label, confidence = classify_effect(effect)
        if label not in {"gof_like", "lof"}:
            stats["records_non_gof_lof_effect"] += 1
            continue

        if not include_likely and effect.strip().lower().startswith("likely"):
            stats["records_likely_excluded"] += 1
            continue

        alterations: List[str] = []
        raw_alterations = extract_alteration_candidates(record)
        for raw_alt in raw_alterations:
            alterations.extend(expand_explicit_missense_alteration(raw_alt))
        alterations = list(dict.fromkeys(x for x in alterations if x))

        if not alterations:
            stats["records_without_explicit_missense"] += 1
            continue

        pmids = extract_pmids(record)
        oncogenic = extract_oncogenicity(record)
        consequence = extract_consequence(record)
        description = extract_description(record)

        for protein in alterations:
            row = {
                "source_database": "OncoKB",
                "source_record_id": f"{gene}:{protein}",
                "source_version": "",
                "source_url": source_url,
                "HugoSymbol": gene,
                "ProteinChange": protein,
                "GenomeAssembly": "",
                "directionality_label_normalized": label,
                "label_confidence": confidence,
                "label_source": "allAnnotatedVariants.mutationEffect",
                "label_evidence": (
                    f"mutationEffect={effect}; oncogenic={oncogenic}; "
                    f"bulk_curated_variant=true"
                ),
                "is_explicit_variant_level": True,
                "publication_ids": ";".join(pmids),
                "disease_context": "",
                "raw_directionality": oncogenic,
                "raw_effect": effect,
                "oncokb_consequence": consequence,
                "oncokb_mutation_effect_description": description,
                "oncokb_bulk_record_index": idx,
                "oncokb_raw_alteration": ";".join(raw_alterations),
            }
            if include_raw_json:
                row["oncokb_raw_json"] = json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            rows.append(row)

    stats["expanded_output_rows_before_dedup"] = len(rows)

    if not rows:
        return ensure_canonical_columns(canonical_empty_frame()), stats

    df = pd.DataFrame(rows)

    # The same variant may occur more than once in bulk data due to evidence/context.
    # Keep distinct GOF/LOF effects separate first, then consolidate exact duplicates.
    dedup_cols = [
        "HugoSymbol",
        "ProteinChange",
        "directionality_label_normalized",
        "raw_effect",
    ]
    df = df.drop_duplicates(dedup_cols).reset_index(drop=True)

    # Flag variants with conflicting OncoKB directionality across retained records.
    conflict_counts = (
        df.groupby(["HugoSymbol", "ProteinChange"])["directionality_label_normalized"]
        .nunique()
    )
    conflicting_keys = set(conflict_counts[conflict_counts > 1].index)
    df["directionality_conflict_within_oncokb"] = [
        (g, p) in conflicting_keys
        for g, p in zip(df["HugoSymbol"], df["ProteinChange"])
    ]

    df = ensure_canonical_columns(df)
    return df, stats


def main() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--out", required=True, type=Path)
    p.add_argument(
        "--workdir",
        type=Path,
        default=Path("oncokb_cache"),
        help="Cache directory; raw allAnnotatedVariants JSON is kept here.",
    )
    p.add_argument("--base-url", default="https://www.oncokb.org")
    p.add_argument("--token-env", default="ONCOKB_API_TOKEN")
    p.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Use demo.oncokb.org. This is only a limited demo dataset and MUST NOT "
            "be interpreted as the full OncoKB collection."
        ),
    )
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download the bulk JSON even if a cached snapshot exists.",
    )
    p.add_argument(
        "--exclude-likely",
        action="store_true",
        help=(
            "Keep only exact Gain-of-function/Loss-of-function and exclude "
            "Likely Gain/Loss-of-function."
        ),
    )
    p.add_argument(
        "--include-raw-json-column",
        action="store_true",
        help="Store each retained source record as JSON in the output parquet.",
    )
    p.add_argument("--summary-json", type=Path)
    args = p.parse_args()

    token = "" if args.demo else os.environ.get(args.token_env, "").strip()
    if not args.demo and not token:
        raise SystemExit(
            f"Missing OncoKB token in environment variable {args.token_env}. "
            "Set the token before running. Use --demo only for testing; demo output "
            "is not the full OncoKB dataset."
        )

    base = "https://demo.oncokb.org" if args.demo else args.base_url.rstrip("/")
    session = session_with_retry(token)

    payload, raw_cache = load_or_download_bulk(
        session=session,
        base_url=base,
        workdir=args.workdir,
        timeout=args.timeout,
        force=args.force,
    )
    records = unwrap_records(payload)
    log(f"Loaded {len(records):,} records from OncoKB bulk endpoint")

    df, filter_stats = build_directionality_table(
        records=records,
        source_url=base + BULK_ENDPOINT,
        include_likely=not args.exclude_likely,
        include_raw_json=args.include_raw_json_column,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_parquet_safe(df, args.out)

    summary = {
        "source": "OncoKB",
        "retrieval_endpoint": BULK_ENDPOINT,
        "raw_cache": str(raw_cache),
        "demo_mode": bool(args.demo),
        "include_likely": not args.exclude_likely,
        **filter_stats,
        "output_rows": int(len(df)),
        "unique_variants": int(
            df[["HugoSymbol", "ProteinChange"]].drop_duplicates().shape[0]
        ) if len(df) else 0,
        "unique_genes": int(df["HugoSymbol"].nunique()) if len(df) else 0,
        "labels": (
            df["directionality_label_normalized"].value_counts().to_dict()
            if len(df)
            else {}
        ),
        "raw_effects": (
            df["raw_effect"].value_counts().to_dict()
            if len(df) and "raw_effect" in df.columns
            else {}
        ),
        "conflicting_variants": int(
            df["directionality_conflict_within_oncokb"].sum()
        ) if len(df) and "directionality_conflict_within_oncokb" in df.columns else 0,
    }

    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
