#!/usr/bin/env python3
"""
Build a protein-level GOF directionality table from GoFCards.

Main output: a parquet file, one row per GoFCards variant record after optional
filtering/standardization. GoFCards is a gain-of-function resource, so the
normalized directionality label emitted by this script is `gof` / `gof_like`.

The script is intentionally defensive because GoFCards is a Vue/Spring web app
and public download endpoints may change. It tries, in order:

1. A user-supplied local file via --input-file.
2. A user-supplied direct URL via --download-url.
3. Automatic discovery from the GoFCards web app/download pages/assets.

Examples
--------
# Fully automatic, using the official GoFCards site/download discovery
python gofcards_directionality_table.py \
  --out gofcards_gof_variants.parquet \
  --workdir gofcards_cache \
  --missense-only \
  --curated-only \
  --summary-json gofcards_summary.json

# More reproducible: pass a direct download URL you copied from GoFCards
python gofcards_directionality_table.py \
  --download-url 'https://...' \
  --out gofcards_gof_variants.parquet

# Local file mode
python gofcards_directionality_table.py \
  --input-file GoFCards_download.tsv \
  --out gofcards_gof_variants.parquet

Notes
-----
- GoFCards is GOF-focused. This script does not create LOF labels.
- It marks records as high confidence when it detects curation/literature fields
  such as PMID, evidence, experimental model, curated/manual status, or explicit
  GOF curation indicators.
- Predicted-only records can be retained as weak labels or removed with
  --curated-only.
- Always inspect the output summary and column mappings before training.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import re
import sys
import time
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd
import requests

DEFAULT_BASE_URLS = [
    "https://www.genemed.tech/gofcards/",
    "http://www.genemed.tech/gofcards/",
    "https://www.genemed.tech/gofcards",
    "http://www.genemed.tech/gofcards",
]

DEFAULT_DOWNLOAD_PAGES = [
    "#/genemed/download",
    "#/download",
    "download",
    "downloads",
    "data",
    "api",
]

USER_AGENT = (
    "Mozilla/5.0 (compatible; gofcards-directionality-builder/1.0; "
    "+https://openai.com; research data download)"
)

DOWNLOAD_EXTENSIONS = (
    ".csv", ".tsv", ".txt", ".xlsx", ".xls", ".zip", ".gz", ".bgz", ".tar.gz"
)

MISSENSE_TERMS = (
    "missense", "nonsynonymous", "non-synonymous", "non synonymous",
    "protein_altering", "substitution", "snv", "snp"
)

PREDICTED_TERMS = ("predicted", "prediction", "logofunc", "computational", "in_silico", "in silico")
CURATED_TERMS = (
    "curated", "manual", "literature", "pubmed", "pmid", "experimental",
    "experiment", "validated", "validation"
)
FALSE_LIKE = {"", "0", "false", "f", "no", "n", "none", "nan", "na", "unknown", "not available"}
TRUE_LIKE = {"1", "true", "t", "yes", "y", "validated", "curated", "manual"}
VALID_ALLELE_RE = re.compile(r"^[ACGTN]+$", flags=re.I)
AA_TOKEN_RE = r"(?:Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|Leu|Lys|Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val|[ACDEFGHIKLMNPQRSTVWY])"
PREDICTED_FIELD_TOKENS = ("prediction", "predicted", "logofunc", "computational", "model", "score")


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_colname(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(c).strip().lower())


def find_col(df: pd.DataFrame, candidates: Sequence[str], required: bool = False) -> Optional[str]:
    norm_to_real = {normalize_colname(c): c for c in df.columns}
    for cand in candidates:
        hit = norm_to_real.get(normalize_colname(cand))
        if hit is not None:
            return hit
    for real in df.columns:
        nreal = normalize_colname(real)
        for cand in candidates:
            ncand = normalize_colname(cand)
            if ncand and ncand in nreal:
                return real
    if required:
        raise KeyError(
            f"Could not find any of columns {candidates}. "
            f"Available columns: {list(df.columns)[:80]}"
        )
    return None


def safe_str_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str)


def make_parquet_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy whose object columns have a single Arrow-safe string type.

    Excel columns such as chromosome commonly contain both numbers (1-22) and
    strings (X/Y/MT). Pandas therefore stores them as ``object`` with mixed
    Python types, which can make PyArrow infer ``int64`` and then fail when it
    reaches a value such as ``X``. Converting object columns to pandas' nullable
    string dtype preserves missing values and makes Parquet serialization stable.
    """
    safe = df.copy()
    object_cols = list(safe.select_dtypes(include=["object"]).columns)
    for col in object_cols:
        safe[col] = safe[col].astype("string")
    if object_cols:
        log(f"Converted {len(object_cols)} object columns to nullable strings for Parquet output")
    return safe


def best_sep_from_bytes(content: bytes, fallback: str = ",") -> str:
    sample = content[:100000].decode("utf-8", errors="replace")
    counts = {sep: sample.count(sep) for sep in ["\t", ",", ";", "|"]}
    return max(counts, key=counts.get) if max(counts.values()) > 0 else fallback


def sniff_filename_from_response(resp: requests.Response, url: str) -> str:
    cd = resp.headers.get("content-disposition", "")
    m = re.search(r"filename\*?=(?:UTF-8''|\")?([^\";]+)", cd, flags=re.I)
    if m:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", m.group(1).strip())
    parsed = urlparse(url)
    name = Path(parsed.path).name or "downloaded_gofcards_data"
    if "." not in name:
        ctype = resp.headers.get("content-type", "").lower()
        if "excel" in ctype or "spreadsheet" in ctype:
            name += ".xlsx"
        elif "tab" in ctype:
            name += ".tsv"
        else:
            name += ".csv"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def request_get(session: requests.Session, url: str, timeout: int = 120) -> requests.Response:
    resp = session.get(url, timeout=timeout, allow_redirects=True)
    resp.raise_for_status()
    return resp


def download_url(url: str, outdir: Path, force: bool = False, session: Optional[requests.Session] = None) -> Path:
    ensure_dir(outdir)
    session = session or make_session()
    log(f"Downloading: {url}")
    with session.get(url, stream=True, timeout=300, allow_redirects=True) as resp:
        resp.raise_for_status()
        name = sniff_filename_from_response(resp, url)
        out = outdir / name
        if out.exists() and out.stat().st_size > 0 and not force:
            log(f"Using cached file: {out}")
            return out
        with open(out, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    log(f"Saved: {out}")
    return out


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml,text/csv,text/tab-separated-values,application/json,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.8",
        "Connection": "keep-alive",
    })
    return s


def extract_urls_from_text(text: str, base_url: str) -> List[str]:
    urls: List[str] = []
    # src/href links
    for m in re.finditer(r"(?:href|src)=[\"']([^\"']+)[\"']", text, flags=re.I):
        urls.append(urljoin(base_url, m.group(1)))
    # quoted paths/URLs that look like data/download/API files
    for m in re.finditer(r"[\"']([^\"']+(?:csv|tsv|txt|xlsx|xls|zip|gz|download|api|gof|variant)[^\"']*)[\"']", text, flags=re.I):
        val = m.group(1)
        if val.startswith(("http://", "https://", "/", "./", "../")):
            urls.append(urljoin(base_url, val))
    # bare URLs
    for m in re.finditer(r"https?://[^\s\"'<>]+", text, flags=re.I):
        urls.append(m.group(0).rstrip(",);"))
    # de-duplicate
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def url_looks_downloadable(url: str) -> bool:
    low = url.lower()
    return (
        any(low.split("?")[0].endswith(ext) for ext in DOWNLOAD_EXTENSIONS)
        or any(tok in low for tok in ["download", "export", "file", "variant", "gof"])
    )


def score_candidate_url(url: str) -> int:
    low = url.lower()
    score = 0
    for tok, val in [
        ("gof", 5), ("variant", 5), ("snv", 3), ("missense", 3),
        ("curated", 4), ("download", 4), ("data", 2), ("all", 1),
        ("gene", -2), ("image", -5), ("slide", -5), ("logo", -5),
        ("css", -10), ("js", -2),
    ]:
        if tok in low:
            score += val
    if any(low.split("?")[0].endswith(ext) for ext in DOWNLOAD_EXTENSIONS):
        score += 10
    return score


def discover_gofcards_download_urls(base_urls: Sequence[str], max_asset_fetches: int = 50) -> List[str]:
    session = make_session()
    visited = set()
    candidate_downloads: List[str] = []
    asset_urls: List[str] = []

    initial_urls = []
    for base in base_urls:
        initial_urls.append(base)
        for page in DEFAULT_DOWNLOAD_PAGES:
            initial_urls.append(urljoin(base.rstrip("/") + "/", page))

    for url in initial_urls:
        if url in visited:
            continue
        visited.add(url)
        try:
            log(f"Inspecting GoFCards page: {url}")
            resp = request_get(session, url, timeout=60)
        except Exception as e:
            log(f"Could not fetch {url}: {e}")
            continue
        ctype = resp.headers.get("content-type", "").lower()
        if any(resp.url.lower().split("?")[0].endswith(ext) for ext in DOWNLOAD_EXTENSIONS):
            candidate_downloads.append(resp.url)
            continue
        if "text" in ctype or "html" in ctype or "javascript" in ctype or not ctype:
            text = resp.text
            urls = extract_urls_from_text(text, resp.url)
            for u in urls:
                ulow = u.lower().split("?")[0]
                if ulow.endswith((".js", ".json")) or "/static/" in ulow or "/assets/" in ulow:
                    asset_urls.append(u)
                if url_looks_downloadable(u):
                    candidate_downloads.append(u)

    # Fetch JavaScript/assets and look for API/data/download URLs inside.
    for asset in sorted(set(asset_urls), key=score_candidate_url, reverse=True)[:max_asset_fetches]:
        if asset in visited:
            continue
        visited.add(asset)
        try:
            log(f"Inspecting GoFCards asset: {asset}")
            resp = request_get(session, asset, timeout=60)
        except Exception as e:
            log(f"Could not fetch asset {asset}: {e}")
            continue
        text = resp.text
        for u in extract_urls_from_text(text, resp.url):
            if url_looks_downloadable(u):
                candidate_downloads.append(u)

    # Common Spring-style endpoints to try. These are intentionally best-effort.
    common_paths = [
        "/gofcards/download",
        "/gofcards/api/download",
        "/gofcards/api/download/variant",
        "/gofcards/api/download/variants",
        "/gofcards/api/export",
        "/gofcards/api/variant/download",
        "/gofcards/api/variants/download",
        "/gofcards/genemed/download",
        "/gofcards/genemed/api/download",
        "/api/gofcards/download",
        "/api/gofcards/variants/download",
        "/api/download",
        "/api/variant/download",
    ]
    parsed_bases = []
    for base in base_urls:
        p = urlparse(base)
        if p.scheme and p.netloc:
            parsed_bases.append(f"{p.scheme}://{p.netloc}")
    for origin in sorted(set(parsed_bases)):
        for p in common_paths:
            candidate_downloads.append(urljoin(origin, p))

    # Rank and de-duplicate.
    seen = set()
    ranked = []
    for u in sorted(candidate_downloads, key=score_candidate_url, reverse=True):
        if u not in seen:
            seen.add(u)
            ranked.append(u)
    return ranked


def try_download_discovered(candidates: Sequence[str], outdir: Path, force: bool = False) -> Path:
    session = make_session()
    errors = []
    for url in candidates:
        try:
            resp = session.get(url, stream=True, timeout=120, allow_redirects=True)
            if resp.status_code >= 400:
                errors.append(f"{url}: HTTP {resp.status_code}")
                continue
            ctype = resp.headers.get("content-type", "").lower()
            first = next(resp.iter_content(chunk_size=4096), b"")
            # Avoid saving HTML pages unless URL has a data extension.
            lower_url = resp.url.lower().split("?")[0]
            looks_file = any(lower_url.endswith(ext) for ext in DOWNLOAD_EXTENSIONS)
            if (b"<html" in first.lower() or "text/html" in ctype) and not looks_file:
                errors.append(f"{url}: looked like HTML, not data")
                continue
            name = sniff_filename_from_response(resp, resp.url)
            out = outdir / name
            if out.exists() and out.stat().st_size > 0 and not force:
                return out
            ensure_dir(outdir)
            with open(out, "wb") as f:
                f.write(first)
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            if out.stat().st_size == 0:
                errors.append(f"{url}: empty download")
                continue
            log(f"Downloaded discovered GoFCards data file: {out}")
            return out
        except Exception as e:
            errors.append(f"{url}: {e}")
            continue
    msg = "Could not automatically download a GoFCards data file.\n"
    msg += "Tried candidate URLs such as:\n"
    msg += "\n".join("  - " + e for e in errors[:30])
    msg += "\n\nOpen the GoFCards Download page in a browser and pass the direct file URL with --download-url, "
    msg += "or save the file locally and pass --input-file."
    raise RuntimeError(msg)


def decompress_if_needed(path: Path, workdir: Path) -> Path:
    ensure_dir(workdir)
    name = path.name.lower()

    if name.endswith((".gz", ".bgz")) and not name.endswith(".tar.gz"):
        suffix_len = 4 if name.endswith(".bgz") else 3
        out = workdir / path.name[:-suffix_len]
        if not out.exists() or out.stat().st_size == 0:
            log(f"Decompressing gzip/BGZF-compatible stream: {path}")
            with gzip.open(path, "rb") as fin, open(out, "wb") as fout:
                while True:
                    chunk = fin.read(1024 * 1024)
                    if not chunk:
                        break
                    fout.write(chunk)
        return out

    if name.endswith(".tar.gz"):
        outdir = workdir / path.name[:-7]
        ensure_dir(outdir)
        with tarfile.open(path, mode="r:gz") as tf:
            members = [m for m in tf.getmembers() if m.isfile()]
            members = sorted(members, key=lambda m: score_candidate_url(m.name), reverse=True)
            for member in members:
                low = member.name.lower()
                if low.endswith(DOWNLOAD_EXTENSIONS) and not low.endswith((".zip", ".gz", ".bgz", ".tar.gz")):
                    out = outdir / Path(member.name).name
                    if not out.exists() or out.stat().st_size == 0:
                        extracted = tf.extractfile(member)
                        if extracted is None:
                            continue
                        log(f"Extracting {member.name} from {path}")
                        with extracted, open(out, "wb") as fout:
                            while True:
                                chunk = extracted.read(1024 * 1024)
                                if not chunk:
                                    break
                                fout.write(chunk)
                    return out
        raise RuntimeError(f"Tar archive did not contain a recognized tabular data file: {path}")

    if name.endswith(".zip"):
        outdir = workdir / (path.stem + "_unzipped")
        ensure_dir(outdir)
        with zipfile.ZipFile(path) as zf:
            members = [m for m in zf.namelist() if not m.endswith("/")]
            members = sorted(members, key=lambda m: score_candidate_url(m), reverse=True)
            for member in members:
                low = member.lower()
                if low.endswith(DOWNLOAD_EXTENSIONS) and not low.endswith((".zip", ".gz", ".bgz", ".tar.gz")):
                    out = outdir / Path(member).name
                    if not out.exists() or out.stat().st_size == 0:
                        log(f"Extracting {member} from {path}")
                        with zf.open(member) as fin, open(out, "wb") as fout:
                            while True:
                                chunk = fin.read(1024 * 1024)
                                if not chunk:
                                    break
                                fout.write(chunk)
                    return out
        raise RuntimeError(f"Zip file did not contain a recognized tabular data file: {path}")
    return path

def read_any_table(path: Path, sheet: Optional[str] = None) -> pd.DataFrame:
    path = Path(path)
    low = path.name.lower()
    if low.endswith((".xlsx", ".xls")):
        if sheet is None:
            xls = pd.ExcelFile(path)
            # Choose sheet with the most variant-like name, otherwise first sheet.
            sheets = sorted(xls.sheet_names, key=score_candidate_url, reverse=True)
            sheet = sheets[0]
        log(f"Reading Excel sheet {sheet!r} from {path}")
        return pd.read_excel(path, sheet_name=sheet)
    content = path.read_bytes()[:200000]
    sep = "\t" if low.endswith((".tsv", ".txt", ".maf")) else best_sep_from_bytes(content)
    log(f"Reading table {path} with sep={repr(sep)}")
    return pd.read_csv(path, sep=sep, low_memory=False, comment="#")


def hgvsp_to_three_letter(protein: pd.Series) -> pd.Series:
    # Keep as-is; this helper mainly normalizes prefixes and whitespace.
    return (
        protein.fillna("").astype(str)
        .str.strip()
        .str.replace(r"^p\.\(?", "p.", regex=True)
        .str.replace(r"\)$", "", regex=True)
    )


def derive_missense_flag(df: pd.DataFrame) -> pd.Series:
    candidate_cols = [
        find_col(df, ["variant_type", "VariantType", "type", "mutation_type"]),
        find_col(df, ["consequence", "Consequence", "variant_effect", "annotation", "function", "effect"]),
        find_col(df, ["ProteinChange", "protein_change", "HGVSp", "aa_change", "amino_acid_change", "variant"]),
    ]
    text = pd.Series("", index=df.index)
    for c in [c for c in candidate_cols if c is not None]:
        text = text + " " + safe_str_series(df[c])
    low = text.str.lower()
    # HGVSp missense-like: p.A123B, p.Arg123His, not fs/*/delins.
    prot_col = find_col(df, ["ProteinChange", "protein_change", "HGVSp", "HGVSp_short", "aa_change", "amino_acid_change", "variant"])
    prot = safe_str_series(df[prot_col]) if prot_col else pd.Series("", index=df.index)
    hgvsp_missense = prot.str.contains(
        rf"p\.\(?{AA_TOKEN_RE}\d+{AA_TOKEN_RE}\)?$",
        regex=True,
        na=False,
    )
    not_missense = low.str.contains(r"frameshift|fs|stop|nonsense|splice|delins|deletion|duplication|insertion|indel", regex=True, na=False)
    return low.str.contains("|".join(re.escape(t) for t in MISSENSE_TERMS), regex=True, na=False) | (hgvsp_missense & ~not_missense)


def nonempty_col(df: pd.DataFrame, names: Sequence[str], *, allow_partial: bool = False) -> pd.Series:
    norm_to_real = {normalize_colname(c): c for c in df.columns}
    col: Optional[str] = None
    for name in names:
        col = norm_to_real.get(normalize_colname(name))
        if col is not None:
            break
    if col is None and allow_partial:
        col = find_col(df, names)
    if col is None:
        return pd.Series(False, index=df.index)
    values = safe_str_series(df[col]).str.strip()
    return values.ne("") & ~values.str.lower().isin(FALSE_LIKE)


def _first_exact_col(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    norm_to_real = {normalize_colname(c): c for c in df.columns}
    for name in names:
        hit = norm_to_real.get(normalize_colname(name))
        if hit is not None:
            return hit
    return None


def _positive_evidence_value(series: pd.Series) -> pd.Series:
    text = safe_str_series(series).str.strip().str.lower()
    explicit_positive = text.isin(TRUE_LIKE)
    experimental_phrase = text.str.contains(
        r"\b(experimental(?:ly)?|functional assay|validated|validation|curated|manual|literature-supported)\b",
        regex=True,
        na=False,
    )
    predicted_phrase = text.str.contains(
        r"\b(predicted|prediction|computational|in[ _-]?silico|logofunc|model score)\b",
        regex=True,
        na=False,
    )
    return (explicit_positive | experimental_phrase) & ~predicted_phrase & ~text.isin(FALSE_LIKE)


def derive_curated_and_confidence(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Derive conservative evidence tiers without treating field presence as curation."""
    combined = pd.Series("", index=df.index, dtype="string")
    predicted_text = pd.Series("", index=df.index, dtype="string")
    for col in df.columns:
        normalized = normalize_colname(col)
        if any(token.replace("_", "") in normalized for token in CURATED_TERMS):
            combined = combined.fillna("") + " " + safe_str_series(df[col])
        if any(token.replace("_", "") in normalized for token in PREDICTED_FIELD_TOKENS):
            predicted_text = predicted_text.fillna("") + " " + safe_str_series(df[col])
    low = combined.str.lower()
    predicted_low = predicted_text.str.lower()

    pmid_col = _first_exact_col(df, ["pmid", "pubmed", "pubmed_id", "pubmedid", "publication_id"])
    has_pmid = pd.Series(False, index=df.index)
    if pmid_col is not None:
        pmid_text = safe_str_series(df[pmid_col]).str.strip()
        has_pmid = pmid_text.str.contains(r"(?:^|\D)\d{5,9}(?:\D|$)", regex=True, na=False)

    status_col = _first_exact_col(
        df,
        ["curation_status", "status", "evidence_type", "label_source", "data_source", "source_type"],
    )
    status = safe_str_series(df[status_col]).str.lower() if status_col else pd.Series("", index=df.index)
    status_curated = status.str.contains(
        r"\b(?:curated|manual|literature|experimental|validated)\b",
        regex=True,
        na=False,
    ) & ~status.str.contains(r"\b(?:predicted|computational|in[ _-]?silico)\b", regex=True, na=False)

    positive_evidence = pd.Series(False, index=df.index)
    for name in ["functional_evidence", "gof_evidence", "experimental_evidence", "validation", "validated", "experiment"]:
        col = _first_exact_col(df, [name])
        if col is not None:
            positive_evidence |= _positive_evidence_value(df[col])

    predicted = (
        predicted_low.str.contains(r"\b(?:predicted|prediction|logofunc|computational|in[ _-]?silico)\b", regex=True, na=False)
        | status.str.contains(r"\b(?:predicted|prediction|logofunc|computational|in[ _-]?silico)\b", regex=True, na=False)
    )
    curated = has_pmid | status_curated | positive_evidence

    confidence = np.select(
        [curated, predicted],
        ["high", "low_predicted"],
        default="medium_unspecified",
    )

    evidence: List[str] = []
    for idx in df.index:
        bits: List[str] = []
        if bool(has_pmid.loc[idx]):
            bits.append("valid PMID present")
        if bool(status_curated.loc[idx]):
            bits.append("explicit curated/manual/literature/experimental status")
        if bool(positive_evidence.loc[idx]):
            bits.append("positive experimental/validation evidence value")
        if bool(predicted.loc[idx]):
            bits.append("predicted/computational evidence detected")
        evidence.append("; ".join(bits) if bits else "GoFCards GOF record; evidence tier unspecified")
    return curated.astype(bool), pd.Series(confidence, index=df.index), pd.Series(evidence, index=df.index)

def standardize_gofcards_table(raw: pd.DataFrame, source_file: str, missense_only: bool, curated_only: bool) -> pd.DataFrame:
    df = raw.copy()
    # Clean entirely empty columns.
    df = df.dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]

    gene_col = find_col(df, ["gene", "gene_symbol", "symbol", "hugo", "hugo_symbol", "GeneSymbol"])
    prot_col = find_col(df, ["protein_change", "ProteinChange", "HGVSp", "HGVSp_short", "aa_change", "amino_acid_change", "protein", "variant"])
    dna_col = find_col(df, ["cdna_change", "DNAChange", "HGVSc", "nucleotide_change", "cDNA", "dna_change"])
    transcript_col = find_col(df, ["transcript", "transcript_id", "refseq", "ensembl_transcript", "NM"])
    chrom_col = find_col(df, ["chrom", "chromosome", "chr"])
    pos_col = find_col(df, ["pos", "position", "start", "start_position", "coordinate"])
    ref_col = _first_exact_col(df, ["ref", "reference_allele", "referenceallele", "Ref"])
    alt_col = _first_exact_col(df, ["alt", "alternate", "alternate_allele", "variant_allele", "Alt"])
    disease_col = find_col(df, ["disease", "phenotype", "trait", "condition", "associated_disease"])
    pmid_col = _first_exact_col(df, ["pmid", "pubmed", "pubmed_id", "publication_id", "references", "citation"])
    score_col = find_col(df, ["gof_score", "score", "prediction_score", "logofunc_score", "GOFScore"])

    out = pd.DataFrame(index=df.index)
    out["source_database"] = "GoFCards"
    out["source_file"] = source_file
    out["directionality_label"] = "gof"
    out["directionality_label_normalized"] = "gof"
    out["binary_label_lof0_gof1"] = 1
    out["binary_label_name"] = "gof"

    def col_or_empty(c: Optional[str]) -> pd.Series:
        return safe_str_series(df[c]) if c is not None else pd.Series("", index=df.index)

    out["HugoSymbol"] = col_or_empty(gene_col).str.upper()
    out["ProteinChange"] = hgvsp_to_three_letter(col_or_empty(prot_col))
    out["DNAChange"] = col_or_empty(dna_col)
    out["TranscriptID"] = col_or_empty(transcript_col)
    out["Chrom"] = col_or_empty(chrom_col)
    out["Pos"] = col_or_empty(pos_col)
    out["Ref"] = col_or_empty(ref_col).str.upper()
    out["Alt"] = col_or_empty(alt_col).str.upper()
    valid_ref = out["Ref"].str.fullmatch(VALID_ALLELE_RE, na=False)
    valid_alt = out["Alt"].str.fullmatch(VALID_ALLELE_RE, na=False)
    out.loc[~valid_ref, "Ref"] = ""
    out.loc[~valid_alt, "Alt"] = ""
    out["DiseaseOrPhenotype"] = col_or_empty(disease_col)
    out["PMID_or_reference"] = col_or_empty(pmid_col)

    if score_col is not None:
        out["gofcards_score"] = pd.to_numeric(df[score_col], errors="coerce")
    else:
        out["gofcards_score"] = np.nan

    missense = derive_missense_flag(df)
    curated, confidence, evidence = derive_curated_and_confidence(df)
    out["is_missense_like"] = missense
    out["is_curated_or_literature_supported"] = curated
    out["label_confidence"] = confidence
    out["label_evidence"] = evidence
    out["source_tier"] = np.select(
        [curated, out["label_confidence"].eq("low_predicted")],
        ["tier1_direct_curated_gof", "tier3_predicted_gof"],
        default="tier2_gofcards_unspecified",
    )

    # Variant keys.
    out["protein_variant_key"] = np.where(
        out["HugoSymbol"].ne("") & out["ProteinChange"].ne(""),
        out["HugoSymbol"] + ":" + out["ProteinChange"],
        "",
    )
    out["genomic_variant_key"] = np.where(
        out["Chrom"].ne("") & out["Pos"].ne("") & out["Ref"].ne("") & out["Alt"].ne(""),
        out["Chrom"] + ":" + out["Pos"] + ":" + out["Ref"] + ">" + out["Alt"],
        "",
    )

    # Preserve all original columns with a prefix to avoid collisions.
    for c in df.columns:
        cname = "gofcards_raw__" + re.sub(r"[^A-Za-z0-9_]+", "_", str(c).strip())
        if cname not in out.columns:
            out[cname] = df[c]

    # Drop obvious non-variant rows.
    has_some_variant_id = (
        out["ProteinChange"].ne("")
        | out["DNAChange"].ne("")
        | out["genomic_variant_key"].ne("")
    )
    dropped_unidentified = int((~has_some_variant_id).sum())
    if dropped_unidentified:
        log(f"Dropping {dropped_unidentified:,} rows without a protein, cDNA, or complete genomic variant identifier")
    out = out.loc[has_some_variant_id].copy()

    if missense_only:
        out = out.loc[out["is_missense_like"]].copy()
    if curated_only:
        out = out.loc[out["is_curated_or_literature_supported"]].copy()

    # Drop exact duplicate rows on best available keys; keep raw-column duplicates if no keys exist.
    dedup_cols = [c for c in ["protein_variant_key", "genomic_variant_key", "DNAChange", "DiseaseOrPhenotype", "PMID_or_reference"] if c in out.columns]
    if dedup_cols:
        out = out.drop_duplicates(dedup_cols)

    return out.reset_index(drop=True)


def write_summary(df: pd.DataFrame, summary_json: Optional[Path], summary_csv: Optional[Path]) -> None:
    summary = {
        "n_rows": int(len(df)),
        "n_unique_protein_variant_keys": int(df["protein_variant_key"].replace("", np.nan).nunique(dropna=True)) if "protein_variant_key" in df else None,
        "n_unique_genes": int(df["HugoSymbol"].replace("", np.nan).nunique(dropna=True)) if "HugoSymbol" in df else None,
        "label_counts": df["directionality_label_normalized"].value_counts(dropna=False).to_dict(),
        "confidence_counts": df["label_confidence"].value_counts(dropna=False).to_dict() if "label_confidence" in df else {},
        "source_tier_counts": df["source_tier"].value_counts(dropna=False).to_dict() if "source_tier" in df else {},
        "missense_counts": df["is_missense_like"].value_counts(dropna=False).astype(int).to_dict() if "is_missense_like" in df else {},
        "curated_counts": df["is_curated_or_literature_supported"].value_counts(dropna=False).astype(int).to_dict() if "is_curated_or_literature_supported" in df else {},
    }
    if summary_json is not None:
        ensure_dir(summary_json.parent if summary_json.parent != Path("") else Path("."))
        summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if summary_csv is not None:
        ensure_dir(summary_csv.parent if summary_csv.parent != Path("") else Path("."))
        rows = []
        for group in ["label_counts", "confidence_counts", "source_tier_counts", "missense_counts", "curated_counts"]:
            for k, v in summary.get(group, {}).items():
                rows.append({"group": group, "value": str(k), "count": int(v)})
        pd.DataFrame(rows).to_csv(summary_csv, index=False)
    log("Summary: " + json.dumps(summary, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--out", required=True, help="Output parquet path.")
    ap.add_argument("--workdir", default="gofcards_cache", help="Cache/download directory.")
    ap.add_argument("--input-file", default=None, help="Optional local GoFCards downloaded file. If supplied, no web download is attempted.")
    ap.add_argument("--download-url", default=None, help="Optional direct GoFCards download URL. More reproducible than auto-discovery.")
    ap.add_argument("--base-url", action="append", default=None, help="GoFCards base URL(s) for auto-discovery. Can be repeated.")
    ap.add_argument("--force-download", action="store_true", help="Re-download even if cached.")
    ap.add_argument("--excel-sheet", default=None, help="Excel sheet name if the downloaded/local file is xlsx/xls.")
    ap.add_argument("--missense-only", action="store_true", help="Keep only missense-like protein variants.")
    ap.add_argument("--curated-only", action="store_true", help="Drop predicted-only/unspecified records; keep literature/experimental/curated-like records.")
    ap.add_argument("--summary-json", default=None, help="Optional JSON summary output.")
    ap.add_argument("--summary-csv", default=None, help="Optional CSV summary output.")
    ap.add_argument("--save-raw-copy", default=None, help="Optional path to save raw input table as parquet after reading.")
    args = ap.parse_args()

    workdir = Path(args.workdir)
    ensure_dir(workdir)

    if args.input_file:
        data_path = Path(args.input_file)
        if not data_path.exists():
            raise FileNotFoundError(data_path)
    elif args.download_url:
        data_path = download_url(args.download_url, workdir / "downloads", force=args.force_download)
    else:
        base_urls = args.base_url or DEFAULT_BASE_URLS
        log("Discovering GoFCards download URLs from website/assets")
        candidates = discover_gofcards_download_urls(base_urls)
        log(f"Discovered {len(candidates)} candidate URLs. Top candidates: {candidates[:10]}")
        data_path = try_download_discovered(candidates, workdir / "downloads", force=args.force_download)

    data_path = decompress_if_needed(Path(data_path), workdir / "extracted")
    raw = read_any_table(data_path, sheet=args.excel_sheet)
    log(f"Loaded raw GoFCards table: {raw.shape[0]:,} rows x {raw.shape[1]:,} columns")

    if args.save_raw_copy:
        raw_out = Path(args.save_raw_copy)
        ensure_dir(raw_out.parent if raw_out.parent != Path("") else Path("."))
        make_parquet_safe(raw).to_parquet(raw_out, index=False)

    out_df = standardize_gofcards_table(
        raw,
        source_file=str(data_path),
        missense_only=args.missense_only,
        curated_only=args.curated_only,
    )

    out = Path(args.out)
    ensure_dir(out.parent if out.parent != Path("") else Path("."))
    log(f"Writing parquet: {out}")
    make_parquet_safe(out_df).to_parquet(out, index=False)

    write_summary(
        out_df,
        Path(args.summary_json) if args.summary_json else None,
        Path(args.summary_csv) if args.summary_csv else None,
    )
    log("Done.")


if __name__ == "__main__":
    main()
