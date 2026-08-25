#!/usr/bin/env python3
"""
Build a DepMap mutation directionality table with conservative LOF labels,
heuristic GOF-like labels, useful metadata, and optional drug-response support.

Main output: a parquet file, one row per DepMap mutation record.

Examples
--------
# DepMap-only labels using latest DepMap Public release discovered from manifest
python depmap_directionality_table.py \
  --out depmap_mutation_directionality.parquet \
  --workdir depmap_cache

# Pin a release
python depmap_directionality_table.py \
  --release "DepMap Public 26Q1" \
  --out depmap_26Q1_directionality.parquet

# Add CancerMine gene roles
python depmap_directionality_table.py \
  --use-cancermine \
  --out depmap_directionality.cancermine.parquet

# Add drug-response support from a local wide drug-response matrix and metadata
python depmap_directionality_table.py \
  --use-cancermine \
  --drug-response-matrix PRISM_secondary.csv \
  --drug-metadata PRISM_compound_info.csv \
  --drug-lower-is-more-sensitive \
  --out depmap_directionality.with_drug.parquet

Notes
-----
- DepMap's mutation file supports LOF much more directly than GOF. This script
  therefore emits two labels:
    1) directionality_label_conservative: lof / unknown
    2) directionality_label_heuristic: lof / gof_like / ambiguous / unknown
- Drug response is treated only as support for GOF-like interpretation. It does
  not override LOF annotations.
- GOF-like labels are heuristic and should be described that way in manuscripts.
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
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

try:
    from scipy.stats import mannwhitneyu
except Exception:  # pragma: no cover
    mannwhitneyu = None

DEP_MAP_MANIFEST_URL = "https://depmap.org/portal/api/download/files"
ZENODO_RECORD_API = "https://zenodo.org/api/records/7689627"  # CancerMine v50 record; file discovery is dynamic.
FIGSHARE_API_BASE = "https://api.figshare.com/v2"
# DepMap moved some recent releases off Figshare, so this map intentionally contains
# releases known to have public Figshare/ Figshare+ records. The portal manifest is
# still used first in --download-source auto mode.
FIGSHARE_RELEASE_ARTICLE_IDS = {
    "24Q4": 27993248,
    "DepMap Public 24Q4": 27993248,
    "DepMap 24Q4 Public": 27993248,
    "24Q2": 25880521,
    "DepMap Public 24Q2": 25880521,
    "DepMap 24Q2 Public": 25880521,
}

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; depmap-directionality-table/0.3; +https://depmap.org/)",
    "Accept": "text/csv,application/json,text/plain,*/*",
}

SESSION = requests.Session()
SESSION.headers.update(REQUEST_HEADERS)

TRUNCATING_TERMS = (
    "frameshift",
    "stop_gained",
    "stop lost",
    "stop_lost",
    "start_lost",
    "splice_acceptor",
    "splice_donor",
    "splice_region",
    "transcript_ablation",
    "nonsense",
    "essential_splice",
)

MISSENSE_TERMS = ("missense", "inframe", "protein_altering")

BOOL_TRUE = {"true", "t", "1", "yes", "y"}
BOOL_FALSE = {"false", "f", "0", "no", "n", "", "nan", "none"}


@dataclass
class DownloadedFile:
    name: str
    path: Path
    release: Optional[str] = None
    source_url: Optional[str] = None


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
    # partial fallback
    for real in df.columns:
        nreal = normalize_colname(real)
        for cand in candidates:
            ncand = normalize_colname(cand)
            if ncand and ncand in nreal:
                return real
    if required:
        raise KeyError(f"Could not find any of columns {candidates}. Available columns: {list(df.columns)[:50]}...")
    return None


def as_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(float).ne(0)
    x = s.astype(str).str.strip().str.lower()
    return x.isin(BOOL_TRUE)


def safe_str_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str)


def read_table(path: Path, **kwargs) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path, **kwargs)
    if suffix in {".tsv"}:
        return pd.read_csv(path, sep="\t", low_memory=False, **kwargs)
    if suffix in {".maf"}:
        return pd.read_csv(path, sep="\t", comment="#", low_memory=False, **kwargs)
    return pd.read_csv(path, low_memory=False, **kwargs)


def fetch_manifest(workdir: Path, force: bool = False) -> pd.DataFrame:
    ensure_dir(workdir)
    out = workdir / "depmap_download_manifest.csv"
    if out.exists() and not force:
        log(f"Using cached DepMap manifest: {out}")
        return pd.read_csv(out)
    log(f"Downloading DepMap file manifest from {DEP_MAP_MANIFEST_URL}")
    r = SESSION.get(DEP_MAP_MANIFEST_URL, timeout=120)
    if r.status_code == 403:
        raise RuntimeError(
            "DepMap portal manifest returned 403 Forbidden. This is usually portal-side bot/download protection. "
            "Use --download-source figshare for a fully standalone public Figshare release, or pass local files."
        )
    r.raise_for_status()
    out.write_bytes(r.content)
    return pd.read_csv(io.BytesIO(r.content))


def release_to_figshare_article_id(release: Optional[str], article_id: Optional[int] = None) -> Tuple[str, int]:
    if article_id is not None:
        return (release or f"figshare_article_{article_id}"), int(article_id)
    if release is None:
        # Latest release known to be on Figshare/Figshare+ at the time this script was written.
        return "DepMap Public 24Q4", FIGSHARE_RELEASE_ARTICLE_IDS["24Q4"]
    rel = str(release).strip()
    if rel in FIGSHARE_RELEASE_ARTICLE_IDS:
        return rel, FIGSHARE_RELEASE_ARTICLE_IDS[rel]
    m = re.search(r"(\d{2})\s*Q([1-4])", rel, flags=re.I)
    if m:
        key = f"{m.group(1)}Q{m.group(2)}"
        if key in FIGSHARE_RELEASE_ARTICLE_IDS:
            return rel, FIGSHARE_RELEASE_ARTICLE_IDS[key]
    raise ValueError(
        f"No built-in Figshare article id for release={release!r}. "
        "Use --figshare-article-id with the numeric article id from the Figshare/Figshare+ URL, "
        "or use --download-source portal if the DepMap portal manifest is reachable."
    )


def fetch_figshare_article(article_id: int, workdir: Path, force: bool = False) -> dict:
    ensure_dir(workdir)
    out = workdir / f"figshare_article_{article_id}.json"
    if out.exists() and not force:
        log(f"Using cached Figshare metadata: {out}")
        return json.loads(out.read_text())
    url = f"{FIGSHARE_API_BASE}/articles/{article_id}"
    log(f"Downloading Figshare metadata from {url}")
    r = SESSION.get(url, timeout=120)
    r.raise_for_status()
    out.write_text(r.text)
    return r.json()


def select_figshare_file(article: dict, filename: str) -> dict:
    files = article.get("files", [])
    if not files:
        raise FileNotFoundError("Figshare article metadata contains no files.")
    # Exact first, then case-insensitive, then substring; also allow .gz/.zip wrappers.
    names = [(f, str(f.get("name", ""))) for f in files]
    target = filename.lower()
    for f, name in names:
        if name == filename:
            return f
    for f, name in names:
        if name.lower() == target:
            return f
    for f, name in names:
        nl = name.lower()
        if nl == target + ".gz" or nl == target + ".zip":
            return f
    for f, name in names:
        if target in name.lower():
            return f
    available = [name for _, name in names[:50]]
    raise FileNotFoundError(f"Could not find {filename!r} in Figshare article. First files: {available}")


def download_from_figshare(
    article: dict,
    filename: str,
    workdir: Path,
    release: Optional[str] = None,
    force: bool = False,
) -> DownloadedFile:
    f = select_figshare_file(article, filename)
    real_name = str(f.get("name"))
    rel = release or str(article.get("title", "figshare_release"))
    safe_rel = re.sub(r"[^A-Za-z0-9_.-]+", "_", rel or "figshare_release")
    outdir = workdir / safe_rel
    ensure_dir(outdir)
    outpath = outdir / real_name
    if outpath.exists() and outpath.stat().st_size > 0 and not force:
        log(f"Using cached Figshare file: {outpath}")
        return DownloadedFile(real_name, outpath, rel, f.get("download_url"))
    url = f.get("download_url")
    if not url:
        fid = f.get("id")
        if fid is None:
            raise ValueError(f"Figshare file has neither download_url nor id: {f}")
        url = f"https://figshare.com/ndownloader/files/{fid}"
    log(f"Downloading {real_name} from Figshare [{rel}]")
    with SESSION.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(outpath, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
    return DownloadedFile(real_name, outpath, rel, url)


def parse_release_key(release: str) -> Tuple[int, int]:
    """Sort keys for strings such as 'DepMap Public 26Q1' or 'DepMap Public 2024 Q4'."""
    s = str(release)
    m = re.search(r"(20)?(\d{2})\s*Q([1-4])", s, flags=re.I)
    if not m:
        return (-1, -1)
    year2 = int(m.group(2))
    year = 2000 + year2 if year2 < 80 else 1900 + year2
    quarter = int(m.group(3))
    return (year, quarter)


def manifest_columns(manifest: pd.DataFrame) -> Tuple[str, str, Optional[str], Optional[str]]:
    file_col = find_col(manifest, ["file_name", "filename", "name", "file", "File Name"], required=True)
    url_col = find_col(manifest, ["url", "download_url", "downloadUrl", "Download URL", "signed_url"], required=True)
    release_col = find_col(manifest, ["release", "releasename", "release_name", "Release", "Release Name"])
    size_col = find_col(manifest, ["size", "filesize", "file_size", "Size"])
    return file_col, url_col, release_col, size_col


def latest_depmap_public_release(manifest: pd.DataFrame) -> str:
    file_col, _, release_col, _ = manifest_columns(manifest)
    if release_col is None:
        raise ValueError("Manifest has no release column. Please pass --release explicitly.")
    m = manifest.copy()
    rels = sorted(
        [r for r in m[release_col].dropna().unique() if "depmap public" in str(r).lower()],
        key=parse_release_key,
    )
    if not rels:
        raise ValueError("Could not find DepMap Public releases in the manifest.")
    return rels[-1]


def select_manifest_row(manifest: pd.DataFrame, filename: str, release: Optional[str] = None) -> pd.Series:
    file_col, url_col, release_col, _ = manifest_columns(manifest)
    m = manifest.copy()
    # Exact filename first, then case-insensitive exact basename.
    exact = m[m[file_col].astype(str).eq(filename)]
    if exact.empty:
        exact = m[m[file_col].astype(str).str.lower().eq(filename.lower())]
    if exact.empty:
        exact = m[m[file_col].astype(str).str.contains(re.escape(filename), case=False, regex=True, na=False)]
    if release is not None and release_col is not None and not exact.empty:
        exact_release = exact[exact[release_col].astype(str).eq(release)]
        if exact_release.empty:
            exact_release = exact[exact[release_col].astype(str).str.contains(re.escape(release), case=False, regex=True, na=False)]
        exact = exact_release
    if exact.empty:
        available = m[file_col].dropna().astype(str).head(30).tolist()
        raise FileNotFoundError(f"Could not find {filename!r} in manifest for release={release!r}. First files: {available}")
    if release is None and release_col is not None:
        exact = exact.assign(_release_key=exact[release_col].map(parse_release_key)).sort_values("_release_key")
    return exact.iloc[-1]


def download_from_manifest(
    manifest: pd.DataFrame,
    filename: str,
    workdir: Path,
    release: Optional[str] = None,
    force: bool = False,
) -> DownloadedFile:
    file_col, url_col, release_col, _ = manifest_columns(manifest)
    row = select_manifest_row(manifest, filename, release=release)
    real_name = str(row[file_col])
    rel = str(row[release_col]) if release_col and not pd.isna(row[release_col]) else release
    safe_rel = re.sub(r"[^A-Za-z0-9_.-]+", "_", rel or "unknown_release")
    outdir = workdir / safe_rel
    ensure_dir(outdir)
    outpath = outdir / real_name
    if outpath.exists() and outpath.stat().st_size > 0 and not force:
        log(f"Using cached file: {outpath}")
        return DownloadedFile(real_name, outpath, rel, str(row[url_col]))
    log(f"Downloading {real_name} [{rel}]")
    url = str(row[url_col])
    with SESSION.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(outpath, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return DownloadedFile(real_name, outpath, rel, url)


def download_cancermine_collated(workdir: Path, force: bool = False) -> Path:
    out = workdir / "cancermine_collated.tsv"
    if out.exists() and out.stat().st_size > 0 and not force:
        log(f"Using cached CancerMine: {out}")
        return out
    log("Discovering CancerMine collated TSV from Zenodo")
    r = SESSION.get(ZENODO_RECORD_API, timeout=120)
    r.raise_for_status()
    record = r.json()
    files = record.get("files", [])
    hit = None
    for f in files:
        key = f.get("key", "")
        if key.endswith("cancermine_collated.tsv") or "cancermine_collated" in key:
            hit = f
            break
    if hit is None:
        raise FileNotFoundError("Could not find cancermine_collated.tsv in Zenodo record.")
    url = hit["links"]["self"]
    rr = SESSION.get(url, timeout=300)
    rr.raise_for_status()
    out.write_bytes(rr.content)
    return out


def load_gene_roles_from_cancermine(path: Path, min_citations: int = 1) -> pd.DataFrame:
    cm = pd.read_csv(path, sep="\t", low_memory=False)
    gene_col = find_col(cm, ["gene_normalized", "gene", "symbol", "hugo_symbol"], required=True)
    role_col = find_col(cm, ["role", "cancer_role", "classification"], required=True)
    cit_col = find_col(cm, ["citation_count", "citations", "num_citations", "n_citations"])
    cm[gene_col] = cm[gene_col].astype(str).str.upper()
    if cit_col is not None:
        cm = cm[pd.to_numeric(cm[cit_col], errors="coerce").fillna(0) >= min_citations]
    role_text = cm[role_col].astype(str).str.lower()
    cm["is_oncogene_cm"] = role_text.str.contains("oncogene|proto-oncogene|gof|gain", regex=True, na=False)
    cm["is_tsg_cm"] = role_text.str.contains("tumou?r suppress|tsg|suppressor|lof|loss", regex=True, na=False)
    agg = cm.groupby(gene_col, as_index=False).agg(
        cancermine_oncogene=("is_oncogene_cm", "max"),
        cancermine_tumor_suppressor=("is_tsg_cm", "max"),
    )
    agg = agg.rename(columns={gene_col: "HugoSymbol_upper"})
    agg["cancermine_gene_role"] = np.select(
        [agg["cancermine_oncogene"] & agg["cancermine_tumor_suppressor"], agg["cancermine_oncogene"], agg["cancermine_tumor_suppressor"]],
        ["both", "oncogene", "tumor_suppressor"],
        default="unknown",
    )
    return agg


def first_existing_bool(df: pd.DataFrame, names: Sequence[str]) -> pd.Series:
    col = find_col(df, names)
    if col is None:
        return pd.Series(False, index=df.index)
    return as_bool_series(df[col])


def contains_any(df: pd.DataFrame, names: Sequence[str], terms: Sequence[str]) -> pd.Series:
    col = find_col(df, names)
    if col is None:
        return pd.Series(False, index=df.index)
    pattern = "|".join(re.escape(t) for t in terms)
    return safe_str_series(df[col]).str.lower().str.contains(pattern, regex=True, na=False)


def derive_gene_role(mut: pd.DataFrame) -> pd.DataFrame:
    """Make row-level gene-role fields from DepMap row flags and optional CancerMine columns."""
    onc_hi = first_existing_bool(mut, ["OncogeneHighImpact", "Oncogene High Impact"])
    tsg_hi = first_existing_bool(mut, ["TumorSuppressorHighImpact", "Tumor Suppressor High Impact"])
    cm_onc = mut.get("cancermine_oncogene", pd.Series(False, index=mut.index)).fillna(False).astype(bool)
    cm_tsg = mut.get("cancermine_tumor_suppressor", pd.Series(False, index=mut.index)).fillna(False).astype(bool)

    is_oncogene = onc_hi | cm_onc
    is_tsg = tsg_hi | cm_tsg
    role = np.select(
        [is_oncogene & is_tsg, is_oncogene, is_tsg],
        ["both", "oncogene", "tumor_suppressor"],
        default="unknown",
    )
    mut["gene_role"] = role
    mut["gene_role_source"] = np.select(
        [onc_hi | tsg_hi, cm_onc | cm_tsg],
        ["depmap_oncokb_high_impact_flags", "cancermine"],
        default="none",
    )
    return mut


def derive_directionality(mut: pd.DataFrame) -> pd.DataFrame:
    mut = mut.copy()

    likely_lof = first_existing_bool(mut, ["LikelyLoF", "Likely LoF", "Likely_LOF"])
    hotspot = first_existing_bool(mut, ["Hotspot", "IsHotspot"])
    hess_driver = first_existing_bool(mut, ["HessDriver", "Hess Driver"])
    onc_hi = first_existing_bool(mut, ["OncogeneHighImpact", "Oncogene High Impact"])
    tsg_hi = first_existing_bool(mut, ["TumorSuppressorHighImpact", "Tumor Suppressor High Impact"])

    vep_high = False
    vep_col = find_col(mut, ["VepImpact", "VEP_Impact", "IMPACT"])
    if vep_col is not None:
        vep_high = safe_str_series(mut[vep_col]).str.upper().eq("HIGH")
    else:
        vep_high = pd.Series(False, index=mut.index)

    truncating = contains_any(mut, ["VariantInfo", "Consequence", "Variant_Classification"], TRUNCATING_TERMS)
    missense = contains_any(mut, ["VariantInfo", "Consequence", "Variant_Classification"], MISSENSE_TERMS)

    mut["depmap_likely_lof_flag"] = likely_lof
    mut["depmap_vep_high_flag"] = vep_high
    mut["depmap_hotspot_flag"] = hotspot
    mut["depmap_hess_driver_flag"] = hess_driver
    mut["depmap_oncogene_high_impact_flag"] = onc_hi
    mut["depmap_tumor_suppressor_high_impact_flag"] = tsg_hi
    mut["variant_truncating_or_splice_flag"] = truncating
    mut["variant_missense_or_inframe_flag"] = missense

    # Conservative: only directly LOF-oriented DepMap flags.
    conservative_lof = likely_lof | tsg_hi | (vep_high & truncating)
    mut["directionality_label_conservative"] = np.where(conservative_lof, "lof", "unknown")

    role = mut["gene_role"].fillna("unknown").astype(str)
    is_onc = role.isin(["oncogene", "both"])
    is_tsg = role.isin(["tumor_suppressor", "both"])

    # Heuristic hierarchy, ordered from most defensible to ambiguous.
    lof = conservative_lof | (is_tsg & truncating)
    gof_like = (~lof) & is_onc & (hotspot | hess_driver) & (missense | onc_hi)
    ambiguous = (
        ((hotspot | hess_driver) & is_tsg & missense)
        | (lof & is_onc & (hotspot | onc_hi))
        | (onc_hi & tsg_hi)
    )

    mut["directionality_label_heuristic"] = np.select(
        [ambiguous, lof, gof_like],
        ["ambiguous", "lof", "gof_like"],
        default="unknown",
    )

    evidence = []
    for i in mut.index:
        bits = []
        if bool(likely_lof.loc[i]):
            bits.append("DepMap LikelyLoF")
        if bool(tsg_hi.loc[i]):
            bits.append("DepMap tumor-suppressor high-impact")
        if bool(onc_hi.loc[i]):
            bits.append("DepMap oncogene high-impact")
        if bool(vep_high.loc[i]):
            bits.append("VEP HIGH")
        if bool(truncating.loc[i]):
            bits.append("truncating/splice consequence")
        if bool(hotspot.loc[i]):
            bits.append("DepMap Hotspot")
        if bool(hess_driver.loc[i]):
            bits.append("HessDriver")
        gr = role.loc[i]
        if gr != "unknown":
            bits.append(f"gene_role={gr}")
        evidence.append("; ".join(bits) if bits else "no direct directionality evidence")
    mut["directionality_evidence"] = evidence

    mut["directionality_confidence"] = np.select(
        [mut["directionality_label_conservative"].eq("lof"), gof_like & hotspot & is_onc, ambiguous],
        ["high", "medium", "low"],
        default="low",
    )
    return mut


def standardize_mutation_table(mut: pd.DataFrame, source_release: str) -> pd.DataFrame:
    mut = mut.copy()
    mut["source_release"] = source_release
    hugo = find_col(mut, ["HugoSymbol", "Hugo_Symbol", "gene", "symbol"], required=True)
    mut["HugoSymbol"] = mut[hugo].astype(str)
    mut["HugoSymbol_upper"] = mut["HugoSymbol"].str.upper()

    # Add canonical variant ID fields without dropping original columns.
    chrom = find_col(mut, ["Chrom", "Chromosome", "chr"])
    pos = find_col(mut, ["Pos", "Start_position", "Start", "position"])
    ref = find_col(mut, ["Ref", "Reference_Allele", "ref_allele"])
    alt = find_col(mut, ["Alt", "Tumor_Seq_Allele2", "alt_allele"])
    prot = find_col(mut, ["ProteinChange", "Protein_Change", "HGVSp_Short", "HGVSp"])
    dna = find_col(mut, ["DNAChange", "cDNA_Change", "HGVSc"])
    model = find_col(mut, ["ModelID", "DepMap_ID", "ModelConditionID", "CCLE_Name"])

    def get_or_empty(c: Optional[str]) -> pd.Series:
        return safe_str_series(mut[c]) if c is not None else pd.Series("", index=mut.index)

    mut["variant_key"] = (
        get_or_empty(chrom) + ":" + get_or_empty(pos) + ":" + get_or_empty(ref) + ">" + get_or_empty(alt)
    )
    mut["protein_key"] = mut["HugoSymbol"] + ":" + get_or_empty(prot)
    mut["dna_key"] = mut["HugoSymbol"] + ":" + get_or_empty(dna)
    if model is not None:
        mut["ModelID_standard"] = mut[model].astype(str)
    else:
        mut["ModelID_standard"] = ""
    return mut


def load_model_metadata(path: Optional[Path], manifest: Optional[pd.DataFrame], workdir: Path, release: str, force: bool) -> Optional[pd.DataFrame]:
    if path is None:
        try:
            dl = download_from_manifest(manifest, "Model.csv", workdir, release=release, force=force) if manifest is not None else None
            path = dl.path if dl is not None else None
        except Exception as e:
            log(f"Could not auto-download Model.csv; continuing without lineage metadata. Reason: {e}")
            return None
    model = read_table(path)
    mid = find_col(model, ["ModelID", "DepMap_ID", "ModelConditionID"], required=True)
    lineage = find_col(model, ["OncotreeLineage", "Lineage", "lineage", "PrimaryDisease", "OncotreePrimaryDisease"])
    cols = [mid] + ([lineage] if lineage else [])
    out = model[cols].copy().rename(columns={mid: "ModelID_standard"})
    if lineage:
        out = out.rename(columns={lineage: "model_lineage"})
    else:
        out["model_lineage"] = "unknown"
    out["ModelID_standard"] = out["ModelID_standard"].astype(str)
    return out.drop_duplicates("ModelID_standard")


def bh_fdr(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    mask = np.isfinite(p)
    if mask.sum() == 0:
        return out
    pm = p[mask]
    order = np.argsort(pm)
    ranked = pm[order]
    n = len(ranked)
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    tmp = np.empty(n, dtype=float)
    tmp[order] = q
    out[mask] = tmp
    return out


def infer_compound_columns(drug_meta: pd.DataFrame) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    id_col = find_col(drug_meta, ["compound_id", "broad_id", "pert_id", "drug_id", "name", "compound", "drug", "column"])
    if id_col is None:
        id_col = drug_meta.columns[0]
    name_col = find_col(drug_meta, ["name", "compound_name", "drug_name", "pert_iname", "compound"])
    target_col = find_col(drug_meta, ["target", "targets", "gene_target", "target_genes", "moa_target", "Target"])
    pathway_col = find_col(drug_meta, ["pathway", "target_pathway", "moa", "mechanism", "MOA"])
    return id_col, name_col, target_col, pathway_col


def add_drug_response_support(
    mut: pd.DataFrame,
    drug_matrix_path: Path,
    drug_metadata_path: Optional[Path],
    model_meta: Optional[pd.DataFrame],
    lower_is_more_sensitive: bool,
    min_mutant_models: int,
    min_wt_models: int,
    max_candidates: int,
    fdr_threshold: float,
    min_effect_size: float,
) -> pd.DataFrame:
    log("Loading drug-response matrix")
    drug = read_table(drug_matrix_path)
    mid = find_col(drug, ["ModelID", "DepMap_ID", "ModelConditionID", "CCLE_Name", "cell_line", "CellLine"], required=True)
    drug = drug.rename(columns={mid: "ModelID_standard"})
    drug["ModelID_standard"] = drug["ModelID_standard"].astype(str)
    if model_meta is not None:
        drug = drug.merge(model_meta[["ModelID_standard", "model_lineage"]], on="ModelID_standard", how="left")
    else:
        drug["model_lineage"] = "unknown"

    non_drug_cols = {"ModelID_standard", "model_lineage"}
    numeric_cols = [c for c in drug.columns if c not in non_drug_cols and pd.api.types.is_numeric_dtype(drug[c])]
    if not numeric_cols:
        # Try coercing all non-ID columns.
        for c in drug.columns:
            if c not in non_drug_cols:
                drug[c] = pd.to_numeric(drug[c], errors="coerce")
        numeric_cols = [c for c in drug.columns if c not in non_drug_cols and pd.api.types.is_numeric_dtype(drug[c])]
    log(f"Drug matrix: {drug.shape[0]} models x {len(numeric_cols)} numeric drug columns")

    target_map: Dict[str, Dict[str, str]] = {}
    if drug_metadata_path is not None:
        dm = read_table(drug_metadata_path)
        id_col, name_col, target_col, pathway_col = infer_compound_columns(dm)
        for _, row in dm.iterrows():
            cid = str(row[id_col])
            if cid in numeric_cols:
                target_map[cid] = {
                    "drug_name": str(row[name_col]) if name_col else cid,
                    "drug_target": str(row[target_col]) if target_col else "",
                    "drug_target_pathway": str(row[pathway_col]) if pathway_col else "",
                }
    else:
        log("No drug metadata supplied; cannot map inhibitors to targets. Drug-response GOF support will be skipped.")

    if not target_map:
        mut["drug_response_supports_gof"] = False
        mut["drug_response_gof_confidence"] = "not_evaluated_no_drug_target_metadata"
        mut["drug_response_evidence"] = "no target-mapped drug metadata supplied"
        return mut

    # Candidate variants: oncogene hotspot/gof_like rows, evaluated by protein_key.
    candidates = (
        mut.loc[mut["directionality_label_heuristic"].isin(["gof_like", "unknown", "ambiguous"]), ["HugoSymbol", "HugoSymbol_upper", "protein_key", "gene_role"]]
        .dropna()
        .drop_duplicates()
    )
    candidates = candidates[candidates["gene_role"].isin(["oncogene", "both"])]
    if len(candidates) > max_candidates:
        log(f"Limiting drug-response association candidates from {len(candidates)} to {max_candidates}")
        candidates = candidates.head(max_candidates)

    model_universe = set(drug["ModelID_standard"].dropna().astype(str))
    mut_models_by_gene = mut.groupby("HugoSymbol_upper")["ModelID_standard"].apply(lambda x: set(x.dropna().astype(str)))
    mut_models_by_variant = mut.groupby("protein_key")["ModelID_standard"].apply(lambda x: set(x.dropna().astype(str)))

    records = []
    for _, cand in candidates.iterrows():
        gene = str(cand["HugoSymbol_upper"]).upper()
        protein_key = str(cand["protein_key"])
        mutant_models = mut_models_by_variant.get(protein_key, set()) & model_universe
        wt_models = model_universe - (mut_models_by_gene.get(gene, set()) & model_universe)
        if len(mutant_models) < min_mutant_models or len(wt_models) < min_wt_models:
            continue

        # Select target-matched drugs: target field contains the gene symbol as a token.
        target_re = re.compile(rf"(^|[^A-Za-z0-9]){re.escape(gene)}([^A-Za-z0-9]|$)", flags=re.I)
        matched_drugs = [d for d, meta in target_map.items() if target_re.search(meta.get("drug_target", ""))]
        if not matched_drugs:
            continue

        mut_block = drug[drug["ModelID_standard"].isin(mutant_models)]
        wt_block = drug[drug["ModelID_standard"].isin(wt_models)]
        for dcol in matched_drugs:
            if dcol not in drug.columns:
                continue
            xm = pd.to_numeric(mut_block[dcol], errors="coerce").dropna()
            xw = pd.to_numeric(wt_block[dcol], errors="coerce").dropna()
            if len(xm) < min_mutant_models or len(xw) < min_wt_models:
                continue
            # Positive effect means mutant is more sensitive.
            raw_effect = xw.mean() - xm.mean() if lower_is_more_sensitive else xm.mean() - xw.mean()
            p = np.nan
            if mannwhitneyu is not None:
                try:
                    # one-sided: mutant more sensitive.
                    alt = "less" if lower_is_more_sensitive else "greater"
                    p = float(mannwhitneyu(xm, xw, alternative=alt).pvalue)
                except Exception:
                    p = np.nan
            meta = target_map[dcol]
            records.append(
                {
                    "protein_key": protein_key,
                    "HugoSymbol_upper": gene,
                    "drug_column": dcol,
                    "drug_name": meta.get("drug_name", dcol),
                    "drug_target": meta.get("drug_target", ""),
                    "drug_target_pathway": meta.get("drug_target_pathway", ""),
                    "target_matches_gene": True,
                    "n_mutant_models_drug": len(xm),
                    "n_wt_models_drug": len(xw),
                    "mutant_mean_response": float(xm.mean()),
                    "wt_mean_response": float(xw.mean()),
                    "mutant_vs_wt_effect_size_more_sensitive_positive": float(raw_effect),
                    "mutant_vs_wt_pvalue": p,
                }
            )

    if not records:
        mut["drug_response_supports_gof"] = False
        mut["drug_response_gof_confidence"] = "not_supported_or_not_evaluable"
        mut["drug_response_evidence"] = "no target-matched inhibitor sensitivity association passed evaluability filters"
        return mut

    assoc = pd.DataFrame(records)
    assoc["mutant_vs_wt_fdr"] = bh_fdr(assoc["mutant_vs_wt_pvalue"].values)
    assoc["passes_drug_gof_support"] = (
        assoc["mutant_vs_wt_effect_size_more_sensitive_positive"].ge(min_effect_size)
        & assoc["mutant_vs_wt_fdr"].le(fdr_threshold)
    )
    # Best association per protein change, prioritizing pass, effect, then FDR.
    assoc = assoc.sort_values(
        ["protein_key", "passes_drug_gof_support", "mutant_vs_wt_effect_size_more_sensitive_positive", "mutant_vs_wt_fdr"],
        ascending=[True, False, False, True],
    )
    best = assoc.groupby("protein_key", as_index=False).head(1)
    best_cols = [
        "protein_key",
        "drug_column",
        "drug_name",
        "drug_target",
        "drug_target_pathway",
        "target_matches_gene",
        "n_mutant_models_drug",
        "n_wt_models_drug",
        "mutant_mean_response",
        "wt_mean_response",
        "mutant_vs_wt_effect_size_more_sensitive_positive",
        "mutant_vs_wt_pvalue",
        "mutant_vs_wt_fdr",
        "passes_drug_gof_support",
    ]
    mut = mut.merge(best[best_cols], on="protein_key", how="left")
    mut["drug_response_supports_gof"] = mut["passes_drug_gof_support"].fillna(False).astype(bool)
    mut["drug_response_gof_confidence"] = np.where(mut["drug_response_supports_gof"], "high_target_matched", "not_supported_or_not_evaluable")
    mut["drug_response_evidence"] = np.where(
        mut["drug_response_supports_gof"],
        "target-matched inhibitor response: "
        + mut["drug_name"].fillna(mut["drug_column"].fillna("drug"))
        + "; target="
        + mut["drug_target"].fillna("")
        + "; effect="
        + mut["mutant_vs_wt_effect_size_more_sensitive_positive"].round(4).astype(str)
        + "; fdr="
        + mut["mutant_vs_wt_fdr"].round(4).astype(str),
        "no significant target-matched inhibitor support",
    )

    # Do not override LOF/ambiguous; create an additional integrated label.
    mut["directionality_label_with_drug_support"] = mut["directionality_label_heuristic"]
    idx = (
        mut["directionality_label_heuristic"].eq("unknown")
        & mut["drug_response_supports_gof"]
        & mut["gene_role"].isin(["oncogene", "both"])
    )
    mut.loc[idx, "directionality_label_with_drug_support"] = "gof_like_drug_supported"
    return mut


def maybe_auto_download_drug_files(manifest: pd.DataFrame, workdir: Path, release: str, force: bool) -> Tuple[Optional[Path], Optional[Path]]:
    """Best-effort only. DepMap/PRISM drug file names vary by release/file set."""
    file_col, _, release_col, _ = manifest_columns(manifest)
    m = manifest.copy()
    if release_col is not None:
        m = m[m[release_col].astype(str).str.contains(re.escape(release), case=False, regex=True, na=False)]
    names = m[file_col].astype(str).tolist()
    # Prefer matrices that sound like drug sensitivity / viability, not metadata.
    matrix_patterns = [
        r"PRISM.*(secondary|repurposing).*(auc|logfold|viability).*\.csv$",
        r".*drug.*(auc|response|sensitivity|viability).*\.csv$",
        r".*compound.*(auc|response|sensitivity|viability).*\.csv$",
    ]
    meta_patterns = [
        r"PRISM.*(compound|treatment|metadata|info).*\.csv$",
        r".*compound.*(metadata|info|annotation).*\.csv$",
        r".*drug.*(metadata|info|annotation).*\.csv$",
    ]
    def pick(patterns):
        for pat in patterns:
            for n in names:
                if re.search(pat, n, flags=re.I):
                    return n
        return None
    matrix_name = pick(matrix_patterns)
    meta_name = pick(meta_patterns)
    matrix_path = None
    meta_path = None
    if matrix_name:
        try:
            matrix_path = download_from_manifest(manifest, matrix_name, workdir, release=release, force=force).path
        except Exception as e:
            log(f"Auto drug matrix download failed for {matrix_name}: {e}")
    if meta_name:
        try:
            meta_path = download_from_manifest(manifest, meta_name, workdir, release=release, force=force).path
        except Exception as e:
            log(f"Auto drug metadata download failed for {meta_name}: {e}")
    return matrix_path, meta_path


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--release", default=None, help="DepMap release name. With --download-source portal/auto this can be e.g. 'DepMap Public 26Q1'. With --download-source figshare, built-in releases include 24Q4 and 24Q2 unless --figshare-article-id is supplied.")
    ap.add_argument("--download-source", choices=["auto", "portal", "figshare"], default="auto", help="Where to fetch DepMap files. auto tries the portal manifest first and falls back to Figshare if blocked.")
    ap.add_argument("--figshare-article-id", type=int, default=None, help="Optional Figshare/Figshare+ numeric article id, e.g. 27993248 for DepMap 24Q4 Public.")
    ap.add_argument("--workdir", default="depmap_directionality_cache", help="Cache/download directory.")
    ap.add_argument("--out", required=True, help="Output parquet path.")
    ap.add_argument("--force-download", action="store_true", help="Re-download files even when cached.")
    ap.add_argument("--mutation-file", default=None, help="Optional local OmicsSomaticMutations.csv/maf path. If omitted, downloaded from DepMap manifest.")
    ap.add_argument("--model-file", default=None, help="Optional local Model.csv path for lineage metadata. If omitted, script tries to download it.")
    ap.add_argument("--use-cancermine", action="store_true", help="Augment gene role with CancerMine collated gene roles.")
    ap.add_argument("--cancermine-file", default=None, help="Optional local cancermine_collated.tsv path.")
    ap.add_argument("--cancermine-min-citations", type=int, default=1, help="Minimum CancerMine citation count when citation column exists.")
    ap.add_argument("--drug-response-matrix", default=None, help="Optional wide drug-response matrix path with rows=models and columns=drugs.")
    ap.add_argument("--drug-metadata", default=None, help="Optional drug metadata with drug IDs/names/targets matching matrix columns.")
    ap.add_argument("--auto-drug-download", action="store_true", help="Best-effort attempt to locate drug response and metadata files in DepMap manifest.")
    ap.add_argument("--drug-lower-is-more-sensitive", action="store_true", help="Set when lower metric means more sensitive, e.g. AUC or viability. If false, higher means more sensitive.")
    ap.add_argument("--min-mutant-models", type=int, default=3, help="Minimum mutated models for drug association.")
    ap.add_argument("--min-wt-models", type=int, default=20, help="Minimum wild-type models for drug association.")
    ap.add_argument("--max-drug-candidates", type=int, default=5000, help="Maximum protein-level variant candidates for drug association.")
    ap.add_argument("--drug-fdr-threshold", type=float, default=0.05, help="FDR threshold for drug support.")
    ap.add_argument("--drug-min-effect-size", type=float, default=0.0, help="Minimum sensitivity effect size; positive means mutant more sensitive.")
    args = ap.parse_args()

    workdir = Path(args.workdir)
    ensure_dir(workdir)

    manifest = None
    figshare_article = None
    using_figshare = False

    # If the user provides all required local files, do not touch the network for the DepMap manifest.
    if args.mutation_file:
        release = args.release or "local"
        log(f"Using local mutation file; release label: {release}")
    elif args.download_source == "figshare":
        release, article_id = release_to_figshare_article_id(args.release, args.figshare_article_id)
        figshare_article = fetch_figshare_article(article_id, workdir, force=args.force_download)
        using_figshare = True
        log(f"Using Figshare release: {release} [article_id={article_id}]")
    else:
        try:
            manifest = fetch_manifest(workdir, force=args.force_download)
            release = args.release or latest_depmap_public_release(manifest)
            log(f"Using DepMap portal release: {release}")
        except Exception as e:
            if args.download_source == "portal":
                raise
            log(f"Portal manifest unavailable ({e}). Falling back to public Figshare release.")
            release, article_id = release_to_figshare_article_id(args.release, args.figshare_article_id)
            figshare_article = fetch_figshare_article(article_id, workdir, force=args.force_download)
            using_figshare = True
            log(f"Using Figshare release: {release} [article_id={article_id}]")

    if args.mutation_file:
        mut_path = Path(args.mutation_file)
        source_release = release or "local"
    elif using_figshare:
        mut_dl = download_from_figshare(figshare_article, "OmicsSomaticMutations.csv", workdir, release=release, force=args.force_download)
        mut_path = mut_dl.path
        source_release = mut_dl.release or release
    else:
        mut_dl = download_from_manifest(manifest, "OmicsSomaticMutations.csv", workdir, release=release, force=args.force_download)
        mut_path = mut_dl.path
        source_release = mut_dl.release or release

    log(f"Loading mutation table: {mut_path}")
    mut = read_table(mut_path)
    mut = standardize_mutation_table(mut, source_release=source_release)

    if args.use_cancermine:
        cm_path = Path(args.cancermine_file) if args.cancermine_file else download_cancermine_collated(workdir, force=args.force_download)
        log(f"Loading CancerMine gene roles: {cm_path}")
        cm_roles = load_gene_roles_from_cancermine(cm_path, min_citations=args.cancermine_min_citations)
        mut = mut.merge(cm_roles, on="HugoSymbol_upper", how="left")
        mut["cancermine_oncogene"] = mut["cancermine_oncogene"].fillna(False)
        mut["cancermine_tumor_suppressor"] = mut["cancermine_tumor_suppressor"].fillna(False)
        mut["cancermine_gene_role"] = mut["cancermine_gene_role"].fillna("unknown")
    else:
        mut["cancermine_oncogene"] = False
        mut["cancermine_tumor_suppressor"] = False
        mut["cancermine_gene_role"] = "not_used"

    mut = derive_gene_role(mut)
    mut = derive_directionality(mut)

    if args.model_file:
        model_file_for_meta = Path(args.model_file)
    elif using_figshare:
        try:
            model_file_for_meta = download_from_figshare(figshare_article, "Model.csv", workdir, release=release, force=args.force_download).path
        except Exception as e:
            log(f"Could not auto-download Model.csv from Figshare; continuing without lineage metadata. Reason: {e}")
            model_file_for_meta = None
    else:
        model_file_for_meta = None

    model_meta = load_model_metadata(
        model_file_for_meta,
        manifest=manifest,
        workdir=workdir,
        release=release,
        force=args.force_download,
    )
    if model_meta is not None:
        mut = mut.merge(model_meta, on="ModelID_standard", how="left")

    drug_matrix = Path(args.drug_response_matrix) if args.drug_response_matrix else None
    drug_meta = Path(args.drug_metadata) if args.drug_metadata else None
    if args.auto_drug_download and drug_matrix is None:
        if manifest is None:
            log("--auto-drug-download currently requires the DepMap portal manifest; skipping because the run is using Figshare/local files.")
        else:
            dm, dmeta = maybe_auto_download_drug_files(manifest, workdir, release=release, force=args.force_download)
            drug_matrix = dm
            drug_meta = drug_meta or dmeta

    if drug_matrix is not None:
        mut = add_drug_response_support(
            mut=mut,
            drug_matrix_path=drug_matrix,
            drug_metadata_path=drug_meta,
            model_meta=model_meta,
            lower_is_more_sensitive=args.drug_lower_is_more_sensitive,
            min_mutant_models=args.min_mutant_models,
            min_wt_models=args.min_wt_models,
            max_candidates=args.max_drug_candidates,
            fdr_threshold=args.drug_fdr_threshold,
            min_effect_size=args.drug_min_effect_size,
        )
    else:
        mut["drug_response_supports_gof"] = False
        mut["drug_response_gof_confidence"] = "not_evaluated"
        mut["drug_response_evidence"] = "drug response not supplied"
        mut["directionality_label_with_drug_support"] = mut["directionality_label_heuristic"]

    # Prefer common useful columns first, preserve the rest afterward.
    preferred = [
        "source_release",
        "ModelID_standard",
        "model_lineage",
        "HugoSymbol",
        "HugoSymbol_upper",
        "gene_role",
        "gene_role_source",
        "cancermine_gene_role",
        "variant_key",
        "protein_key",
        "dna_key",
        "depmap_likely_lof_flag",
        "depmap_vep_high_flag",
        "depmap_hotspot_flag",
        "depmap_hess_driver_flag",
        "depmap_oncogene_high_impact_flag",
        "depmap_tumor_suppressor_high_impact_flag",
        "variant_truncating_or_splice_flag",
        "variant_missense_or_inframe_flag",
        "directionality_label_conservative",
        "directionality_label_heuristic",
        "directionality_label_with_drug_support",
        "directionality_confidence",
        "directionality_evidence",
        "drug_response_supports_gof",
        "drug_response_gof_confidence",
        "drug_response_evidence",
        "drug_name",
        "drug_target",
        "drug_target_pathway",
        "n_mutant_models_drug",
        "n_wt_models_drug",
        "mutant_vs_wt_effect_size_more_sensitive_positive",
        "mutant_vs_wt_pvalue",
        "mutant_vs_wt_fdr",
    ]
    preferred = [c for c in preferred if c in mut.columns]
    rest = [c for c in mut.columns if c not in preferred]
    mut = mut[preferred + rest]

    out = Path(args.out)
    ensure_dir(out.parent if out.parent != Path("") else Path("."))
    log(f"Writing parquet: {out}")
    mut.to_parquet(out, index=False)

    summary = mut["directionality_label_heuristic"].value_counts(dropna=False).to_dict()
    conservative = mut["directionality_label_conservative"].value_counts(dropna=False).to_dict()
    log(f"Done. Rows={len(mut):,}. Conservative labels={conservative}. Heuristic labels={summary}")


if __name__ == "__main__":
    main()

