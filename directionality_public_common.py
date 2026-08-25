#!/usr/bin/env python3
from __future__ import annotations

import gzip
import json
import re
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

CANONICAL_COLUMNS = [
    "source_database", "source_record_id", "source_version", "source_url",
    "HugoSymbol", "ProteinChange", "GenomeAssembly",
    "directionality_label_normalized", "label_confidence", "label_source",
    "label_evidence", "is_explicit_variant_level", "publication_ids",
    "disease_context", "raw_directionality", "raw_effect",
]

AA1 = "ACDEFGHIKLMNPQRSTVWY"
AA3 = (
    "Ala","Arg","Asn","Asp","Cys","Gln","Glu","Gly","His","Ile",
    "Leu","Lys","Met","Phe","Pro","Ser","Thr","Trp","Tyr","Val",
)
AA3_RE = "(?:" + "|".join(AA3) + ")"
HGVSP_RE = re.compile(
    rf"p\.\(?((?:[{AA1}])\d+(?:[{AA1}])|(?:{AA3_RE})\d+(?:{AA3_RE}))\)?(?![A-Za-z])",
    re.I,
)

def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def safe_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()

def safe_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()

def normalize_colname(x: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", safe_str(x).lower())

def find_col(df: pd.DataFrame, names: Sequence[str], required: bool=False) -> Optional[str]:
    exact = {normalize_colname(c): c for c in df.columns}
    for n in names:
        hit = exact.get(normalize_colname(n))
        if hit is not None:
            return hit
    for c in df.columns:
        nc = normalize_colname(c)
        for n in names:
            nn = normalize_colname(n)
            if nn and nn in nc:
                return c
    if required:
        raise KeyError(f"Could not find any of {list(names)}. Columns={list(df.columns)[:100]}")
    return None

def normalize_protein_change(x: Any) -> str:
    s = safe_str(x)
    if not s or s in {".", "-", "NA", "N/A"}:
        return ""
    m = HGVSP_RE.search(s)
    if m:
        return "p." + m.group(1)
    s2 = re.sub(r"^p\.\(?", "", s, flags=re.I)
    s2 = re.sub(r"\)$", "", s2)
    if re.fullmatch(rf"[{AA1}]\d+[{AA1}]", s2, flags=re.I):
        return "p." + s2
    if re.fullmatch(rf"{AA3_RE}\d+{AA3_RE}", s2, flags=re.I):
        return "p." + s2
    return ""

def extract_pmids(text: Any) -> str:
    hits = sorted(set(re.findall(r"(?<!\d)(\d{5,9})(?!\d)", safe_str(text))), key=int)
    return ";".join(hits)

def session_with_retry() -> requests.Session:
    retry = Retry(
        total=5, connect=5, read=5, status=5, backoff_factor=1.0,
        status_forcelist=(429,500,502,503,504),
        allowed_methods=frozenset({"GET","HEAD","POST"}),
        respect_retry_after_header=True,
    )
    s = requests.Session()
    a = HTTPAdapter(max_retries=retry)
    s.mount("https://", a)
    s.mount("http://", a)
    s.headers.update({"User-Agent":"public-directionality-collector/1.0","Accept":"*/*"})
    return s

def download(url: str, path: Path, force: bool=False, timeout: int=300) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0 and not force:
        return path
    s = session_with_retry()
    with s.get(url, stream=True, timeout=timeout, allow_redirects=True) as r:
        r.raise_for_status()
        cd = r.headers.get("content-disposition", "")
        m = re.search(r'filename\*?=(?:UTF-8\'\')?["\']?([^"\';]+)', cd, re.I)
        if m:
            path = path.parent / re.sub(r"[^A-Za-z0-9_.-]+", "_", m.group(1).strip())
        with path.open("wb") as f:
            for chunk in r.iter_content(1024*1024):
                if chunk:
                    f.write(chunk)
    return path

def read_table(path: Path) -> pd.DataFrame:
    p = Path(path)
    low = p.name.lower()
    if low.endswith(".parquet"):
        return pd.read_parquet(p)
    if low.endswith((".xlsx",".xls")):
        return pd.read_excel(p)
    if low.endswith(".json"):
        return pd.read_json(p)
    if low.endswith(".gz"):
        raw = gzip.open(p, "rb").read(200000)
    else:
        raw = p.read_bytes()[:200000]
    sample = raw.decode("utf-8", errors="replace")
    counts = {x: sample.count(x) for x in ["\t",",",";","|"]}
    sep = max(counts, key=counts.get)
    return pd.read_csv(p, sep=sep, low_memory=False, compression="infer")

def ensure_canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    defaults = {
        "source_database":"", "source_record_id":"", "source_version":"",
        "source_url":"", "HugoSymbol":"", "ProteinChange":"",
        "GenomeAssembly":"", "directionality_label_normalized":"unknown",
        "label_confidence":"low", "label_source":"", "label_evidence":"",
        "is_explicit_variant_level":False, "publication_ids":"",
        "disease_context":"", "raw_directionality":"", "raw_effect":"",
    }
    for c, v in defaults.items():
        if c not in out:
            out[c] = v
    order = CANONICAL_COLUMNS + [c for c in out.columns if c not in CANONICAL_COLUMNS]
    return out[order]

def write_parquet_safe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = ensure_canonical_columns(df)
    for c in out.select_dtypes(include=["object"]).columns:
        out[c] = out[c].astype("string")
    out.to_parquet(path, index=False)
    log(f"Wrote {len(out):,} rows -> {path}")

GOF_PATTERNS = [
    r"\bgain[ -]?of[ -]?function\b", r"\bactivating\b", r"\bactivation\b",
    r"\bconstitutively active\b", r"\bhyperactiv\w*\b",
    r"\bincreased (?:protein )?(?:activity|function)\b",
    r"\benhanced (?:protein )?(?:activity|function)\b",
    r"\bincrease[sd]? (?:catalytic|enzymatic) activity\b",
]
LOF_PATTERNS = [
    r"\bloss[ -]?of[ -]?function\b", r"\binactivating\b", r"\binactivation\b",
    r"\bno (?:detectable )?(?:activity|function)\b",
    r"\babolish(?:es|ed)? (?:the )?(?:activity|function)\b",
    r"\binactive\b", r"\bhypomorph\w*\b",
    r"\breduced (?:protein )?(?:activity|function)\b",
    r"\bdecreased (?:protein )?(?:activity|function)\b",
    r"\bimpaired (?:catalytic |enzymatic )?(?:activity|function)\b",
]
NEGATING = [
    r"\bno effect on (?:activity|function)\b",
    r"\bdoes not (?:affect|alter|reduce|increase) (?:activity|function)\b",
]

def explicit_directionality(text: Any) -> tuple[str,str]:
    t = safe_str(text)
    if any(re.search(p,t,re.I) for p in NEGATING):
        return "unknown",""
    g = [m.group(0) for p in GOF_PATTERNS for m in [re.search(p,t,re.I)] if m]
    l = [m.group(0) for p in LOF_PATTERNS for m in [re.search(p,t,re.I)] if m]
    if g and l:
        return "ambiguous", f"{g[0]} / {l[0]}"
    if g:
        return "gof_like", g[0]
    if l:
        return "lof", l[0]
    return "unknown",""

def add_conflict_flag(df: pd.DataFrame, colname: str="directionality_conflict_within_source") -> pd.DataFrame:
    if df.empty:
        df[colname] = pd.Series(dtype=bool)
        return df
    counts = df.groupby(["HugoSymbol","ProteinChange"])["directionality_label_normalized"].nunique()
    conflict = set(counts[counts > 1].index)
    df[colname] = [(g,p) in conflict for g,p in zip(df.HugoSymbol,df.ProteinChange)]
    return df

def write_summary(df: pd.DataFrame, path: Optional[Path], extra: Optional[dict]=None) -> dict:
    summary = {
        "output_rows": int(len(df)),
        "unique_variants": int(df[["HugoSymbol","ProteinChange"]].drop_duplicates().shape[0]) if len(df) else 0,
        "unique_genes": int(df["HugoSymbol"].nunique()) if len(df) else 0,
        "labels": df["directionality_label_normalized"].value_counts(dropna=False).to_dict() if len(df) else {},
        "confidence": df["label_confidence"].value_counts(dropna=False).to_dict() if len(df) else {},
    }
    if extra:
        summary.update(extra)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return summary
