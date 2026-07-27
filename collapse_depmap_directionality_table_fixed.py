#!/usr/bin/env python3
"""
Collapse a full DepMap mutation directionality parquet into a unique variant-level
training table for LOF/GOF classifier development.

Input
-----
A parquet file produced by depmap_directionality_table.py, with one row per
DepMap mutation record / model.

Output
------
A parquet file with one row per collapsed variant key, with consensus labels,
evidence counts, model counts, and optional training split columns.

Examples
--------
python collapse_depmap_directionality_table.py \
  --input depmap_mutation_directionality.parquet \
  --out depmap_variant_directionality_collapsed.parquet

# Protein-level labels, useful if genomic coordinates vary across transcripts
python collapse_depmap_directionality_table.py \
  --input depmap_mutation_directionality.parquet \
  --out depmap_variant_directionality_collapsed.protein.parquet \
  --collapse-level protein

# Create a stricter training table containing only LOF and GOF-like labels
python collapse_depmap_directionality_table.py \
  --input depmap_mutation_directionality.parquet \
  --out depmap_variant_directionality_trainable.parquet \
  --drop-ambiguous \
  --drop-unknown \
  --min-confidence medium \
  --add-train-val-test-split
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


LABEL_ORDER = ["lof", "gof_like", "ambiguous", "unknown"]
LABEL_NORMALIZATION = {"gof_like_drug_supported": "gof_like", "gof": "gof_like"}
DRUG_CONFIDENCE_RANK = {"": 0, "none": 0, "not_evaluated": 0, "not_supported_or_not_evaluable": 0, "high_target_matched": 3}
CONFIDENCE_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}
RANK_CONFIDENCE = {v: k for k, v in CONFIDENCE_RANK.items()}

BOOL_TRUE = {"true", "t", "1", "yes", "y"}


CORE_PREFERRED_COLUMNS = [
    "source_release",
    "ModelID",
    "ModelID_standard",
    "ModelConditionID",
    "DepMap_ID",
    "HugoSymbol",
    "EntrezGeneID",
    "EnsemblGeneID",
    "EnsemblFeatureID",
    "Chrom",
    "Pos",
    "Ref",
    "Alt",
    "DNAChange",
    "ProteinChange",
    "VariantType",
    "VariantInfo",
    "VepImpact",
    "AF",
    "DP",
    "GT",
    "LikelyLoF",
    "Hotspot",
    "HessDriver",
    "OncogeneHighImpact",
    "TumorSuppressorHighImpact",
    "CivicID",
    "CivicDescription",
    "CivicScore",
    "Sift",
    "Polyphen",
    "ProveanPrediction",
    "AMClass",
    "AMPathogenicity",
    "gene_role",
    "directionality_label_conservative",
    "directionality_label_heuristic",
    "directionality_label_with_drug_support",
    "directionality_confidence",
    "directionality_evidence",
    "drug_response_supports_gof",
    "drug_response_gof_confidence",
    "drug_response_evidence",
    "best_gof_drug",
    "best_gof_drug_target",
    "best_gof_drug_effect_size",
    "best_gof_drug_pvalue",
    "drug_name",
    "drug_target",
    "drug_target_pathway",
    "mutant_vs_wt_effect_size_more_sensitive_positive",
    "mutant_vs_wt_pvalue",
    "mutant_vs_wt_fdr",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def normalize_colname(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(c).strip().lower())


def find_col(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    norm_to_real = {normalize_colname(c): c for c in columns}
    for cand in candidates:
        hit = norm_to_real.get(normalize_colname(cand))
        if hit is not None:
            return hit
    for real in columns:
        nreal = normalize_colname(real)
        for cand in candidates:
            ncand = normalize_colname(cand)
            if ncand and ncand in nreal:
                return real
    return None


def available_columns(path: Path) -> List[str]:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pq
            return list(pq.ParquetFile(path).schema.names)
        except ImportError:
            return list(pd.read_parquet(path).columns)
    if suffix == ".tsv":
        return list(pd.read_csv(path, sep="\t", nrows=0).columns)
    return list(pd.read_csv(path, nrows=0).columns)

def read_input(path: Path, requested_cols: Optional[List[str]] = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        if requested_cols:
            cols = available_columns(path)
            real_cols = [c for c in requested_cols if c in cols]
            log(f"Reading {len(real_cols)} selected columns from {path}")
            return pd.read_parquet(path, columns=real_cols)
        log(f"Reading all columns from {path}")
        return pd.read_parquet(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", low_memory=False, usecols=requested_cols)
    return pd.read_csv(path, low_memory=False, usecols=requested_cols)


def as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(float).ne(0)
    return s.fillna("").astype(str).str.strip().str.lower().isin(BOOL_TRUE)


def first_nonempty(values: Iterable[object]) -> str:
    for v in values:
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s and s.lower() not in {"nan", "none", "null", "na"}:
            return s
    return ""


def join_unique(values: Iterable[object], max_items: int = 12) -> str:
    seen: List[str] = []
    for v in values:
        if pd.isna(v):
            continue
        s = str(v).strip()
        if not s or s.lower() in {"nan", "none", "null", "na"}:
            continue
        if s not in seen:
            seen.append(s)
        if len(seen) >= max_items:
            break
    return "; ".join(seen)


def mode_label(values: Iterable[object], allowed: Sequence[str] = LABEL_ORDER) -> str:
    vals = [str(v).strip() for v in values if not pd.isna(v)]
    vals = [v for v in vals if v in allowed]
    if not vals:
        return "unknown"
    counts = pd.Series(vals).value_counts()
    max_count = counts.max()
    tied = set(counts[counts == max_count].index)
    for lab in allowed:
        if lab in tied:
            return lab
    return "unknown"


def max_confidence(values: Iterable[object]) -> str:
    best = 0
    for v in values:
        s = str(v).strip().lower() if not pd.isna(v) else "none"
        best = max(best, CONFIDENCE_RANK.get(s, 0))
    return RANK_CONFIDENCE.get(best, "none")


def max_drug_confidence(values: Iterable[object]) -> str:
    candidates = [str(v).strip() for v in values if not pd.isna(v) and str(v).strip()]
    if not candidates:
        return "none"
    return max(candidates, key=lambda value: DRUG_CONFIDENCE_RANK.get(value.lower(), 1))


def confidence_meets(value: str, minimum: str) -> bool:
    return CONFIDENCE_RANK.get(str(value).lower(), 0) >= CONFIDENCE_RANK.get(str(minimum).lower(), 0)


def stable_float(x: object) -> float:
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def stable_split_key(row: pd.Series, seed: str, split_level: str = "gene") -> float:
    if split_level == "gene":
        identity = str(row.get("HugoSymbol", "")).strip().upper()
    elif split_level == "variant":
        identity = "|".join(
            str(row.get(col, ""))
            for col in ["HugoSymbol", "ProteinChange", "Chrom", "Pos", "Ref", "Alt", "DNAChange"]
        )
    else:
        raise ValueError(f"Unsupported split_level: {split_level}")
    digest = hashlib.sha1((seed + "|" + identity).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12 - 1)


def assign_split(
    df: pd.DataFrame,
    seed: str,
    val_frac: float,
    test_frac: float,
    split_level: str = "gene",
) -> pd.Series:
    if val_frac < 0 or test_frac < 0 or val_frac + test_frac >= 1:
        raise ValueError("Require 0 <= val_frac, 0 <= test_frac, and val_frac + test_frac < 1")
    scores = df.apply(lambda row: stable_split_key(row, seed, split_level), axis=1)
    test_cut = test_frac
    val_cut = test_frac + val_frac
    return pd.Series(
        np.where(scores < test_cut, "test", np.where(scores < val_cut, "val", "train")),
        index=df.index,
        dtype="string",
    )

def infer_required_columns(all_cols: List[str]) -> List[str]:
    wanted = []
    for c in CORE_PREFERRED_COLUMNS:
        hit = find_col(all_cols, [c])
        if hit is not None and hit not in wanted:
            wanted.append(hit)

    # Ensure common alternative model IDs and directionality fields are included.
    for group in [
        ["ModelID", "ModelConditionID", "DepMap_ID", "depmap_id", "CCLEName", "stripped_cell_line_name"],
        ["HugoSymbol", "gene", "symbol"],
        ["ProteinChange", "protein_change", "HGVSp"],
        ["DNAChange", "dna_change", "HGVSc"],
        ["directionality_label_with_drug_support", "directionality_label_heuristic"],
    ]:
        hit = find_col(all_cols, group)
        if hit is not None and hit not in wanted:
            wanted.append(hit)
    return wanted


def make_canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    aliases = {
        "ModelID": ["ModelID_standard", "ModelID", "ModelConditionID", "DepMap_ID", "depmap_id", "model_id"],
        "HugoSymbol": ["HugoSymbol", "Hugo_Symbol", "gene", "symbol"],
        "ProteinChange": ["ProteinChange", "protein_change", "HGVSp", "hgvsp"],
        "DNAChange": ["DNAChange", "dna_change", "HGVSc", "hgvsc"],
        "Chrom": ["Chrom", "Chromosome", "chr"],
        "Pos": ["Pos", "Start_position", "Start_Position", "position"],
        "Ref": ["Ref", "Reference_Allele", "ref"],
        "Alt": ["Alt", "Tumor_Seq_Allele2", "alt"],
        "VariantInfo": ["VariantInfo", "Consequence", "variant_info"],
        "VepImpact": ["VepImpact", "IMPACT", "vep_impact"],
        "gene_role": ["gene_role", "GeneRole", "cancermine_role"],
        "directionality_label_conservative": ["directionality_label_conservative"],
        "directionality_label_heuristic": ["directionality_label_heuristic"],
        "directionality_label_with_drug_support": ["directionality_label_with_drug_support"],
        "directionality_confidence": ["directionality_confidence"],
        "directionality_evidence": ["directionality_evidence"],
        "drug_response_supports_gof": ["drug_response_supports_gof"],
        "drug_response_gof_confidence": ["drug_response_gof_confidence"],
        "drug_response_evidence": ["drug_response_evidence"],
    }
    for canon, cands in aliases.items():
        if canon not in df.columns:
            hit = find_col(df.columns, cands)
            if hit is not None:
                df[canon] = df[hit]
    for c in [
        "ModelID", "HugoSymbol", "ProteinChange", "DNAChange", "Chrom", "Pos", "Ref", "Alt",
        "VariantInfo", "VepImpact", "gene_role", "directionality_label_conservative",
        "directionality_label_heuristic", "directionality_label_with_drug_support",
        "directionality_confidence", "directionality_evidence", "drug_response_gof_confidence",
        "drug_response_evidence"
    ]:
        if c not in df.columns:
            df[c] = ""
    if "drug_response_supports_gof" not in df.columns:
        df["drug_response_supports_gof"] = False
    return df


def build_variant_key(df: pd.DataFrame, collapse_level: str, keep_unidentified: bool = False) -> pd.DataFrame:
    df = df.copy()
    for col in ["HugoSymbol", "ProteinChange", "DNAChange", "Chrom", "Pos", "Ref", "Alt"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    gene_ok = df["HugoSymbol"].ne("")
    protein_ok = gene_ok & df["ProteinChange"].ne("")
    dna_ok = gene_ok & df["DNAChange"].ne("")
    genomic_ok = gene_ok & df[["Chrom", "Pos", "Ref", "Alt"]].ne("").all(axis=1)

    if collapse_level == "genomic":
        valid = genomic_ok
        key = df[["HugoSymbol", "Chrom", "Pos", "Ref", "Alt"]].agg("|".join, axis=1)
    elif collapse_level == "protein":
        valid = protein_ok | dna_ok | genomic_ok
        fallback = np.where(
            dna_ok,
            "DNA:" + df["DNAChange"],
            "GENOMIC:" + df[["Chrom", "Pos", "Ref", "Alt"]].agg(":".join, axis=1),
        )
        identity = np.where(protein_ok, "PROTEIN:" + df["ProteinChange"], fallback)
        key = df["HugoSymbol"] + "|" + pd.Series(identity, index=df.index)
    elif collapse_level == "gene_dna_protein":
        valid = gene_ok & (df["DNAChange"].ne("") | df["ProteinChange"].ne(""))
        key = df[["HugoSymbol", "DNAChange", "ProteinChange"]].agg("|".join, axis=1)
    else:
        raise ValueError(f"Unknown collapse level: {collapse_level}")

    df["has_valid_variant_key"] = valid
    df["variant_key"] = np.where(valid, key, "")
    if keep_unidentified:
        missing = ~valid
        df.loc[missing, "variant_key"] = [f"UNIDENTIFIED|{idx}" for idx in df.index[missing]]
    else:
        dropped = int((~valid).sum())
        if dropped:
            log(f"Dropping {dropped:,} rows without a valid {collapse_level} variant identity")
        df = df.loc[valid].copy()
    df["variant_key_type"] = collapse_level
    return df

def normalize_directionality_labels(labels: pd.Series) -> pd.Series:
    normalized = labels.fillna("unknown").astype(str).str.strip().str.lower().replace("", "unknown")
    normalized = normalized.replace(LABEL_NORMALIZATION)
    return normalized.where(normalized.isin(LABEL_ORDER), "unknown")


def pick_final_label(labels: pd.Series, confidences: pd.Series) -> Tuple[str, str, str]:
    vals = normalize_directionality_labels(labels)
    conf = confidences.fillna("none").astype(str).str.strip().str.lower().replace("", "none")
    counts = vals.value_counts().to_dict()

    n_lof = int(counts.get("lof", 0))
    n_gof = int(counts.get("gof_like", 0))
    n_amb = int(counts.get("ambiguous", 0))
    n_unk = int(counts.get("unknown", 0))

    if n_lof > 0 and n_gof > 0:
        return "ambiguous", "low", "conflicting lof and gof_like evidence across model-level records"
    if n_lof > 0:
        return "lof", max_confidence(conf[vals == "lof"]), "lof evidence present; no conflicting gof_like evidence"
    if n_gof > 0:
        return "gof_like", max_confidence(conf[vals == "gof_like"]), "gof_like evidence present; no conflicting lof evidence"
    if n_amb > 0:
        return "ambiguous", "low", "only ambiguous directionality evidence found"
    return "unknown", "none", "no directionality evidence found"


def collapse(df: pd.DataFrame, collapse_level: str, label_column: str, keep_unidentified: bool = False) -> pd.DataFrame:
    df = make_canonical_columns(df)
    df = build_variant_key(df, collapse_level, keep_unidentified=keep_unidentified)

    if label_column not in df.columns:
        raise KeyError(f"Requested label column {label_column!r} is not present. Available columns include: {list(df.columns)[:80]}")

    if "LikelyLoF" in df.columns:
        df["_likely_lof_bool"] = as_bool(df["LikelyLoF"])
    else:
        df["_likely_lof_bool"] = False
    if "Hotspot" in df.columns:
        df["_hotspot_bool"] = as_bool(df["Hotspot"])
    else:
        df["_hotspot_bool"] = False
    if "HessDriver" in df.columns:
        df["_hess_driver_bool"] = as_bool(df["HessDriver"])
    else:
        df["_hess_driver_bool"] = False
    if "OncogeneHighImpact" in df.columns:
        df["_oncogene_hi_bool"] = as_bool(df["OncogeneHighImpact"])
    else:
        df["_oncogene_hi_bool"] = False
    if "TumorSuppressorHighImpact" in df.columns:
        df["_tsg_hi_bool"] = as_bool(df["TumorSuppressorHighImpact"])
    else:
        df["_tsg_hi_bool"] = False
    df["_drug_gof_bool"] = as_bool(df["drug_response_supports_gof"])
    df["_af_num"] = pd.to_numeric(df.get("AF", np.nan), errors="coerce") if "AF" in df.columns else np.nan
    df["_dp_num"] = pd.to_numeric(df.get("DP", np.nan), errors="coerce") if "DP" in df.columns else np.nan

    df["_normalized_directionality_label"] = normalize_directionality_labels(df[label_column])
    # Precompute label counts for transparent training filters.
    for lab in LABEL_ORDER:
        df[f"_is_{lab}"] = df["_normalized_directionality_label"].eq(lab)

    rows: List[Dict[str, object]] = []
    group_cols = ["variant_key", "variant_key_type"]
    log(f"Collapsing {len(df):,} mutation-model rows by {collapse_level!r}")

    for (variant_key, variant_key_type), g in df.groupby(group_cols, dropna=False, sort=False):
        final_label, final_conf, consensus_reason = pick_final_label(g["_normalized_directionality_label"], g["directionality_confidence"])
        n_records = len(g)
        model_ids = g["ModelID"].fillna("").astype(str)
        n_models = int(model_ids[model_ids.str.len() > 0].nunique())

        rec: Dict[str, object] = {
            "variant_key": variant_key,
            "variant_key_type": variant_key_type,
            "HugoSymbol": first_nonempty(g["HugoSymbol"]),
            "ProteinChange": first_nonempty(g["ProteinChange"]),
            "DNAChange": first_nonempty(g["DNAChange"]),
            "Chrom": first_nonempty(g["Chrom"]),
            "Pos": first_nonempty(g["Pos"]),
            "Ref": first_nonempty(g["Ref"]),
            "Alt": first_nonempty(g["Alt"]),
            "EntrezGeneID": first_nonempty(g["EntrezGeneID"]) if "EntrezGeneID" in g else "",
            "EnsemblGeneID": first_nonempty(g["EnsemblGeneID"]) if "EnsemblGeneID" in g else "",
            "VariantType": join_unique(g["VariantType"]) if "VariantType" in g else "",
            "VariantInfo": join_unique(g["VariantInfo"]),
            "VepImpact": join_unique(g["VepImpact"]),
            "gene_role": join_unique(g["gene_role"]),
            "directionality_label": final_label,
            "directionality_confidence": final_conf,
            "directionality_consensus_reason": consensus_reason,
            "directionality_evidence": join_unique(g["directionality_evidence"], max_items=20),
            "n_depmap_records": int(n_records),
            "n_models_with_variant": n_models,
            "n_label_lof_records": int(g["_is_lof"].sum()),
            "n_label_gof_like_records": int(g["_is_gof_like"].sum()),
            "n_label_ambiguous_records": int(g["_is_ambiguous"].sum()),
            "n_label_unknown_records": int(g["_is_unknown"].sum()),
            "n_likely_lof_records": int(g["_likely_lof_bool"].sum()),
            "n_hotspot_records": int(g["_hotspot_bool"].sum()),
            "n_hess_driver_records": int(g["_hess_driver_bool"].sum()),
            "n_oncogene_high_impact_records": int(g["_oncogene_hi_bool"].sum()),
            "n_tumor_suppressor_high_impact_records": int(g["_tsg_hi_bool"].sum()),
            "n_drug_response_gof_support_records": int(g["_drug_gof_bool"].sum()),
            "drug_response_supports_gof": bool(g["_drug_gof_bool"].any()),
            "drug_response_gof_confidence": max_drug_confidence(g["drug_response_gof_confidence"]),
            "drug_response_evidence": join_unique(g["drug_response_evidence"], max_items=12),
            "mean_AF": float(g["_af_num"].mean()) if g["_af_num"].notna().any() else np.nan,
            "median_AF": float(g["_af_num"].median()) if g["_af_num"].notna().any() else np.nan,
            "mean_DP": float(g["_dp_num"].mean()) if g["_dp_num"].notna().any() else np.nan,
            "median_DP": float(g["_dp_num"].median()) if g["_dp_num"].notna().any() else np.nan,
            "example_model_ids": join_unique(model_ids, max_items=8),
        }

        for c in [
            "source_release", "CivicID", "CivicDescription", "CivicScore", "Sift", "Polyphen",
            "ProveanPrediction", "AMClass", "AMPathogenicity", "best_gof_drug",
            "best_gof_drug_target", "best_gof_drug_effect_size", "best_gof_drug_pvalue",
            "drug_name", "drug_target", "drug_target_pathway",
            "mutant_vs_wt_effect_size_more_sensitive_positive", "mutant_vs_wt_pvalue", "mutant_vs_wt_fdr",
        ]:
            if c in g.columns:
                rec[c] = join_unique(g[c], max_items=12)

        rows.append(rec)

    if not rows:
        return pd.DataFrame(columns=[
            "variant_key", "variant_key_type", "HugoSymbol", "ProteinChange", "DNAChange",
            "Chrom", "Pos", "Ref", "Alt", "directionality_label",
            "directionality_confidence", "directionality_consensus_reason",
            "directionality_evidence", "n_depmap_records", "n_models_with_variant",
            "drug_response_supports_gof", "drug_response_gof_confidence",
            "drug_response_evidence", "is_trainable_lof_gof",
            "binary_label_lof0_gof1", "binary_label_name",
        ])

    out = pd.DataFrame(rows)
    out["is_trainable_lof_gof"] = out["directionality_label"].isin(["lof", "gof_like"])
    out["binary_label_lof0_gof1"] = np.where(
        out["directionality_label"].eq("gof_like"),
        1,
        np.where(out["directionality_label"].eq("lof"), 0, np.nan),
    )
    out["binary_label_name"] = np.where(
        out["directionality_label"].eq("gof_like"),
        "gof_like",
        np.where(out["directionality_label"].eq("lof"), "lof", "not_trainable"),
    )
    return out


def apply_filters(
    df: pd.DataFrame,
    drop_ambiguous: bool,
    drop_unknown: bool,
    min_confidence: str,
    min_models: int,
    require_drug_for_gof: bool,
) -> pd.DataFrame:
    before = len(df)
    out = df.copy()
    if drop_ambiguous:
        out = out[~out["directionality_label"].eq("ambiguous")]
    if drop_unknown:
        out = out[~out["directionality_label"].eq("unknown")]
    if min_confidence != "none":
        out = out[out["directionality_confidence"].map(lambda x: confidence_meets(str(x), min_confidence))]
    if min_models > 1:
        out = out[out["n_models_with_variant"] >= min_models]
    if require_drug_for_gof:
        out = out[(~out["directionality_label"].eq("gof_like")) | (out["drug_response_supports_gof"].astype(bool))]
    log(f"Filters retained {len(out):,} / {before:,} collapsed variants")
    return out.reset_index(drop=True)


def label_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["directionality_label", "directionality_confidence"]
    return (
        df.groupby(cols, dropna=False)
        .size()
        .reset_index(name="n_variants")
        .sort_values(["directionality_label", "directionality_confidence"])
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collapse full DepMap mutation directionality table to variant-level parquet.")
    p.add_argument("--input", required=True, type=Path, help="Full mutation-level parquet produced by depmap_directionality_table.py")
    p.add_argument("--out", required=True, type=Path, help="Output collapsed parquet path")
    p.add_argument(
        "--collapse-level",
        choices=["protein", "genomic", "gene_dna_protein"],
        default="protein",
        help="Variant identity used for collapsing. protein is usually best for LOF/GOF classifier labels.",
    )
    p.add_argument(
        "--label-column",
        default="directionality_label_with_drug_support",
        help="Model-level label column to collapse. Falls back manually by specifying directionality_label_heuristic if needed.",
    )
    p.add_argument("--drop-ambiguous", action="store_true", help="Drop ambiguous collapsed variants")
    p.add_argument("--drop-unknown", action="store_true", help="Drop unknown collapsed variants")
    p.add_argument(
        "--min-confidence",
        choices=["none", "low", "medium", "high"],
        default="none",
        help="Minimum final confidence to retain",
    )
    p.add_argument("--min-models", type=int, default=1, help="Minimum number of distinct models containing the variant")
    p.add_argument(
        "--require-drug-for-gof",
        action="store_true",
        help="Keep GOF-like rows only if there is drug-response support. Very strict; may remove many known hotspot GOF-like mutations.",
    )
    p.add_argument("--add-train-val-test-split", action="store_true", help="Add deterministic split column")
    p.add_argument("--split-seed", default="depmap_directionality_v1", help="String seed for deterministic hash split")
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--summary-json", type=Path, default=None, help="Optional JSON summary output")
    p.add_argument("--summary-csv", type=Path, default=None, help="Optional CSV label summary output")
    p.add_argument("--split-level", choices=["gene", "variant"], default="gene", help="Grouping unit for deterministic split; gene prevents gene leakage.")
    p.add_argument("--keep-unidentified", action="store_true", help="Retain unidentified rows as unique non-merged records instead of dropping them.")
    p.add_argument("--read-all-columns", action="store_true", help="Read all input columns instead of selected useful columns")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cols = None if args.read_all_columns else infer_required_columns(available_columns(args.input))
    df = read_input(args.input, requested_cols=cols)

    # If preferred label column is missing, fall back to heuristic if available.
    if args.label_column not in df.columns:
        fallback = "directionality_label_heuristic"
        if fallback in df.columns:
            log(f"Label column {args.label_column!r} not found; using {fallback!r}")
            args.label_column = fallback
        else:
            raise KeyError(f"Neither {args.label_column!r} nor {fallback!r} exists in input.")

    collapsed = collapse(df, args.collapse_level, args.label_column, keep_unidentified=args.keep_unidentified)
    collapsed = apply_filters(
        collapsed,
        drop_ambiguous=args.drop_ambiguous,
        drop_unknown=args.drop_unknown,
        min_confidence=args.min_confidence,
        min_models=args.min_models,
        require_drug_for_gof=args.require_drug_for_gof,
    )

    if args.add_train_val_test_split:
        collapsed["split"] = assign_split(collapsed, args.split_seed, args.val_frac, args.test_frac, split_level=args.split_level)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    log(f"Writing collapsed parquet: {args.out}")
    collapsed.to_parquet(args.out, index=False)

    summ = label_summary(collapsed)
    print("\nLabel summary:")
    print(summ.to_string(index=False))

    if args.summary_csv:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        summ.to_csv(args.summary_csv, index=False)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "input": str(args.input),
            "output": str(args.out),
            "collapse_level": args.collapse_level,
            "label_column": args.label_column,
            "split_level": args.split_level if args.add_train_val_test_split else None,
            "n_variants": int(len(collapsed)),
            "n_trainable_lof_gof": int(collapsed["is_trainable_lof_gof"].sum()),
            "label_summary": summ.to_dict(orient="records"),
        }
        args.summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
