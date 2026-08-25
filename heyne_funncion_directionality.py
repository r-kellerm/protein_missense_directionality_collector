#!/usr/bin/env python3
'''Collect the published Heyne et al. funNCion LOF/GOF source variants (SCN/CACNA1 genes).'''
from __future__ import annotations
import argparse, re
from pathlib import Path
import pandas as pd
from directionality_public_common import *

RAW="https://raw.githubusercontent.com/heyhen/funNCion/master/SupplementaryTable_S1_pathvariantsusedintraining_revision2.txt"
REPO="https://github.com/heyhen/funNCion"

def read_auto(path):
    sample=path.read_text(encoding="utf-8",errors="replace")[:100000]
    counts={x:sample.count(x) for x in ["\t",",",";"]}
    sep=max(counts,key=counts.get)
    return pd.read_csv(path,sep=sep,low_memory=False)

def row_label(row):
    # Prefer a dedicated label-like column, but accept exact GOF/LOF cells anywhere.
    for k,v in row.items():
        nk=normalize_colname(k); u=safe_str(v).strip().upper()
        if any(z in nk for z in ["goflof","functional","effect","class","label","mechanism"]):
            if u in {"GOF","GAIN OF FUNCTION","GAIN-OF-FUNCTION"}:
                return "gof_like",safe_str(v)
            if u in {"LOF","LOSS OF FUNCTION","LOSS-OF-FUNCTION"}:
                return "lof",safe_str(v)
    for v in row.values:
        u=safe_str(v).strip().upper()
        if u in {"GOF","GAIN OF FUNCTION","GAIN-OF-FUNCTION"}:
            return "gof_like",safe_str(v)
        if u in {"LOF","LOSS OF FUNCTION","LOSS-OF-FUNCTION"}:
            return "lof",safe_str(v)
    return "unknown",""

def guess_gene_protein(row):
    gene=""; prot=""
    for k,v in row.items():
        nk=normalize_colname(k); s=safe_str(v)
        if not gene and any(z in nk for z in ["gene","symbol"]):
            if re.fullmatch(r"(?:SCN\d+[A-Z]?|CACNA1[A-Z])",s,re.I):
                gene=s.upper()
        if not prot and any(z in nk for z in ["protein","aachange","variant","mutation","hgvsp"]):
            prot=normalize_protein_change(s)
    txt=" | ".join(safe_str(v) for v in row.values)
    if not gene:
        m=re.search(r"\b(SCN\d+[A-Z]?|CACNA1[A-Z])\b",txt,re.I)
        if m:
            gene=m.group(1).upper()
    if not prot:
        prot=normalize_protein_change(txt)
    return gene,prot

def main():
    ap=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--workdir",type=Path,default=Path("funncion_cache"))
    ap.add_argument("--input-file",type=Path)
    ap.add_argument("--force-download",action="store_true")
    ap.add_argument("--summary-json",type=Path)
    args=ap.parse_args()
    args.workdir.mkdir(parents=True,exist_ok=True)
    path=args.input_file or download(RAW,args.workdir/"SupplementaryTable_S1_pathvariantsusedintraining_revision2.txt",args.force_download)
    raw=read_auto(path)
    rows=[]
    for i,x in raw.iterrows():
        label,rawlab=row_label(x)
        if label=="unknown":
            continue
        gene,prot=guess_gene_protein(x)
        if not gene or not prot:
            continue
        txt=" | ".join(f"{k}={safe_str(v)}" for k,v in x.items() if safe_str(v))
        rows.append({
            "source_database":"Heyne_funNCion","source_record_id":f"{gene}:{prot}:{i}",
            "source_version":"Heyne et al.","source_url":REPO,
            "HugoSymbol":gene,"ProteinChange":prot,
            "directionality_label_normalized":label,"label_confidence":"medium",
            "label_source":"Heyne et al. curated pathogenic variant mechanism",
            "label_evidence":txt[:4000],"is_explicit_variant_level":True,
            "publication_ids":"","raw_directionality":rawlab,"raw_effect":rawlab,
            "evidence_kind":"clinical_mechanism_curated_not_uniform_direct_assay",
        })
    if not rows:
        raise RuntimeError(f"Downloaded table but could not identify GOF/LOF missense rows. Columns={list(raw.columns)}")
    df=add_conflict_flag(pd.DataFrame(rows))
    write_parquet_safe(df,args.out)
    write_summary(df,args.summary_json,{"source":"Heyne/funNCion","input_rows":len(raw),"input_file":str(path)})

if __name__=="__main__":
    main()
