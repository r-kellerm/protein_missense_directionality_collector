#!/usr/bin/env python3
'''Download the public GLOF Hugging Face dataset and normalize LOF/GOF missense labels.'''
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from directionality_public_common import *

API="https://huggingface.co/api/datasets/victormaricato/glof/parquet"
PAGE="https://huggingface.co/datasets/victormaricato/glof"

def parquet_urls(x):
    out=[]
    if isinstance(x,str) and x.startswith("http") and ".parquet" in x:
        out.append(x)
    elif isinstance(x,dict):
        for v in x.values():
            out.extend(parquet_urls(v))
    elif isinstance(x,list):
        for v in x:
            out.extend(parquet_urls(v))
    return list(dict.fromkeys(out))

def main():
    ap=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--workdir",type=Path,default=Path("glof_cache"))
    ap.add_argument("--force-download",action="store_true")
    ap.add_argument("--include-neutral",action="store_true")
    ap.add_argument("--summary-json",type=Path)
    args=ap.parse_args()
    args.workdir.mkdir(parents=True,exist_ok=True)
    r=session_with_retry().get(API,timeout=120); r.raise_for_status()
    urls=parquet_urls(r.json())
    if not urls:
        raise RuntimeError(f"No parquet URLs returned by {API}: {str(r.json())[:1000]}")
    dfs=[]
    for i,u in enumerate(urls):
        p=download(u,args.workdir/f"glof_{i}.parquet",args.force_download)
        dfs.append(pd.read_parquet(p))
    raw=pd.concat(dfs,ignore_index=True)
    label_col=find_col(raw,["LABEL"],True)
    gene_col=find_col(raw,["GENE_SYMBOL"],True)
    pos_col=find_col(raw,["AA_POSITION"],True)
    ref_col=find_col(raw,["PROTEIN_REF"],True)
    alt_col=find_col(raw,["PROTEIN_ALT"],True)
    key_col=find_col(raw,["VARIANTKEY"])
    rows=[]
    for _,x in raw.iterrows():
        lab=safe_str(x[label_col]).upper()
        if lab=="NEUTRAL" and not args.include_neutral:
            continue
        if lab not in {"LOF","GOF","NEUTRAL"}:
            continue
        label={"LOF":"lof","GOF":"gof_like","NEUTRAL":"neutral"}[lab]
        gene=safe_str(x[gene_col]).upper()
        try:
            pos=str(int(float(x[pos_col])))
        except Exception:
            pos=safe_str(x[pos_col])
        ref=safe_str(x[ref_col]).upper()
        alt=safe_str(x[alt_col]).upper()
        if not gene or not pos or ref not in AA1 or alt not in AA1:
            continue
        prot=f"p.{ref}{pos}{alt}"
        rows.append({
            "source_database":"GLOF","source_record_id":safe_str(x[key_col]) if key_col else f"{gene}:{prot}",
            "source_version":"2026-06-05","source_url":PAGE,
            "HugoSymbol":gene,"ProteinChange":prot,"GenomeAssembly":"GRCh38",
            "directionality_label_normalized":label,"label_confidence":"medium",
            "label_source":"GLOF expert-curated functional mechanism",
            "label_evidence":"Expert annotation integrating ClinVar, published functional studies, phenotype correlations, and gene-disease mechanism",
            "is_explicit_variant_level":True,
            "raw_directionality":lab,"raw_effect":lab,
            "glof_variantkey":safe_str(x[key_col]) if key_col else "",
            "glof_provenance_note":"Not independent of ClinVar; preserve source lineage.",
        })
    df=add_conflict_flag(pd.DataFrame(rows) if rows else ensure_canonical_columns(pd.DataFrame()))
    write_parquet_safe(df,args.out)
    write_summary(df,args.summary_json,{"source":"GLOF","input_rows":len(raw),"parquet_parts":len(urls)})

if __name__=="__main__":
    main()
