#!/usr/bin/env python3
'''Collect explicit gain/loss missense annotations from the public TSHR mutation database.'''
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from directionality_public_common import *

URL="https://tsh-receptor-mutation-database.org/list.html"

def flatten_cols(df):
    if isinstance(df.columns,pd.MultiIndex):
        df.columns=[" ".join(safe_str(x) for x in tup if safe_str(x) and not safe_str(x).startswith("Unnamed")).strip() for tup in df.columns]
    else:
        df.columns=[safe_str(x) for x in df.columns]
    return df

def main():
    ap=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--url",default=URL)
    ap.add_argument("--summary-json",type=Path)
    args=ap.parse_args()
    tables=[flatten_cols(x) for x in pd.read_html(args.url)]
    table=None
    for t in tables:
        typ=find_col(t,["type","effect","gain loss","functional type"])
        mut=find_col(t,["mutation","protein","amino acid","aa change","variant"])
        if typ and mut:
            vals=" ".join(safe_series(t[typ]).str.lower().head(100).tolist())
            if "gain" in vals or "loss" in vals:
                table=t
                break
    if table is None:
        raise RuntimeError("Could not identify the TSHR gain/loss mutation table. Site layout may have changed.")
    typ=find_col(table,["type","effect","gain loss","functional type"],True)
    mut=find_col(table,["mutation","protein","amino acid","aa change","variant"],True)
    ref=find_col(table,["reference","pubmed","pmid"])
    disease=find_col(table,["phenotype","disease","clinical"])
    rows=[]
    for i,x in table.iterrows():
        raw=safe_str(x[typ]).lower()
        label="gof_like" if "gain" in raw else "lof" if "loss" in raw else "unknown"
        if label=="unknown":
            continue
        prot=normalize_protein_change(x[mut])
        if not prot:
            continue
        rt=safe_str(x[ref]) if ref else ""
        rows.append({
            "source_database":"TSHR_Mutation_Database","source_record_id":f"TSHR:{prot}:{i}",
            "source_url":args.url,"HugoSymbol":"TSHR","ProteinChange":prot,
            "directionality_label_normalized":label,"label_confidence":"high",
            "label_source":"TSHR database explicit Type=gain/loss",
            "label_evidence":f"type={safe_str(x[typ])}",
            "is_explicit_variant_level":True,"publication_ids":extract_pmids(rt),
            "disease_context":safe_str(x[disease]) if disease else "",
            "raw_directionality":safe_str(x[typ]),"raw_effect":safe_str(x[typ]),
        })
    df=add_conflict_flag(pd.DataFrame(rows) if rows else ensure_canonical_columns(pd.DataFrame()))
    write_parquet_safe(df,args.out)
    write_summary(df,args.summary_json,{"source":"TSHR Mutation Database","input_rows":len(table)})

if __name__=="__main__":
    main()
