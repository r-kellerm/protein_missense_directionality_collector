#!/usr/bin/env python3
'''Collect experimentally activating/inactivating missense variants from Ng et al. FASMIC supplementary tables.'''
from __future__ import annotations
import argparse, re
from pathlib import Path
import pandas as pd
from directionality_public_common import *

SUPP="https://pmc.ncbi.nlm.nih.gov/articles/PMC5926201/bin/NIHMS940011-supplement-2.xlsx"
ARTICLE="https://pmc.ncbi.nlm.nih.gov/articles/PMC5926201/"
PMID="29533785"

def load_sheets(path):
    xls=pd.ExcelFile(path)
    return [(s,pd.read_excel(path,sheet_name=s)) for s in xls.sheet_names]

def call_from_row(row):
    vals=[]
    for k,v in row.items():
        s=safe_str(v).strip()
        if re.search(r"\b(?:activating|inactivating)\b",s,re.I):
            vals.append((safe_str(k),s))
    go=any(re.search(r"\bactivating\b",v,re.I) and not re.search(r"\binactivating\b",v,re.I) for _,v in vals)
    lo=any(re.search(r"\binactivating\b",v,re.I) for _,v in vals)
    if go and lo:
        return "ambiguous",vals
    if go:
        return "gof_like",vals
    if lo:
        return "lof",vals
    return "unknown",vals

def guess_gene_protein(row):
    gene=""; prot=""
    for k,v in row.items():
        nk=normalize_colname(k); s=safe_str(v)
        if not gene and any(z in nk for z in ["gene","genesymbol","hugo"]):
            if re.fullmatch(r"[A-Za-z0-9._-]{2,30}",s):
                gene=s.upper()
        if not prot and any(z in nk for z in ["protein","aachange","mutation","alteration","variant"]):
            prot=normalize_protein_change(s)
    text=" | ".join(safe_str(v) for v in row.values)
    if not prot:
        prot=normalize_protein_change(text)
    if not gene:
        m=re.search(r"\b([A-Z][A-Z0-9._-]{1,20})\s+(?:p\.)?[A-Z]\d+[A-Z]\b",text)
        if m:
            gene=m.group(1)
    return gene,prot

def main():
    ap=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--workdir",type=Path,default=Path("fasmic_cache"))
    ap.add_argument("--input-file",type=Path)
    ap.add_argument("--force-download",action="store_true")
    ap.add_argument("--include-ambiguous",action="store_true")
    ap.add_argument("--summary-json",type=Path)
    args=ap.parse_args()
    args.workdir.mkdir(parents=True,exist_ok=True)
    path=args.input_file or download(SUPP,args.workdir/"NIHMS940011-supplement-2.xlsx",args.force_download)
    if path.read_bytes()[:2] != b"PK":
        raise RuntimeError(f"{path} is not an XLSX/ZIP file; PMC supplement URL may have changed.")
    rows=[]; sheets=load_sheets(path); diagnostics={}
    for sheet,raw in sheets:
        diagnostics[sheet]=list(map(str,raw.columns))
        for i,x in raw.iterrows():
            label,calls=call_from_row(x)
            if label=="unknown" or (label=="ambiguous" and not args.include_ambiguous):
                continue
            gene,prot=guess_gene_protein(x)
            if not gene or not prot:
                continue
            ev="; ".join(f"{k}={v}" for k,v in calls)
            rows.append({
                "source_database":"FASMIC_Ng2018","source_record_id":f"{sheet}:{i}",
                "source_version":"Ng et al. 2018","source_url":ARTICLE,
                "HugoSymbol":gene,"ProteinChange":prot,
                "directionality_label_normalized":label,"label_confidence":"high",
                "label_source":"experimental Ba/F3/MCF10A functional classification",
                "label_evidence":ev,"is_explicit_variant_level":True,
                "publication_ids":PMID,"raw_directionality":ev,"raw_effect":ev,
                "fasmic_sheet":sheet,
            })
    if not rows:
        raise RuntimeError("No activating/inactivating missense rows parsed. Sheet columns: "+repr(diagnostics))
    df=add_conflict_flag(pd.DataFrame(rows))
    write_parquet_safe(df,args.out)
    write_summary(df,args.summary_json,{"source":"FASMIC/Ng2018","input_file":str(path),"sheets":[s for s,_ in sheets]})

if __name__=="__main__":
    main()
