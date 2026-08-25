#!/usr/bin/env python3
'''Collect explicit TP53 Loss_of_Function / Gain_of_Function functional assay annotations.'''
from __future__ import annotations
import argparse, re
from pathlib import Path
from urllib.parse import urljoin
import pandas as pd
from directionality_public_common import *

HOME="https://tp53.cancer.gov/get_tp53data"
HELP="https://tp53.cancer.gov/help"
TARGET="FunctionDownload_r21.csv"

def discover_url():
    s=session_with_retry()
    for page in [HOME,HELP]:
        r=s.get(page,timeout=120); r.raise_for_status(); html=r.text
        patterns = [
            r'(?:href|src)=["\']([^"\']*FunctionDownload[^"\']*\.csv[^"\']*)["\']',
            r'["\']([^"\']*/FunctionDownload_r\d+\.csv[^"\']*)["\']',
        ]
        for pat in patterns:
            m=re.search(pat,html,re.I)
            if m:
                return urljoin(r.url,m.group(1))
    return ""

def nonempty(v):
    s=safe_str(v)
    return bool(s and s.lower() not in {"na","n/a","none",".","0","false","no"})

def main():
    ap=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--workdir",type=Path,default=Path("tp53_cache"))
    ap.add_argument("--input-file",type=Path)
    ap.add_argument("--download-url",default="")
    ap.add_argument("--force-download",action="store_true")
    ap.add_argument("--summary-json",type=Path)
    args=ap.parse_args()
    args.workdir.mkdir(parents=True,exist_ok=True)
    if args.input_file:
        path=args.input_file; src=str(path)
    else:
        url=args.download_url or discover_url()
        if not url:
            raise RuntimeError("Could not discover FunctionDownload_r21.csv automatically. Download it from https://tp53.cancer.gov/get_tp53data and pass --input-file.")
        path=download(url,args.workdir/TARGET,args.force_download); src=url
    raw=read_table(path)
    lof=find_col(raw,["Loss_of_Function","Loss of Function"],True)
    gof=find_col(raw,["Gain_of_Function","Gain of Function"],True)
    aa=find_col(raw,["AAchange","ProtDescription","protein","mutation"],True)
    pm=find_col(raw,["PubMed","PMID","reference"])
    dne=find_col(raw,["Dominant_Negative_Activity","DNE"])
    rows=[]
    for i,x in raw.iterrows():
        prot=normalize_protein_change(x[aa])
        if not prot:
            continue
        effects=[]
        if nonempty(x[lof]):
            effects.append(("lof",safe_str(x[lof]),"Loss_of_Function"))
        if nonempty(x[gof]):
            effects.append(("gof_like",safe_str(x[gof]),"Gain_of_Function"))
        for label,ev,field in effects:
            rows.append({
                "source_database":"NCI_TP53","source_record_id":f"TP53:{prot}:{i}:{field}",
                "source_version":path.name,"source_url":src,
                "HugoSymbol":"TP53","ProteinChange":prot,
                "directionality_label_normalized":label,"label_confidence":"high",
                "label_source":f"NCI TP53 {field} functional-assay field",
                "label_evidence":ev,"is_explicit_variant_level":True,
                "publication_ids":extract_pmids(x[pm]) if pm else "",
                "raw_directionality":field,"raw_effect":ev,
                "tp53_dominant_negative_activity":safe_str(x[dne]) if dne else "",
            })
    df=add_conflict_flag(pd.DataFrame(rows) if rows else ensure_canonical_columns(pd.DataFrame()))
    write_parquet_safe(df,args.out)
    write_summary(df,args.summary_json,{"source":"NCI TP53","input_rows":len(raw),"input_file":str(path)})

if __name__=="__main__":
    main()
