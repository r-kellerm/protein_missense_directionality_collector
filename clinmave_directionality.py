#!/usr/bin/env python3
'''
Collect ClinMAVE missense LoF/GoF classifications.

ClinMAVE currently documents a public "Retrieve variants by Gene" CSV export rather
than a stable all-variant bulk URL. This collector therefore supports:
  - --input-file: downloaded ClinMAVE CSV/XLSX/Parquet (deterministic)
  - --download-url: a captured/current CSV export URL
  - best-effort automatic discovery if the public page exposes a direct CSV URL.

Only explicit Loss-of-function / Gain-of-function classifications and protein
missense substitutions are retained.
'''
from __future__ import annotations
import argparse, re
from pathlib import Path
from urllib.parse import urljoin
import pandas as pd
from directionality_public_common import *

PAGE="https://ngdc.cncb.ac.cn/clinmave/download"

def parse(df,src):
    gene=find_col(df,["gene","gene_name","gene symbol"],True)
    cls=find_col(df,["functional classification","functional_classification","consequence"],True)
    variant=find_col(df,["variant","identifier","hgvs","protein change","hgvsp"])
    protein=find_col(df,["protein","protein_change","hgvsp"])
    technique=find_col(df,["mave technique","technique","method"])
    dataset=find_col(df,["dataset id","dataset","dataset_id"])
    pub=find_col(df,["publication","pmid","pubmed"])
    phenotype=find_col(df,["phenotype"])
    rows=[]
    for i,x in df.iterrows():
        raw=safe_str(x[cls]).lower()
        if "loss-of-function" in raw or "loss of function" in raw:
            label="lof"
        elif "gain-of-function" in raw or "gain of function" in raw:
            label="gof_like"
        else:
            continue
        p=normalize_protein_change(x[protein]) if protein else ""
        if not p and variant:
            p=normalize_protein_change(x[variant])
        if not p:
            continue
        g=safe_str(x[gene]).upper()
        if not g:
            continue
        tech=safe_str(x[technique]) if technique else ""
        ds=safe_str(x[dataset]) if dataset else str(i)
        phen=safe_str(x[phenotype]) if phenotype else ""
        rows.append({
            "source_database":"ClinMAVE","source_record_id":f"{ds}:{g}:{p}",
            "source_url":src,"HugoSymbol":g,"ProteinChange":p,
            "directionality_label_normalized":label,"label_confidence":"high",
            "label_source":"ClinMAVE reviewed assay-threshold functional classification",
            "label_evidence":f"classification={safe_str(x[cls])}; technique={tech}; phenotype={phen}",
            "is_explicit_variant_level":True,
            "publication_ids":extract_pmids(x[pub]) if pub else "",
            "disease_context":phen,"raw_directionality":safe_str(x[cls]),"raw_effect":safe_str(x[cls]),
            "clinmave_dataset_id":ds,"clinmave_technique":tech,
            "provenance_note":"Many DMS datasets originate from MaveDB; preserve dataset/publication IDs for deduplication.",
        })
    return pd.DataFrame(rows)

def discover_csv_url():
    s=session_with_retry()
    r=s.get(PAGE,timeout=120); r.raise_for_status()
    texts=[r.text]
    for m in re.finditer(r'<script[^>]+src=["\']([^"\']+)["\']',r.text,re.I):
        try:
            rr=s.get(urljoin(r.url,m.group(1)),timeout=60)
            if rr.ok:
                texts.append(rr.text)
        except Exception:
            pass
    candidates=[]
    for text in texts:
        for m in re.finditer(r'["\']([^"\']+\.csv(?:\?[^"\']*)?)["\']',text,re.I):
            u=urljoin(r.url,m.group(1))
            if "clinmave" in u.lower() or "variant" in u.lower():
                candidates.append(u)
    return candidates[0] if candidates else ""

def main():
    ap=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--workdir",type=Path,default=Path("clinmave_cache"))
    ap.add_argument("--input-file",type=Path)
    ap.add_argument("--download-url",default="")
    ap.add_argument("--force-download",action="store_true")
    ap.add_argument("--summary-json",type=Path)
    args=ap.parse_args()
    args.workdir.mkdir(parents=True,exist_ok=True)
    if args.input_file:
        path=args.input_file; src=str(path)
    else:
        url=args.download_url or discover_csv_url()
        if not url:
            raise RuntimeError(
                "ClinMAVE currently exposes variant export through its web UI but no stable documented all-variant URL "
                "was discoverable. Open https://ngdc.cncb.ac.cn/clinmave/download, use 'Retrieve variants by Gene' -> "
                "'Download CSV', then pass the CSV with --input-file. If you capture the CSV request URL in the browser "
                "Network tab, pass it with --download-url."
            )
        path=download(url,args.workdir/"clinmave_variants.csv",args.force_download); src=url
    raw=read_table(path)
    df=parse(raw,src)
    df=add_conflict_flag(df if len(df) else ensure_canonical_columns(pd.DataFrame()))
    write_parquet_safe(df,args.out)
    write_summary(df,args.summary_json,{"source":"ClinMAVE","input_rows":len(raw),"input_file":str(path)})

if __name__=="__main__":
    main()
