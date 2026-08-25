#!/usr/bin/env python3
'''Collect HPMPdb human single-AA function labels (-1 reduced, +1 increased).'''
from __future__ import annotations
import argparse, re
from pathlib import Path
from urllib.parse import urljoin
import pandas as pd
from directionality_public_common import *

BASES=["https://hpmp.esat.kuleuven.be/","http://hpmp.esat.kuleuven.be/"]

def discover_data_url():
    s=session_with_retry(); candidates=[]
    for base in BASES:
        try:
            r=s.get(base,timeout=60); r.raise_for_status()
        except Exception:
            continue
        html=r.text
        for m in re.finditer(r'(?:href|src)=["\']([^"\']+)["\']',html,re.I):
            u=urljoin(r.url,m.group(1)); low=u.lower().split("?")[0]
            if any(x in u.lower() for x in ["download","data","hpmp"]) and any(low.endswith(e) for e in [".tsv",".csv",".txt",".xlsx",".json"]):
                candidates.append(u)
    return candidates[0] if candidates else ""

def parse_function(v):
    s=safe_str(v).lower()
    if s in {"1","+1","1.0"} or re.search(r"\b(increased|gain(?:ed)?|enhanced)\b",s):
        return "gof_like"
    if s in {"-1","-1.0"} or re.search(r"\b(reduced|impaired|decreased|loss)\b",s):
        return "lof"
    return "unknown"

def guess_variant(row):
    gene=""; prot=""; acc=""
    for k,v in row.items():
        nk=normalize_colname(k); s=safe_str(v)
        if not gene and ("gene" in nk or "symbol" in nk) and re.fullmatch(r"[A-Za-z0-9._-]{2,30}",s):
            gene=s.upper()
        if not acc and ("uniprot" in nk or "accession" in nk) and re.fullmatch(r"[A-Z0-9]{6,10}",s,re.I):
            acc=s.upper()
        if not prot and any(z in nk for z in ["variant","mutation","aachange","proteinchange","sav"]):
            prot=normalize_protein_change(s)
    if not prot:
        prot=normalize_protein_change(" | ".join(safe_str(v) for v in row.values))
    return gene,acc,prot

def accession_gene(acc,cache,s):
    if not acc:
        return ""
    if acc in cache:
        return cache[acc]
    try:
        r=s.get(f"https://rest.uniprot.org/uniprotkb/{acc}.json",timeout=60); r.raise_for_status(); e=r.json()
        g=""
        for z in e.get("genes") or []:
            g=safe_str((z.get("geneName") or {}).get("value"))
            if g:
                break
        cache[acc]=g.upper()
    except Exception:
        cache[acc]=""
    return cache[acc]

def main():
    ap=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--workdir",type=Path,default=Path("hpmpdb_cache"))
    ap.add_argument("--input-file",type=Path)
    ap.add_argument("--download-url",default="")
    ap.add_argument("--force-download",action="store_true")
    ap.add_argument("--summary-json",type=Path)
    args=ap.parse_args()
    args.workdir.mkdir(parents=True,exist_ok=True)
    if args.input_file:
        path=args.input_file; src=str(path)
    else:
        url=args.download_url or discover_data_url()
        if not url:
            raise RuntimeError("HPMPdb does not expose a stable bulk URL in the landing page. Download its TSV/CSV export and rerun with --input-file, or pass the current direct URL with --download-url.")
        path=download(url,args.workdir/Path(url.split("?")[0]).name,args.force_download); src=url
    raw=read_table(path)
    fc=find_col(raw,["function","functional phenotype","function label","molecular function"],True)
    rows=[]; gene_cache={}; s=session_with_retry()
    for i,x in raw.iterrows():
        label=parse_function(x[fc])
        if label=="unknown":
            continue
        gene,acc,prot=guess_variant(x)
        if not gene:
            gene=accession_gene(acc,gene_cache,s)
        if not gene or not prot:
            continue
        txt=" | ".join(f"{k}={safe_str(v)}" for k,v in x.items() if safe_str(v))
        rows.append({
            "source_database":"HPMPdb","source_record_id":f"{gene}:{prot}:{i}",
            "source_version":path.name,"source_url":src,
            "HugoSymbol":gene,"ProteinChange":prot,
            "directionality_label_normalized":label,"label_confidence":"high",
            "label_source":"HPMPdb manually normalized UniProt molecular phenotype: function",
            "label_evidence":f"function={safe_str(x[fc])}",
            "is_explicit_variant_level":True,"publication_ids":extract_pmids(txt),
            "raw_directionality":safe_str(x[fc]),"raw_effect":safe_str(x[fc]),
            "uniprot_accession":acc,
            "provenance_note":"HPMPdb is largely derived from UniProt; do not count as independent confirmation of UniProt.",
        })
    df=add_conflict_flag(pd.DataFrame(rows) if rows else ensure_canonical_columns(pd.DataFrame()))
    write_parquet_safe(df,args.out)
    write_summary(df,args.summary_json,{"source":"HPMPdb","input_rows":len(raw),"input_file":str(path)})

if __name__=="__main__":
    main()
