#!/usr/bin/env python3
'''Collect experimentally curated human Swiss-Prot MUTAGEN features with explicit GOF/LOF text.'''
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from urllib.parse import urlencode
import pandas as pd
from directionality_public_common import *

BASE="https://rest.uniprot.org/uniprotkb/search"
QUERY="(organism_id:9606) AND (reviewed:true) AND (ft_mutagen:*)"

def recursively_pmids(x):
    hits=set()
    def walk(v):
        if isinstance(v,dict):
            for k,z in v.items():
                if "pubmed" in k.lower():
                    hits.update(re.findall(r"\d{5,9}",safe_str(z)))
                walk(z)
        elif isinstance(v,list):
            for z in v:
                walk(z)
        elif isinstance(v,str) and "pubmed" in v.lower():
            hits.update(re.findall(r"\d{5,9}",v))
    walk(x)
    return ";".join(sorted(hits,key=int))

def next_link(link):
    if not link:
        return ""
    for part in link.split(","):
        if 'rel="next"' in part:
            m=re.search(r"<([^>]+)>",part)
            if m:
                return m.group(1)
    return ""

def main():
    ap=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--workdir",type=Path,default=Path("uniprot_cache"))
    ap.add_argument("--force",action="store_true")
    ap.add_argument("--include-ambiguous",action="store_true")
    ap.add_argument("--summary-json",type=Path)
    args=ap.parse_args()
    args.workdir.mkdir(parents=True,exist_ok=True)
    raw_cache=args.workdir/"uniprot_human_reviewed_mutagenesis.jsonl"
    s=session_with_retry()
    if raw_cache.exists() and raw_cache.stat().st_size and not args.force:
        entries=[json.loads(x) for x in raw_cache.read_text(encoding="utf-8").splitlines() if x.strip()]
    else:
        url=BASE+"?"+urlencode({"query":QUERY,"format":"json","size":500})
        entries=[]
        with raw_cache.open("w",encoding="utf-8") as fout:
            while url:
                log(f"GET {url[:180]}")
                r=s.get(url,timeout=300); r.raise_for_status()
                payload=r.json()
                for e in payload.get("results",[]):
                    entries.append(e)
                    fout.write(json.dumps(e,ensure_ascii=False)+"\n")
                url=next_link(r.headers.get("Link",""))
                log(f"Downloaded {len(entries):,} UniProt entries")
    rows=[]
    for e in entries:
        acc=safe_str(e.get("primaryAccession"))
        gene=""
        for g in e.get("genes") or []:
            gene=safe_str((g.get("geneName") or {}).get("value"))
            if gene:
                break
        gene=gene.upper()
        for f in e.get("features") or []:
            if safe_str(f.get("type")).lower()!="mutagenesis":
                continue
            loc=f.get("location") or {}
            st=(loc.get("start") or {}).get("value")
            en=(loc.get("end") or {}).get("value")
            if not st or st!=en:
                continue
            altobj=f.get("alternativeSequence") or {}
            ref=safe_str(altobj.get("originalSequence"))
            alts=altobj.get("alternativeSequences") or []
            desc=safe_str(f.get("description"))
            label,matched=explicit_directionality(desc)
            if label=="unknown" or (label=="ambiguous" and not args.include_ambiguous):
                continue
            for alt in alts:
                alt=safe_str(alt)
                if not re.fullmatch(rf"[{AA1}]",ref,re.I) or not re.fullmatch(rf"[{AA1}]",alt,re.I) or ref.upper()==alt.upper():
                    continue
                prot=f"p.{ref.upper()}{st}{alt.upper()}"
                rows.append({
                    "source_database":"UniProtKB_SwissProt","source_record_id":f"{acc}:{prot}",
                    "source_version":"","source_url":f"https://rest.uniprot.org/uniprotkb/{acc}.json",
                    "HugoSymbol":gene,"ProteinChange":prot,
                    "directionality_label_normalized":label,"label_confidence":"high",
                    "label_source":"UniProt MUTAGEN experimental feature text",
                    "label_evidence":desc,"is_explicit_variant_level":True,
                    "publication_ids":recursively_pmids(f),
                    "raw_directionality":matched,"raw_effect":desc,
                    "uniprot_accession":acc,"variant_origin":"engineered_mutagenesis",
                })
    df=add_conflict_flag(pd.DataFrame(rows) if rows else ensure_canonical_columns(pd.DataFrame()))
    write_parquet_safe(df,args.out)
    write_summary(df,args.summary_json,{"source":"UniProtKB/Swiss-Prot","entries_scanned":len(entries),"raw_cache":str(raw_cache)})

if __name__=="__main__":
    main()
