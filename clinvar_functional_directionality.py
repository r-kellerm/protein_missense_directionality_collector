#!/usr/bin/env python3
'''Collect explicit human ClinVar protein GOF/LOF functional consequences from the weekly VCV XML.'''
from __future__ import annotations
import argparse, gzip, re, xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
from directionality_public_common import *

URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/xml/ClinVarVCVRelease_00-latest.xml.gz"

def local(tag: str) -> str:
    return tag.rsplit("}",1)[-1]

def all_text(el) -> str:
    bits=[]
    for x in el.iter():
        if x.text and x.text.strip():
            bits.append(x.text.strip())
        bits.extend(safe_str(v) for v in x.attrib.values() if safe_str(v))
    return " | ".join(bits)

def gene_symbol(archive) -> str:
    for g in archive.iter():
        if local(g.tag).lower() == "gene":
            for x in g.iter():
                if local(x.tag).lower() in {"symbol","genesymbol"}:
                    txt=safe_str(x.text)
                    if txt and re.fullmatch(r"[A-Za-z0-9._-]{1,31}",txt):
                        return txt.upper()
                if local(x.tag).lower()=="elementvalue" and safe_str(x.attrib.get("Type")).lower()=="preferred":
                    txt=safe_str(x.text)
                    if txt and re.fullmatch(r"[A-Za-z0-9._-]{1,31}",txt):
                        return txt.upper()
    for x in archive.iter():
        for k,v in x.attrib.items():
            if normalize_colname(k) in {"genesymbol","symbol"}:
                txt=safe_str(v)
                if re.fullmatch(r"[A-Za-z0-9._-]{1,31}",txt):
                    return txt.upper()
    return ""

def protein_change(archive) -> str:
    candidates=[]
    for x in archive.iter():
        if local(x.tag).lower() in {"hgvs","name","expression","nucleotideexpression","proteinexpression","elementvalue"}:
            candidates.append(all_text(x))
    for txt in candidates:
        p=normalize_protein_change(txt)
        if p:
            return p
    return normalize_protein_change(all_text(archive))

def functional_consequences(archive):
    for x in archive.iter():
        if local(x.tag).lower()=="functionalconsequence":
            text=all_text(x)
            low=text.lower()
            if "protein gain of function" in low or re.search(r"\bgain[ -]?of[ -]?function\b",low):
                yield "gof_like", text
            if "protein loss of function" in low or re.search(r"\bloss[ -]?of[ -]?function\b",low):
                yield "lof", text

def pmids(archive) -> str:
    hits=set()
    for x in archive.iter():
        attrs=" ".join(safe_str(v) for v in x.attrib.values()).lower()
        if "pubmed" in attrs or local(x.tag).lower() in {"citation","id"}:
            for h in re.findall(r"(?<!\d)(\d{5,9})(?!\d)", all_text(x)):
                hits.add(h)
    return ";".join(sorted(hits,key=int))

def main():
    ap=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--workdir",type=Path,default=Path("clinvar_cache"))
    ap.add_argument("--xml-file",type=Path)
    ap.add_argument("--force-download",action="store_true")
    ap.add_argument("--summary-json",type=Path)
    args=ap.parse_args()
    args.workdir.mkdir(parents=True,exist_ok=True)
    path=args.xml_file or download(URL,args.workdir/"ClinVarVCVRelease_00-latest.xml.gz",args.force_download)
    rows=[]; n=0
    opener=gzip.open if str(path).lower().endswith(".gz") else open
    with opener(path,"rb") as fh:
        for _,el in ET.iterparse(fh,events=("end",)):
            if local(el.tag).lower()!="variationarchive":
                continue
            n+=1
            gene=gene_symbol(el); prot=protein_change(el)
            if gene and prot:
                effects=list(functional_consequences(el))
                if effects:
                    pid=pmids(el)
                    vid=safe_str(el.attrib.get("VariationID") or el.attrib.get("Accession"))
                    full=all_text(el)
                    experimental=bool(re.search(r"\b(experiment|functional assay|functional stud|in vitro|in vivo)\b",full,re.I))
                    for label,evidence in effects:
                        rows.append({
                            "source_database":"ClinVar","source_record_id":vid,"source_version":path.name,
                            "source_url":URL,"HugoSymbol":gene,"ProteinChange":prot,
                            "directionality_label_normalized":label,
                            "label_confidence":"high" if experimental or pid else "medium",
                            "label_source":"ClinVar.FunctionalConsequence",
                            "label_evidence":evidence[:4000],"is_explicit_variant_level":True,
                            "publication_ids":pid,"raw_directionality":evidence,"raw_effect":evidence,
                            "clinvar_experiment_text_detected":experimental,
                        })
            el.clear()
            if n%100000==0:
                log(f"Parsed {n:,} VariationArchive records; retained {len(rows):,} directional rows")
    df=add_conflict_flag(pd.DataFrame(rows) if rows else ensure_canonical_columns(pd.DataFrame()))
    write_parquet_safe(df,args.out)
    write_summary(df,args.summary_json,{"source":"ClinVar","variation_archives_scanned":n,"input_file":str(path)})

if __name__=="__main__":
    main()
