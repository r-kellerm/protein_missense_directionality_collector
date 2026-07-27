#!/usr/bin/env python3
"""Build conservative missense LOF/GOF annotations from CIViC release tables.

Preferred input is a CIViC EvidenceSummaries TSV because Evidence Items retain
functional/oncogenic statements and source metadata. A VariantSummaries TSV can
also be supplied, but directionality is then inferred only from explicit terms.
"""
from __future__ import annotations
import argparse, json, re, sys, time
from pathlib import Path
from typing import Any, List, Tuple
from urllib.parse import urljoin
import pandas as pd
import requests
from directionality_common import *

GOF_RE=re.compile(r'\b(gain[ -]?of[ -]?function|activating|increased function|hyperactiv\w*|hypermorph\w*|constitutively active)\b',re.I)
LOF_RE=re.compile(r'\b(loss[ -]?of[ -]?function|inactivating|decreased function|nonfunctional|hypomorph\w*|null[- ]like)\b',re.I)


def log(msg:str)->None: print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}",file=sys.stderr,flush=True)

def explicit_direction(text:Any)->Tuple[str,str]:
    t=safe_str(text); g=GOF_RE.search(t); l=LOF_RE.search(t)
    if g and l:return 'ambiguous',f'GOF={g.group(0)}; LOF={l.group(0)}'
    if g:return 'gof_like',g.group(0)
    if l:return 'lof',l.group(0)
    return 'unknown',''

def discover_download(base_url:str, kind:str)->str:
    # CIViC exposes Download links on entity pages; prefer TSV links matching entity.
    page=base_url.rstrip('/')+('/evidence' if kind=='evidence' else '/variants')
    r=requests.get(page,timeout=120); r.raise_for_status()
    links=re.findall(r'(?:href|src)=["\']([^"\']+)',r.text,re.I)
    candidates=[urljoin(r.url,x) for x in links if ('.tsv' in x.lower() or '/downloads/' in x.lower())]
    tokens=('evidence','evidencesummaries') if kind=='evidence' else ('variant','variantsummaries')
    candidates=[x for x in candidates if any(t in x.lower() for t in tokens)]
    if not candidates: raise RuntimeError(f'Could not discover CIViC {kind} TSV. Download it manually from {page} and pass --input-file.')
    candidates.sort(key=lambda x:(x.lower().endswith('.tsv'), 'nightly' in x.lower()),reverse=True)
    return candidates[0]

def download(url:str,path:Path,force:bool)->Path:
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.stat().st_size and not force:return path
    with requests.get(url,stream=True,timeout=300) as r:
        r.raise_for_status()
        with path.open('wb') as f:
            for chunk in r.iter_content(1024*1024):
                if chunk:f.write(chunk)
    return path

def get(df:pd.DataFrame,names:List[str])->pd.Series:
    c=find_col(df,names); return safe_series(df[c]) if c else pd.Series('',index=df.index)

def parse_gene_variant(df:pd.DataFrame)->Tuple[pd.Series,pd.Series]:
    gene=get(df,['gene','gene_symbol','feature','feature_name','genes'])
    variant=get(df,['variant','variant_name','name','molecular_profile','molecular_profile_name'])
    hgvsp=get(df,['hgvsp','hgvs_p','protein_change','representative_transcript_hgvs'])
    proteins=[]; genes=[]
    for g,v,h in zip(gene,variant,hgvsp):
        gg=safe_str(g).upper()
        if not gg:
            m=re.match(r'^([A-Za-z0-9-]+)\s+',safe_str(v)); gg=m.group(1).upper() if m else ''
        candidates=[h,v]
        p=''
        for candidate in candidates:
            # Search embedded p.X123Y or bare X123Y.
            m=re.search(r'(?:p\.)?\(?([A-Z][a-z]{2}|[ACDEFGHIKLMNPQRSTVWY])([1-9][0-9]*)([A-Z][a-z]{2}|[ACDEFGHIKLMNPQRSTVWY])\)?',safe_str(candidate))
            if m:
                p=normalize_protein_change('p.'+''.join(m.groups())); break
        genes.append(gg); proteins.append(p)
    return pd.Series(genes,index=df.index),pd.Series(proteins,index=df.index)

def main()->None:
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--input-file',type=Path)
    p.add_argument('--download-url')
    p.add_argument('--kind',choices=['evidence','variants'],default='evidence')
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--workdir',type=Path,default=Path('civic_cache'))
    p.add_argument('--base-url',default='https://civicdb.org')
    p.add_argument('--force-download',action='store_true')
    p.add_argument('--accepted-only',action='store_true',default=True)
    p.add_argument('--summary-json',type=Path)
    args=p.parse_args()
    if args.input_file:path=args.input_file; source_url='local'
    else:
        url=args.download_url or discover_download(args.base_url,args.kind)
        path=download(url,args.workdir/Path(url.split('?')[0]).name,args.force_download); source_url=url
    raw=read_table(path)
    status=get(raw,['status','evidence_status','record_status'])
    if args.accepted_only and status.ne('').any(): raw=raw.loc[status.str.lower().isin(['accepted','active'])].copy()
    gene,protein=parse_gene_variant(raw)
    statement=get(raw,['evidence_statement','statement','description','variant_summary','molecular_profile_summary'])
    significance=get(raw,['significance','evidence_significance','clinical_significance'])
    evidence_type=get(raw,['evidence_type','type'])
    direction=get(raw,['evidence_direction','direction'])
    level=get(raw,['evidence_level','level'])
    trust=get(raw,['trust_rating','rating'])
    eid=get(raw,['evidence_id','eid','id','variant_id'])
    source_ids=get(raw,['source_id','pubmed_id','citation_id','source'])
    disease=get(raw,['disease','disease_name'])
    rows=[]
    for idx in raw.index:
        g=gene.loc[idx]; prot=protein.loc[idx]
        if not g or not prot:continue
        # Only explicit mechanistic terms count; predictive sensitivity/resistance is not directionality.
        texts=[statement.loc[idx],significance.loc[idx]]
        calls=[explicit_direction(x) for x in texts]
        labels={x[0] for x in calls if x[0]!='unknown'}
        label='ambiguous' if len(labels)>1 or 'ambiguous' in labels else next(iter(labels),'unknown')
        matched='; '.join(x[1] for x in calls if x[1])
        et=evidence_type.loc[idx].lower()
        explicit=label in {'lof','gof_like'} and (et in {'functional','oncogenic'} or not et)
        if label in {'lof','gof_like'} and et and et not in {'functional','oncogenic'}:
            # Text may mention another alteration; do not use as a direct functional label.
            label='unknown'; explicit=False
        lv=level.loc[idx].upper()
        conf='high' if lv.startswith(('A','B')) and explicit else 'medium' if lv.startswith(('C','D')) and explicit else 'low'
        rows.append({'source_database':'CIViC','source_record_id':eid.loc[idx] or str(idx),'source_version':path.name,'source_url':source_url,
          'HugoSymbol':g,'ProteinChange':prot,'directionality_label_normalized':label,'label_confidence':conf,
          'label_source':'explicit CIViC functional/oncogenic evidence text','label_evidence':f'evidence_type={evidence_type.loc[idx]}; significance={significance.loc[idx]}; direction={direction.loc[idx]}; matched={matched}',
          'is_explicit_variant_level':explicit,'publication_ids':source_ids.loc[idx],'disease_context':disease.loc[idx],
          'raw_directionality':direction.loc[idx],'raw_effect':significance.loc[idx], 'civic_evidence_level':level.loc[idx],
          'civic_trust_rating':trust.loc[idx], 'civic_evidence_statement':statement.loc[idx]})
    out=ensure_canonical_columns(pd.DataFrame(rows) if rows else canonical_empty_frame())
    write_parquet_safe(out,args.out)
    summary={'source':'CIViC','input_rows':len(raw),'output_missense_rows':len(out),'labels':out.directionality_label_normalized.value_counts().to_dict(),'input_file':str(path)}
    if args.summary_json: args.summary_json.parent.mkdir(parents=True,exist_ok=True); args.summary_json.write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
