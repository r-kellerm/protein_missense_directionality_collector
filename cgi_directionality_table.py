#!/usr/bin/env python3
"""Build missense LOF/GOF-like annotations from Cancer Genome Interpreter catalogs.

Use --mutations-file with CGI's Catalog of Validated Oncogenic Mutations. An
optional --genes-file (Catalog of Cancer Genes) supplies gene mode of action.
Explicit mutation-level functional effects are preferred; oncogene-mode
inference is retained only as a clearly marked heuristic label.
"""
from __future__ import annotations
import argparse, json, re, sys, time
from pathlib import Path
from typing import Any, Tuple
import pandas as pd
import requests
from directionality_common import *

GOF_RE=re.compile(r'\b(gain[ -]?of[ -]?function|activating|activation|hyperactiv\w*|hypermorph\w*|increased function|constitutively active)\b',re.I)
LOF_RE=re.compile(r'\b(loss[ -]?of[ -]?function|inactivating|inactivation|hypomorph\w*|decreased function|null[- ]like|nonfunctional)\b',re.I)


def explicit(text:Any)->Tuple[str,str]:
    t=safe_str(text); g=GOF_RE.search(t); l=LOF_RE.search(t)
    if g and l:return 'ambiguous',f'{g.group(0)} / {l.group(0)}'
    if g:return 'gof_like',g.group(0)
    if l:return 'lof',l.group(0)
    return 'unknown',''

def download(url:str,path:Path,force:bool)->Path:
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.stat().st_size and not force:return path
    with requests.get(url,stream=True,timeout=300) as r:
        r.raise_for_status()
        cd=r.headers.get('content-disposition',''); m=re.search(r'filename\*?=(?:UTF-8\'\')?["\']?([^"\';]+)',cd,re.I)
        if m:path=path.parent/re.sub(r'[^A-Za-z0-9_.-]+','_',m.group(1))
        with path.open('wb') as f:
            for chunk in r.iter_content(1024*1024):
                if chunk:f.write(chunk)
    return path

def col(df:pd.DataFrame,names:list[str])->pd.Series:
    c=find_col(df,names); return safe_series(df[c]) if c else pd.Series('',index=df.index)

def gene_roles(path:Path|None)->dict[str,str]:
    if path is None:return {}
    df=read_table(path); gene=col(df,['gene','gene_symbol','symbol','hugo_symbol']); moa=col(df,['mode_of_action','moa','role','gene_role'])
    roles={}
    for g,m in zip(gene,moa):
        t=m.lower(); role='both' if ('oncogene' in t and ('tumor suppress' in t or 'tsg' in t)) else 'oncogene' if 'oncogene' in t else 'tumor_suppressor' if ('tumor suppress' in t or 'tsg' in t) else 'unknown'
        if g:roles[g.upper()]=role
    return roles

def main()->None:
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--mutations-file',type=Path)
    p.add_argument('--mutations-url',default='')
    p.add_argument('--genes-file',type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--workdir',type=Path,default=Path('cgi_cache'))
    p.add_argument('--force-download',action='store_true')
    p.add_argument('--allow-oncogene-role-heuristic',action='store_true',default=True)
    p.add_argument('--labeled-only',action='store_true')
    p.add_argument('--summary-json',type=Path)
    args=p.parse_args()
    if args.mutations_file:path=args.mutations_file; source='local'
    elif args.mutations_url:path=download(args.mutations_url,args.workdir/'cgi_validated_oncogenic_mutations.tsv',args.force_download);source=args.mutations_url
    else:raise SystemExit('Supply --mutations-file or --mutations-url. CGI download URLs can change; download the Catalog of Validated Oncogenic Mutations from cancergenomeinterpreter.org/mutations when needed.')
    raw=read_table(path); roles=gene_roles(args.genes_file)
    gene=col(raw,['gene','gene_symbol','symbol','hugo_symbol'])
    protein=col(raw,['protein_change','hgvsp','aa_change','mutation','alteration','protein'])
    effect=col(raw,['effect','functional_effect','mutation_effect','consequence','annotation'])
    role_col=col(raw,['mode_of_action','moa','gene_role','role'])
    source_id=col(raw,['id','mutation_id','source_id','variant_id'])
    pub=col(raw,['pmid','pubmed','publication','references','source'])
    disease=col(raw,['cancer_type','tumor_type','disease','cancer'])
    rows=[]
    for i in raw.index:
        g=gene.loc[i].upper(); pchg=normalize_protein_change(protein.loc[i])
        if not g or not pchg:continue
        label,matched=explicit(' | '.join([effect.loc[i],role_col.loc[i]]))
        role=roles.get(g,'unknown')
        if role=='unknown':
            rt=role_col.loc[i].lower(); role='oncogene' if 'oncogene' in rt else 'tumor_suppressor' if ('tumor suppress' in rt or 'tsg' in rt) else 'both' if 'both' in rt else 'unknown'
        source_kind='explicit_mutation_effect'; conf='high' if label in {'lof','gof_like'} else 'low'; explicit_variant=label in {'lof','gof_like'}
        if label=='unknown' and args.allow_oncogene_role_heuristic and role=='oncogene':
            # CGI catalog entries are validated oncogenic mutations; recurrent missense in an oncogene is GOF-like,
            # but this is not equivalent to a directly measured biochemical gain of function.
            label='gof_like'; conf='medium'; source_kind='validated_oncogenic_mutation_plus_oncogene_role'; explicit_variant=False
            matched='CGI validated oncogenic mutation in oncogene'
        rows.append({'source_database':'CGI','source_record_id':source_id.loc[i] or str(i),'source_version':path.name,'source_url':source,
          'HugoSymbol':g,'ProteinChange':pchg,'directionality_label_normalized':label,'label_confidence':conf,
          'label_source':source_kind,'label_evidence':f'gene_role={role}; effect={effect.loc[i]}; matched={matched}',
          'is_explicit_variant_level':explicit_variant,'publication_ids':pub.loc[i],'disease_context':disease.loc[i],
          'raw_directionality':role_col.loc[i],'raw_effect':effect.loc[i],'cgi_gene_role':role})
    out=ensure_canonical_columns(pd.DataFrame(rows) if rows else canonical_empty_frame())
    if args.labeled_only:out=out[out.directionality_label_normalized.isin(['lof','gof_like'])].copy()
    write_parquet_safe(out,args.out)
    summary={'source':'CGI','input_rows':len(raw),'output_missense_rows':len(out),'labels':out.directionality_label_normalized.value_counts().to_dict(),'input_file':str(path)}
    if args.summary_json:args.summary_json.parent.mkdir(parents=True,exist_ok=True);args.summary_json.write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
