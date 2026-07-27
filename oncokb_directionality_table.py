#!/usr/bin/env python3
"""Annotate a supplied missense-variant universe with OncoKB biological effect.

OncoKB does not provide a bulk download of all annotated variants. This script
therefore requires --input containing at least gene and protein-change columns,
and queries the licensed API (or the limited demo instance).
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
from pathlib import Path
from typing import Any, Dict
from urllib.parse import quote
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from directionality_common import *


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def session_with_retry(token: str = '') -> requests.Session:
    retry = Retry(total=5, connect=5, read=5, status=5, backoff_factor=1.0,
                  status_forcelist=(429,500,502,503,504), allowed_methods=frozenset({'GET'}),
                  respect_retry_after_header=True)
    s = requests.Session(); a = HTTPAdapter(max_retries=retry)
    s.mount('https://', a); s.mount('http://', a)
    s.headers.update({'Accept':'application/json','User-Agent':'oncokb-directionality-builder/1.0'})
    if token: s.headers['Authorization'] = f'Bearer {token}'
    return s


def mutation_effect_to_label(effect: Any) -> str:
    t = safe_str(effect).lower()
    if 'gain' in t or 'activat' in t or 'increase' in t:
        return 'gof_like'
    if 'loss' in t or 'inactivat' in t or 'decrease' in t:
        return 'lof'
    if 'switch' in t or 'mixed' in t:
        return 'ambiguous'
    return 'unknown'


def annotate_one(s: requests.Session, base_url: str, gene: str, protein: str,
                 tumor_type: str, reference_genome: str, timeout: int) -> Dict[str, Any]:
    alteration = protein.removeprefix('p.')
    params = {'hugoSymbol':gene,'alteration':alteration,'consequence':'missense_variant','referenceGenome':reference_genome}
    if tumor_type: params['tumorType'] = tumor_type
    r = s.get(base_url.rstrip('/') + '/api/v1/annotate/mutations/byProteinChange', params=params, timeout=timeout)
    r.raise_for_status(); return r.json()


def main() -> None:
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--input', required=True, type=Path, help='CSV/TSV/Parquet variant universe')
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--workdir', type=Path, default=Path('oncokb_cache'))
    p.add_argument('--base-url', default='https://www.oncokb.org')
    p.add_argument('--token-env', default='ONCOKB_API_TOKEN')
    p.add_argument('--demo', action='store_true', help='Use limited demo.oncokb.org instance')
    p.add_argument('--tumor-type', default='')
    p.add_argument('--reference-genome', choices=['GRCh37','GRCh38'], default='GRCh37')
    p.add_argument('--request-delay', type=float, default=0.15)
    p.add_argument('--timeout', type=int, default=120)
    p.add_argument('--force', action='store_true')
    p.add_argument('--summary-json', type=Path)
    p.add_argument('--errors-csv', type=Path)
    args=p.parse_args()

    raw=read_table(args.input)
    gene_col=find_col(raw,['HugoSymbol','Hugo_Symbol','gene','gene_symbol','symbol'],required=True)
    prot_col=find_col(raw,['ProteinChange','protein_change','HGVSp','HGVSp_short','amino_acid_change','alteration'],required=True)
    tumor_col=find_col(raw,['tumor_type','oncotree_code','cancer_type'])
    variants=pd.DataFrame({'HugoSymbol':safe_series(raw[gene_col]).str.upper(), 'ProteinChange':raw[prot_col].map(normalize_protein_change)})
    variants['tumor_type']=safe_series(raw[tumor_col]) if tumor_col else args.tumor_type
    variants=variants[(variants.HugoSymbol!='') & (variants.ProteinChange!='')].drop_duplicates().reset_index(drop=True)

    token='' if args.demo else os.environ.get(args.token_env,'')
    if not args.demo and not token:
        raise SystemExit(f'Missing OncoKB token in environment variable {args.token_env}. Use --demo only for limited BRAF/TP53/ROS1 testing.')
    base='https://demo.oncokb.org' if args.demo else args.base_url
    s=session_with_retry(token); args.workdir.mkdir(parents=True,exist_ok=True)
    rows=[]; errors=[]
    for i,row in variants.iterrows():
        gene=row.HugoSymbol; protein=row.ProteinChange; tumor=safe_str(row.tumor_type)
        stem=re.sub(r'[^A-Za-z0-9_.-]+','_',f'{gene}_{protein}_{tumor or "pan_cancer"}')
        cache=args.workdir/f'{stem}.json'
        try:
            if cache.exists() and cache.stat().st_size and not args.force:
                payload=json.loads(cache.read_text())
            else:
                payload=annotate_one(s,base,gene,protein,tumor,args.reference_genome,args.timeout)
                cache.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
                if args.request_delay: time.sleep(args.request_delay)
            me=payload.get('mutationEffect') or {}; effect=me.get('knownEffect','')
            label=mutation_effect_to_label(effect)
            citations=me.get('citations') or {}; pmids=citations.get('pmids') or []
            explicit=label in {'lof','gof_like'}
            confidence='high' if explicit and pmids else 'medium' if explicit else 'low'
            q=payload.get('query') or {}
            rows.append({
                'source_database':'OncoKB','source_record_id':f'{gene}:{protein}','source_version':safe_str(payload.get('dataVersion')),
                'source_url':base,'HugoSymbol':gene,'ProteinChange':protein,'GenomeAssembly':args.reference_genome,
                'directionality_label_normalized':label,'label_confidence':confidence,
                'label_source':'mutationEffect.knownEffect','label_evidence':f"knownEffect={effect}; oncogenic={safe_str(payload.get('oncogenic'))}; hotspot={payload.get('hotspot')}",
                'is_explicit_variant_level':explicit,'publication_ids':';'.join(map(str,pmids)),'disease_context':tumor,
                'raw_directionality':safe_str(payload.get('oncogenic')),'raw_effect':effect,
                'oncokb_variant_exist':payload.get('variantExist'),'oncokb_allele_exist':payload.get('alleleExist'),
                'oncokb_hotspot':payload.get('hotspot'),'oncokb_variant_summary':safe_str(payload.get('variantSummary')),
                'oncokb_last_update':safe_str(payload.get('lastUpdate')),'oncokb_query_alteration':safe_str(q.get('alteration')),
            })
        except Exception as exc:
            errors.append({'HugoSymbol':gene,'ProteinChange':protein,'tumor_type':tumor,'error':str(exc)})
            log(f'Failed {gene} {protein}: {exc}')
        if (i+1)%100==0: log(f'Processed {i+1:,}/{len(variants):,}')
    out=ensure_canonical_columns(pd.DataFrame(rows) if rows else canonical_empty_frame())
    write_parquet_safe(out,args.out)
    if args.errors_csv:
        args.errors_csv.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(errors).to_csv(args.errors_csv,index=False)
    summary={'source':'OncoKB','input_variants':len(variants),'output_rows':len(out),'errors':len(errors),'labels':out.directionality_label_normalized.value_counts().to_dict()}
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True,exist_ok=True); args.summary_json.write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
