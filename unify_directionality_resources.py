#!/usr/bin/env python3
"""Unify variant-level LOF/GOF annotations from heterogeneous resources.

Conflicts are never silently resolved: any variant with both LOF and GOF-like
support is written to --conflicts-out. The unified table retains a consensus
label plus source/evidence provenance and conflict flags.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from directionality_common import *


def log(msg:str)->None: print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}",file=sys.stderr,flush=True)

def infer_source(path:Path,df:pd.DataFrame)->str:
    if 'source_database' in df.columns and df['source_database'].notna().any():return first_nonempty(df['source_database'])
    low=path.name.lower()
    for name in ['oncokb','civic','gofcards','mavedb','depmap','cgi']:
        if name in low:return 'CGI' if name=='cgi' else name.capitalize() if name!='oncokb' else 'OncoKB'
    return path.stem

def legacy_to_canonical(df:pd.DataFrame,source:str)->pd.DataFrame:
    x=df.copy()
    aliases={
      'HugoSymbol':['HugoSymbol','Hugo_Symbol','gene','gene_symbol','symbol'],
      'ProteinChange':['ProteinChange','protein_change','HGVSp','HGVSp_short','amino_acid_change','alteration'],
      'DNAChange':['DNAChange','dna_change','HGVSc','cdna_change'],
      'Chrom':['Chrom','Chromosome','chr'], 'Pos':['Pos','position','Start_Position','start'],
      'Ref':['Ref','Reference_Allele','ref'], 'Alt':['Alt','Tumor_Seq_Allele2','alt'],
      'directionality_label_normalized':['directionality_label_normalized','directionality_label','directionality_label_heuristic','directionality_label_with_drug_support','binary_label_name'],
      'label_confidence':['label_confidence','directionality_confidence','confidence'],
      'label_evidence':['label_evidence','directionality_evidence','evidence'],
      'label_source':['label_source','gene_role_source','source_tier'],
      'publication_ids':['publication_ids','PMID_or_reference','publications','pmid'],
      'disease_context':['disease_context','DiseaseOrPhenotype','disease','tumor_type'],
      'source_record_id':['source_record_id','variant_urn','variant_key','protein_key','id'],
      'source_version':['source_version','source_release','data_version'],
      'is_explicit_variant_level':['is_explicit_variant_level'],
    }
    out=pd.DataFrame(index=x.index)
    for canon,cands in aliases.items():
        c=find_col(x,cands); out[canon]=x[c] if c else (False if canon=='is_explicit_variant_level' else '')
    out['source_database']=source
    out['source_url']=''
    out['TranscriptID']='';out['GenomeAssembly']='';out['raw_directionality']='';out['raw_effect']=''
    out=ensure_canonical_columns(out)
    return out

def parse_input(value:str)->Tuple[str|None,Path]:
    if '=' in value:
        s,p=value.split('=',1);return s,Path(p)
    return None,Path(value)

def score_row(row:pd.Series,weights:Dict[str,float])->float:
    base=weights.get(safe_str(row.source_database),1.0)
    conf={'high':1.0,'medium':0.7,'low':0.35,'none':0.1}.get(normalize_confidence(row.label_confidence,'none'),0.1)
    explicit=1.2 if bool(row.is_explicit_variant_level) else 1.0
    return base*conf*explicit

def main()->None:
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--input',action='append',required=True,help='Path or SOURCE=path; repeat for each resource')
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--conflicts-out',required=True,type=Path)
    p.add_argument('--evidence-out',type=Path,help='Optional normalized row-level evidence table')
    p.add_argument('--summary-json',type=Path)
    p.add_argument('--include-unknown',action='store_true')
    p.add_argument('--conflict-policy',choices=['ambiguous','weighted'],default='ambiguous')
    p.add_argument('--minimum-weight-margin',type=float,default=1.5)
    args=p.parse_args()
    frames=[]; source_counts={}
    for item in args.input:
        source_override,path=parse_input(item); df=read_table(path); source=source_override or infer_source(path,df)
        c=legacy_to_canonical(df,source); c['input_path']=str(path); frames.append(c); source_counts[source]=len(c)
        log(f'{source}: {len(c):,} rows from {path}')
    evidence=pd.concat(frames,ignore_index=True,sort=False) if frames else canonical_empty_frame()
    evidence=evidence[(evidence.HugoSymbol!='') & (evidence.ProteinChange!='')].copy()
    if not args.include_unknown:evidence=evidence[evidence.directionality_label_normalized.isin(['lof','gof_like','ambiguous'])].copy()
    evidence['variant_key']=evidence['protein_variant_key']
    weights=SOURCE_WEIGHT_DEFAULT.copy(); evidence['evidence_weight']=evidence.apply(lambda r:score_row(r,weights),axis=1)
    rows=[]; conflicts=[]
    for key,g in evidence.groupby('variant_key',sort=False):
        labels=set(g.directionality_label_normalized)
        lof=g[g.directionality_label_normalized=='lof']; gof=g[g.directionality_label_normalized=='gof_like']; amb=g[g.directionality_label_normalized=='ambiguous']
        lof_w=float(lof.evidence_weight.sum());gof_w=float(gof.evidence_weight.sum())
        conflict=not lof.empty and not gof.empty
        if conflict and args.conflict_policy=='weighted' and abs(lof_w-gof_w)>=args.minimum_weight_margin:
            final='lof' if lof_w>gof_w else 'gof_like'; reason=f'weighted conflict resolution: lof={lof_w:.3f}, gof={gof_w:.3f}'
        elif conflict:
            final='ambiguous';reason=f'cross-resource conflict: lof={lof_w:.3f}, gof={gof_w:.3f}'
        elif not lof.empty:final='lof';reason='one or more LOF annotations and no GOF-like annotation'
        elif not gof.empty:final='gof_like';reason='one or more GOF-like annotations and no LOF annotation'
        elif not amb.empty:final='ambiguous';reason='only ambiguous annotations'
        else:final='unknown';reason='no directional annotation'
        support=g[g.directionality_label_normalized==final] if final in {'lof','gof_like'} else g
        rec={'variant_key':key,'HugoSymbol':first_nonempty(g.HugoSymbol),'ProteinChange':first_nonempty(g.ProteinChange),
          'directionality_label':final,'binary_label_lof0_gof1':0 if final=='lof' else 1 if final=='gof_like' else np.nan,
          'directionality_confidence':max_confidence(support.label_confidence),'consensus_reason':reason,'has_conflict':conflict,
          'n_sources':int(g.source_database.nunique()),'sources':join_unique(g.source_database,50),
          'n_evidence_rows':len(g),'n_lof_rows':len(lof),'n_gof_like_rows':len(gof),'n_ambiguous_rows':len(amb),
          'lof_weight':lof_w,'gof_like_weight':gof_w,'publication_ids':join_unique(g.publication_ids,100),
          'disease_contexts':join_unique(g.disease_context,50),'label_evidence':join_unique(g.apply(lambda r:f'[{r.source_database}] {r.label_evidence}',axis=1),100)}
        rows.append(rec)
        if conflict:
            conflicts.append({**rec,'lof_sources':join_unique(lof.source_database,50),'gof_like_sources':join_unique(gof.source_database,50),
             'lof_evidence':join_unique(lof.apply(lambda r:f'[{r.source_database}] {r.label_evidence}',axis=1),100),
             'gof_like_evidence':join_unique(gof.apply(lambda r:f'[{r.source_database}] {r.label_evidence}',axis=1),100)})
    unified=pd.DataFrame(rows); conflict_df=pd.DataFrame(conflicts)
    write_parquet_safe(unified,args.out);write_parquet_safe(conflict_df,args.conflicts_out)
    if args.evidence_out:write_parquet_safe(evidence,args.evidence_out)
    summary={'input_rows_by_source':source_counts,'normalized_evidence_rows':len(evidence),'unique_variants':len(unified),
      'conflicting_variants':len(conflict_df),'consensus_labels':unified.directionality_label.value_counts().to_dict() if len(unified) else {}}
    if args.summary_json:args.summary_json.parent.mkdir(parents=True,exist_ok=True);args.summary_json.write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
