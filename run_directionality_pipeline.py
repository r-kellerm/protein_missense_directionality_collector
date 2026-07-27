#!/usr/bin/env python3
"""Run source collectors, build an OncoKB query universe, unify annotations, and report counts."""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path
from typing import List, Dict
import pandas as pd

HERE=Path(__file__).resolve().parent

def log(msg:str)->None: print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}",file=sys.stderr,flush=True)

def run(cmd:List[str],name:str,continue_on_error:bool)->Dict:
    log(f'Running {name}: '+' '.join(cmd))
    r=subprocess.run(cmd,text=True,capture_output=True)
    if r.stdout:print(r.stdout)
    if r.stderr:print(r.stderr,file=sys.stderr)
    result={'name':name,'returncode':r.returncode,'command':cmd,'stdout_tail':r.stdout[-2000:],'stderr_tail':r.stderr[-2000:]}
    if r.returncode and not continue_on_error:raise RuntimeError(f'{name} failed with exit code {r.returncode}')
    return result

def add_if(items:List[str],flag:str,value)->None:
    if value not in (None,''):items.extend([flag,str(value)])

def main()->None:
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--outdir',required=True,type=Path)
    p.add_argument('--python',default=sys.executable)
    p.add_argument('--continue-on-error',action='store_true')
    # Existing sources
    p.add_argument('--gofcards-input',type=Path);p.add_argument('--skip-gofcards',action='store_true')
    p.add_argument('--mavedb-assay-map',type=Path);p.add_argument('--skip-mavedb',action='store_true')
    p.add_argument('--depmap-mutation-file',type=Path);p.add_argument('--depmap-use-cancermine',action='store_true');p.add_argument('--skip-depmap',action='store_true')
    # New sources
    p.add_argument('--civic-input',type=Path);p.add_argument('--civic-download-url');p.add_argument('--skip-civic',action='store_true')
    p.add_argument('--cgi-mutations-file',type=Path);p.add_argument('--cgi-mutations-url');p.add_argument('--cgi-genes-file',type=Path);p.add_argument('--skip-cgi',action='store_true')
    p.add_argument('--skip-oncokb',action='store_true');p.add_argument('--oncokb-demo',action='store_true');p.add_argument('--oncokb-tumor-type',default='')
    p.add_argument('--oncokb-token-env',default='ONCOKB_API_TOKEN')
    p.add_argument('--extra-input',action='append',default=[],help='Additional SOURCE=parquet for unification')
    p.add_argument('--conflict-policy',choices=['ambiguous','weighted'],default='ambiguous')
    args=p.parse_args(); out=args.outdir;out.mkdir(parents=True,exist_ok=True); summaries=out/'summaries';summaries.mkdir(exist_ok=True)
    produced=[]; runs=[]
    def execute(script:str,cmdargs:List[str],name:str,outfile:Path,register:bool=True):
        res=run([args.python,str(HERE/script)]+cmdargs,name,args.continue_on_error);runs.append(res)
        if register and res['returncode']==0 and outfile.exists():produced.append((name,outfile))
    if not args.skip_gofcards:
        f=out/'gofcards_directionality.parquet'; c=['--out',str(f),'--workdir',str(out/'gofcards_cache'),'--missense-only','--summary-json',str(summaries/'gofcards.json')]
        if args.gofcards_input:c+=['--input-file',str(args.gofcards_input)]
        execute('gofcards_directionality_table_fixed.py',c,'GoFCards',f)
    if not args.skip_mavedb:
        f=out/'mavedb_directionality.parquet';c=['--all-published','--out',str(f),'--workdir',str(out/'mavedb_cache'),'--missense-only','--labeled-only','--summary-json',str(summaries/'mavedb.json'),'--errors-csv',str(out/'mavedb_errors.csv')]
        if args.mavedb_assay_map:c+=['--assay-map-csv',str(args.mavedb_assay_map)]
        execute('mavedb_directionality_table_fixed.py',c,'MaveDB',f)
    if not args.skip_depmap:
        full=out/'depmap_directionality_full.parquet';c=['--out',str(full),'--workdir',str(out/'depmap_cache'),'--missense-only']
        if args.depmap_mutation_file:c+=['--mutation-file',str(args.depmap_mutation_file)]
        if args.depmap_use_cancermine:c+=['--use-cancermine']
        execute('depmap_directionality_table_fixed.py',c,'DepMap-full',full,register=False)
        if full.exists():
            f=out/'depmap_directionality.parquet'; execute('collapse_depmap_directionality_table_fixed.py',['--input',str(full),'--out',str(f),'--collapse-level','protein','--drop-unknown','--drop-ambiguous','--summary-json',str(summaries/'depmap.json')],'DepMap',f)
    if not args.skip_civic:
        f=out/'civic_directionality.parquet';c=['--out',str(f),'--workdir',str(out/'civic_cache'),'--summary-json',str(summaries/'civic.json')]
        if args.civic_input:c+=['--input-file',str(args.civic_input)]
        if args.civic_download_url:c+=['--download-url',args.civic_download_url]
        execute('civic_directionality_table.py',c,'CIViC',f)
    if not args.skip_cgi and (args.cgi_mutations_file or args.cgi_mutations_url):
        f=out/'cgi_directionality.parquet';c=['--out',str(f),'--workdir',str(out/'cgi_cache'),'--summary-json',str(summaries/'cgi.json')]
        if args.cgi_mutations_file:c+=['--mutations-file',str(args.cgi_mutations_file)]
        if args.cgi_mutations_url:c+=['--mutations-url',args.cgi_mutations_url]
        if args.cgi_genes_file:c+=['--genes-file',str(args.cgi_genes_file)]
        execute('cgi_directionality_table.py',c,'CGI',f)
    elif not args.skip_cgi:
        log('CGI skipped: provide --cgi-mutations-file or --cgi-mutations-url for the downloadable mutation catalog.')
    # Build unique variant universe from all successful collectors for OncoKB.
    oncokb_available = args.oncokb_demo or bool(os.environ.get(args.oncokb_token_env, ''))
    if not args.skip_oncokb and produced and oncokb_available:
        seeds=[]
        for _,path in produced:
            try:
                df=pd.read_parquet(path)
                g=next((x for x in ['HugoSymbol','gene','symbol'] if x in df.columns),None); pc=next((x for x in ['ProteinChange','protein_change','HGVSp'] if x in df.columns),None)
                if g and pc:seeds.append(df[[g,pc]].rename(columns={g:'HugoSymbol',pc:'ProteinChange'}))
            except Exception as exc:log(f'Could not use {path} for OncoKB seed: {exc}')
        if seeds:
            seed=pd.concat(seeds,ignore_index=True).dropna().drop_duplicates();seed_path=out/'oncokb_variant_universe.parquet';seed.to_parquet(seed_path,index=False)
            f=out/'oncokb_directionality.parquet';c=['--input',str(seed_path),'--out',str(f),'--workdir',str(out/'oncokb_cache'),'--summary-json',str(summaries/'oncokb.json'),'--errors-csv',str(out/'oncokb_errors.csv'),'--token-env',args.oncokb_token_env]
            if args.oncokb_demo:c+=['--demo']
            if args.oncokb_tumor_type:c+=['--tumor-type',args.oncokb_tumor_type]
            execute('oncokb_directionality_table.py',c,'OncoKB',f)
    elif not args.skip_oncokb and not oncokb_available:
        log(f'OncoKB skipped: no token found in {args.oncokb_token_env}; use --oncokb-demo only for limited API testing.')
    unify_inputs=[f'{name}={path}' for name,path in produced]+args.extra_input
    if not unify_inputs:raise SystemExit('No source output was produced; nothing to unify.')
    unified=out/'all_resources_directionality.parquet'; conflicts=out/'all_resources_conflicts.parquet'; evidence=out/'all_resources_evidence.parquet';summary=summaries/'unified.json'
    c=[]
    for value in unify_inputs:c+=['--input',value]
    c+=['--out',str(unified),'--conflicts-out',str(conflicts),'--evidence-out',str(evidence),'--summary-json',str(summary),'--conflict-policy',args.conflict_policy]
    execute('unify_directionality_resources.py',c,'Unifier',unified)
    # Organized report
    records=[]
    for name,path in produced:
        try:
            df=pd.read_parquet(path);lab=next((x for x in ['directionality_label_normalized','directionality_label','directionality_label_heuristic'] if x in df.columns),None)
            counts=df[lab].value_counts(dropna=False).to_dict() if lab else {}
            records.append({'resource':name,'path':str(path),'rows':len(df),'unique_protein_variants':df.get('protein_variant_key',pd.Series(dtype=str)).replace('',pd.NA).nunique(),**{f'label_{k}':v for k,v in counts.items()}})
        except Exception as exc:records.append({'resource':name,'path':str(path),'error':str(exc)})
    report=pd.DataFrame(records);report.to_csv(out/'resource_annotation_report.csv',index=False)
    final={'outputs':{'unified':str(unified),'conflicts':str(conflicts),'evidence':str(evidence),'report_csv':str(out/'resource_annotation_report.csv')},'resources':records,'runs':runs}
    (out/'pipeline_report.json').write_text(json.dumps(final,indent=2,default=str))
    print(report.fillna('').to_string(index=False));print(json.dumps(final['outputs'],indent=2))
if __name__=='__main__':main()
