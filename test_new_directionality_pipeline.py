#!/usr/bin/env python3
from __future__ import annotations
import subprocess, sys, tempfile
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent

def call(script,*args): subprocess.run([sys.executable,str(HERE/script),*map(str,args)],check=True)

def main():
    d=Path(tempfile.mkdtemp(prefix='directionality_test_'))
    pd.DataFrame([
      {'Gene':'BRAF','Variant':'V600E','Evidence Type':'Functional','Evidence Statement':'activating gain-of-function','Evidence Level':'D','Status':'accepted','EID':1},
      {'Gene':'TP53','Variant':'R273C','Evidence Type':'Functional','Evidence Statement':'loss-of-function','Evidence Level':'D','Status':'accepted','EID':2},
      {'Gene':'EGFR','Variant':'L858R','Evidence Type':'Predictive','Evidence Statement':'activating and drug sensitive','Evidence Level':'B','Status':'accepted','EID':3},
    ]).to_csv(d/'civic.tsv',sep='\t',index=False)
    call('civic_directionality_table.py','--input-file',d/'civic.tsv','--out',d/'civic.parquet')
    c=pd.read_parquet(d/'civic.parquet').set_index('HugoSymbol')
    assert c.loc['BRAF','directionality_label_normalized']=='gof_like'
    assert c.loc['TP53','directionality_label_normalized']=='lof'
    assert c.loc['EGFR','directionality_label_normalized']=='unknown'

    pd.DataFrame([{'gene':'BRAF','mutation':'V600E','effect':'loss-of-function','id':'x'}]).to_csv(d/'cgi.tsv',sep='\t',index=False)
    call('cgi_directionality_table.py','--mutations-file',d/'cgi.tsv','--out',d/'cgi.parquet')
    call('unify_directionality_resources.py','--input',f'CIViC={d/"civic.parquet"}','--input',f'CGI={d/"cgi.parquet"}','--out',d/'unified.parquet','--conflicts-out',d/'conflicts.parquet')
    u=pd.read_parquet(d/'unified.parquet').set_index('variant_key');conf=pd.read_parquet(d/'conflicts.parquet')
    assert u.loc['BRAF:p.V600E','directionality_label']=='ambiguous'
    assert bool(u.loc['BRAF:p.V600E','has_conflict'])
    assert len(conf)==1
    print('All new directionality pipeline regression tests passed.')
if __name__=='__main__':main()
