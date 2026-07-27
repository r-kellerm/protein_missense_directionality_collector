#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

AA3_TO_1 = {
    'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C','Gln':'Q','Glu':'E','Gly':'G','His':'H','Ile':'I',
    'Leu':'L','Lys':'K','Met':'M','Phe':'F','Pro':'P','Ser':'S','Thr':'T','Trp':'W','Tyr':'Y','Val':'V',
}
AA1 = set('ACDEFGHIKLMNPQRSTVWY')
PROTEIN_RE = re.compile(r'^(?:p\.)?\(?(?P<ref>[A-Z][a-z]{2}|[ACDEFGHIKLMNPQRSTVWY])(?P<pos>[1-9][0-9]*)(?P<alt>[A-Z][a-z]{2}|[ACDEFGHIKLMNPQRSTVWY])\)?$')

CONFIDENCE_RANK = {'none':0,'low':1,'medium':2,'high':3}
SOURCE_WEIGHT_DEFAULT = {
    'OncoKB': 5.0,
    'CIViC': 4.0,
    'CGI': 4.0,
    'GoFCards': 3.5,
    'MaveDB': 3.0,
    'DepMap': 2.0,
}


def normalize_colname(value: Any) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(value).strip().lower())


def find_col(df: pd.DataFrame, candidates: Sequence[str], required: bool = False) -> Optional[str]:
    exact = {normalize_colname(c): c for c in df.columns}
    for candidate in candidates:
        hit = exact.get(normalize_colname(candidate))
        if hit is not None:
            return hit
    if required:
        raise KeyError(f'Could not find any of {list(candidates)}. Available columns: {list(df.columns)[:100]}')
    return None


def safe_str(value: Any) -> str:
    if value is None:
        return ''
    try:
        if pd.isna(value):
            return ''
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def safe_series(series: pd.Series) -> pd.Series:
    return series.fillna('').astype(str).str.strip()


def read_table(path: Path, **kwargs: Any) -> pd.DataFrame:
    path = Path(path)
    low = path.name.lower()
    if low.endswith(('.parquet', '.pq')):
        return pd.read_parquet(path, **kwargs)
    if low.endswith(('.xlsx', '.xls')):
        return pd.read_excel(path, **kwargs)
    if low.endswith(('.tsv', '.txt', '.maf')):
        return pd.read_csv(path, sep='\t', low_memory=False, comment='#', **kwargs)
    return pd.read_csv(path, low_memory=False, **kwargs)


def write_parquet_safe(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(lambda x: json.dumps(x, sort_keys=True) if isinstance(x, (dict, list, tuple)) else x)
            out[col] = out[col].astype('string')
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)


def normalize_protein_change(value: Any) -> str:
    text = safe_str(value)
    text = re.sub(r'^.*?:p\.', 'p.', text)
    m = PROTEIN_RE.match(text)
    if not m:
        return ''
    ref = m.group('ref'); alt = m.group('alt')
    ref1 = AA3_TO_1.get(ref, ref)
    alt1 = AA3_TO_1.get(alt, alt)
    if ref1 not in AA1 or alt1 not in AA1:
        return ''
    return f'p.{ref1}{m.group("pos")}{alt1}'


def protein_key(gene: Any, protein_change: Any) -> str:
    g = safe_str(gene).upper()
    p = normalize_protein_change(protein_change)
    return f'{g}:{p}' if g and p else ''


def genomic_key(chrom: Any, pos: Any, ref: Any, alt: Any, assembly: Any = '') -> str:
    c = safe_str(chrom).removeprefix('chr')
    p = safe_str(pos)
    r = safe_str(ref).upper(); a = safe_str(alt).upper()
    ass = safe_str(assembly)
    if not (c and p and r and a):
        return ''
    prefix = f'{ass}:' if ass else ''
    return f'{prefix}{c}:{p}:{r}>{a}'


def normalize_direction_label(value: Any) -> str:
    text = safe_str(value).lower().replace('_', ' ').replace('-', ' ')
    text = re.sub(r'\s+', ' ', text)
    if not text:
        return 'unknown'
    if text in {'lof','loss of function','loss function','inactivating','inactivation','decreased function','likely lof'}:
        return 'lof'
    if text in {'gof','gain of function','gain function','activating','activation','increased function','likely gof','gof like','gof like drug supported'}:
        return 'gof_like'
    if text in {'ambiguous','conflicting','mixed','both','indeterminate'}:
        return 'ambiguous'
    return 'unknown'


def normalize_confidence(value: Any, default: str = 'low') -> str:
    text = safe_str(value).lower()
    if text.startswith('high') or text in {'a','validated','expert'}:
        return 'high'
    if text.startswith('med') or text in {'b','c','moderate'}:
        return 'medium'
    if text.startswith('low') or text in {'d','e','inferential','predicted'}:
        return 'low'
    if text in {'none','unknown',''}:
        return default
    return default


def max_confidence(values: Iterable[Any]) -> str:
    best = 'none'
    for value in values:
        c = normalize_confidence(value, default='none')
        if CONFIDENCE_RANK[c] > CONFIDENCE_RANK[best]:
            best = c
    return best


def first_nonempty(values: Iterable[Any]) -> str:
    for value in values:
        text = safe_str(value)
        if text and text.lower() not in {'nan','none','null','na'}:
            return text
    return ''


def join_unique(values: Iterable[Any], max_items: int = 30) -> str:
    seen = []
    for value in values:
        text = safe_str(value)
        if not text or text.lower() in {'nan','none','null','na'} or text in seen:
            continue
        seen.append(text)
        if len(seen) >= max_items:
            break
    return '; '.join(seen)


def canonical_empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        'source_database','source_record_id','source_version','source_url','HugoSymbol','ProteinChange','DNAChange',
        'TranscriptID','Chrom','Pos','Ref','Alt','GenomeAssembly','protein_variant_key','genomic_variant_key',
        'directionality_label_normalized','binary_label_lof0_gof1','label_confidence','label_source','label_evidence',
        'is_missense_like','is_explicit_variant_level','publication_ids','disease_context','raw_directionality','raw_effect',
    ])


def ensure_canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    defaults = canonical_empty_frame().columns
    for col in defaults:
        if col not in out.columns:
            if col in {'binary_label_lof0_gof1'}:
                out[col] = pd.Series(pd.NA, index=out.index, dtype='Int8')
            elif col in {'is_missense_like','is_explicit_variant_level'}:
                out[col] = False
            else:
                out[col] = ''
    out['HugoSymbol'] = safe_series(out['HugoSymbol']).str.upper()
    out['ProteinChange'] = out['ProteinChange'].map(normalize_protein_change)
    out['protein_variant_key'] = [protein_key(g,p) for g,p in zip(out['HugoSymbol'], out['ProteinChange'])]
    if not out['genomic_variant_key'].astype(str).str.len().any():
        out['genomic_variant_key'] = [genomic_key(c,p,r,a,ass) for c,p,r,a,ass in zip(out['Chrom'],out['Pos'],out['Ref'],out['Alt'],out['GenomeAssembly'])]
    out['directionality_label_normalized'] = out['directionality_label_normalized'].map(normalize_direction_label)
    out['binary_label_lof0_gof1'] = pd.Series([
        0 if x == 'lof' else 1 if x == 'gof_like' else pd.NA
        for x in out['directionality_label_normalized']
    ], index=out.index, dtype='Int8')
    out['label_confidence'] = out['label_confidence'].map(normalize_confidence)
    out['is_missense_like'] = out['ProteinChange'].ne('')
    return out
