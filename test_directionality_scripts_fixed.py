from pathlib import Path
import importlib.util
import pandas as pd
import numpy as np
import gzip, tempfile


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    import sys
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

BASE = Path(__file__).resolve().parent
G = load('gof', BASE / 'gofcards_directionality_table_fixed.py')
M = load('mave', BASE / 'mavedb_directionality_table_fixed.py')
D = load('dep', BASE / 'depmap_directionality_table_fixed.py')
C = load('col', BASE / 'collapse_depmap_directionality_table_fixed.py')

# GoFCards: computational model is not curated.
df = pd.DataFrame({
    'Gene': ['BRAF'],
    'ProteinChange': ['p.V600E'],
    'model': ['computational model prediction'],
    'score': [0.99],
})
curated, confidence, evidence = G.derive_curated_and_confidence(df)
assert not bool(curated.iloc[0]), (curated, confidence, evidence)
assert confidence.iloc[0] == 'low_predicted'

# GoFCards: reference citation must not become REF allele.
df = pd.DataFrame({
    'Gene': ['BRAF'], 'ProteinChange': ['p.V600E'], 'Chrom': ['7'], 'Pos': [140453136],
    'Reference': ['PMID:12345678'], 'Alt': ['A'], 'PMID': ['12345678']
})
out = G.standardize_gofcards_table(df, 'x', missense_only=True, curated_only=False)
assert out.iloc[0]['Ref'] == ''
assert out.iloc[0]['genomic_variant_key'] == ''

# GoFCards: gzip extraction creates directory.
with tempfile.TemporaryDirectory() as td:
    src = Path(td)/'x.csv.gz'
    with gzip.open(src, 'wb') as f: f.write(b'a,b\n1,2\n')
    outp = G.decompress_if_needed(src, Path(td)/'newdir')
    assert outp.exists() and outp.read_text().startswith('a,b')

# DepMap: searches all consequence columns.
df = pd.DataFrame({'VariantInfo':[''], 'Consequence':['missense_variant']})
assert bool(D.contains_any(df, ['VariantInfo','Consequence'], ['missense']).iloc[0])

# DepMap: ambiguous cannot be high confidence.
df = pd.DataFrame({
    'LikelyLoF':[True], 'Hotspot':[True], 'HessDriver':[False],
    'OncogeneHighImpact':[True], 'TumorSuppressorHighImpact':[True],
    'VepImpact':['HIGH'], 'VariantInfo':['missense_variant'],
    'Consequence':['missense_variant'], 'gene_role':['both'], 'ProteinChange':['p.R1W']
})
out = D.derive_directionality(df)
assert out.iloc[0]['directionality_label_heuristic'] == 'ambiguous'
assert out.iloc[0]['directionality_confidence'] == 'low'
assert bool(out.iloc[0]['variant_missense_flag'])

# DepMap: standardized missing IDs remain empty, not fake keys.
df = pd.DataFrame({'HugoSymbol':['GENE1'], 'ModelID':['M1']})
out = D.standardize_mutation_table(df, 'local')
assert out.iloc[0]['variant_key'] == '' and out.iloc[0]['protein_key'] == '' and out.iloc[0]['dna_key'] == ''
assert not bool(out.iloc[0]['has_valid_variant_identifier'])

# Collapse: normalize drug-supported GOF label and use ModelID_standard.
df = pd.DataFrame({
    'ModelID_standard':['M1'], 'HugoSymbol':['BRAF'], 'ProteinChange':['p.V600E'],
    'DNAChange':['c.1799T>A'], 'Chrom':['7'], 'Pos':['140453136'], 'Ref':['T'], 'Alt':['A'],
    'VariantInfo':['missense'], 'VepImpact':['MODERATE'], 'gene_role':['oncogene'],
    'directionality_label_with_drug_support':['gof_like_drug_supported'],
    'directionality_confidence':['medium'], 'directionality_evidence':['drug'],
    'drug_response_supports_gof':[True], 'drug_response_gof_confidence':['high_target_matched'],
    'drug_response_evidence':['support']
})
out = C.collapse(df, 'protein', 'directionality_label_with_drug_support')
assert out.iloc[0]['directionality_label'] == 'gof_like', out
assert out.iloc[0]['n_models_with_variant'] == 1
assert out.iloc[0]['n_label_gof_like_records'] == 1
assert out.iloc[0]['drug_response_gof_confidence'] == 'high_target_matched'

# Collapse: unidentified rows do not merge by default; they are dropped.
df = pd.DataFrame({
    'ModelID_standard':['M1','M2'], 'HugoSymbol':['GENE1','GENE1'], 'ProteinChange':['',''],
    'DNAChange':['',''], 'Chrom':['',''], 'Pos':['',''], 'Ref':['',''], 'Alt':['',''],
    'directionality_label_heuristic':['lof','gof_like'], 'directionality_confidence':['high','medium'],
    'directionality_evidence':['a','b'], 'drug_response_supports_gof':[False,False],
    'drug_response_gof_confidence':['none','none'], 'drug_response_evidence':['','']
})
out = C.collapse(df, 'protein', 'directionality_label_heuristic')
assert len(out) == 0
out_keep = C.collapse(df, 'protein', 'directionality_label_heuristic', keep_unidentified=True)
assert len(out_keep) == 2

# Gene split: same gene must always get same split.
df = pd.DataFrame({'HugoSymbol':['BRAF','BRAF','TP53'], 'ProteinChange':['p.V1A','p.V2A','p.R1W']})
s = C.assign_split(df, 'seed', .1, .1, split_level='gene')
assert s.iloc[0] == s.iloc[1]

# MaveDB: curated threshold outranks conflicting direct free text.
row = pd.Series({'score': -2.0, 'directionality_label': 'gain of function', 'mavedb_functional_classification':'abnormal'})
call = M.choose_direction_call(
    row,
    annotation_cols=['directionality_label'],
    direct_direction_cols=['directionality_label'],
    assay_mapping={'lof_min':None,'lof_max':-1.0,'gof_min':1.0,'gof_max':None,'score_column':'score','notes':''},
    metadata_direction='', metadata_rule='', allow_metadata_directionality=True,
)
assert call.label == 'lof' and call.source == 'curated_score_threshold', call

# MaveDB binary label stays nullable integer.
raw = pd.DataFrame({'hgvs_pro':['p.Ala1Val'], 'score':[1.0], 'directionality':['gain of function']})
out = M.standardize_score_table(
    raw, urn='urn:mavedb:test-a-1', meta={'title':'x'}, endpoint_name='scores',
    statements=pd.DataFrame(), mapped_variants=pd.DataFrame(), assay_mapping=None,
    allow_metadata_directionality=False, preserve_raw_columns=True,
)
assert str(out['binary_label_lof0_gof1'].dtype) == 'Int8', out.dtypes


# Stop-codon HGVS must not be treated as missense.
df = pd.DataFrame({"ProteinChange":["p.Arg100Ter"], "Consequence":[""]})
assert not bool(G.derive_missense_flag(df).iloc[0])
df["gene_role"] = "unknown"
out = D.derive_directionality(df)
assert not bool(out.iloc[0]["variant_missense_flag"])

# Unrecognized integrated labels become unknown and are counted as such.
df = pd.DataFrame({
    "ModelID_standard":["M1"], "HugoSymbol":["X"], "ProteinChange":["p.A1V"],
    "directionality_label_heuristic":["unexpected_label"], "directionality_confidence":["high"],
    "directionality_evidence":["x"], "drug_response_supports_gof":[False],
    "drug_response_gof_confidence":["none"], "drug_response_evidence":[""]
})
out = C.collapse(df, "protein", "directionality_label_heuristic")
assert out.iloc[0]["directionality_label"] == "unknown"
assert out.iloc[0]["n_label_unknown_records"] == 1

print('ALL TESTS PASSED')
