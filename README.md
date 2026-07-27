# Multi-resource LOF/GOF directionality pipeline

## New collectors

- `oncokb_directionality_table.py`: annotates a supplied missense-variant universe with the OncoKB API. Set `ONCOKB_API_TOKEN`; `--demo` is only for limited BRAF/TP53/ROS1 testing.
- `civic_directionality_table.py`: parses a CIViC Evidence or Variant release TSV. Only explicit functional/oncogenic LOF/GOF language is converted into directional labels.
- `cgi_directionality_table.py`: parses CGI's Catalog of Validated Oncogenic Mutations and optionally the Catalog of Cancer Genes. Explicit mutation effects are preferred; oncogene-role inference is marked as heuristic.
- `unify_directionality_resources.py`: normalizes all source tables, produces one variant-level consensus table, and writes LOF/GOF disagreements separately.
- `run_directionality_pipeline.py`: invokes the existing GoFCards, MaveDB and DepMap scripts, the three new collectors, OncoKB annotation when credentials are available, and the unifier.

## Dependencies

```bash
pip install pandas numpy requests pyarrow scipy openpyxl
```

## Example complete run

```bash
export ONCOKB_API_TOKEN='...'

python run_directionality_pipeline.py \
  --outdir data/directionality \
  --mavedb-assay-map mavedb_assay_directionality_template.csv \
  --cgi-mutations-file CGI_validated_oncogenic_mutations.tsv \
  --cgi-genes-file CGI_cancer_genes.tsv
```

Optional local inputs can be passed with `--gofcards-input`, `--depmap-mutation-file`, and `--civic-input`. CGI is skipped unless a mutation catalog file or URL is supplied. OncoKB is skipped unless the configured token environment variable exists or `--oncokb-demo` is selected.

## Principal outputs

- `all_resources_directionality.parquet`: one row per protein missense variant.
- `all_resources_conflicts.parquet`: variants supported as both LOF and GOF-like.
- `all_resources_evidence.parquet`: normalized source-level evidence rows.
- `resource_annotation_report.csv`: counts by resource and label.
- `pipeline_report.json`: commands, statuses and output locations.

## Conflict policy

The default is `ambiguous`: any LOF/GOF disagreement remains ambiguous. `--conflict-policy weighted` is available, but the conflict remains flagged and retained in the conflict table.

## Tests

```bash
python test_directionality_scripts_fixed.py
python test_new_directionality_pipeline.py
```
