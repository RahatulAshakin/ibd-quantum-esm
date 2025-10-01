# ibd-quantum-esm
Hybrid quantum–classical pipeline for IBD multi-omics using Qiskit (QSVC + QuantumKernel) and protein mapping with ESM.

### Quick start
```bash
# create demo fasta
echo >seq1> data/demo.fa
echo MKTIIALSYIFCLVFA>> data/demo.fa
echo >seq2>> data/demo.fa
echo GVVDSGTTNSSS>> data/demo.fa

# embed (ESM on CPU)
python -m src embed data\demo.fa --backend esm2_t6_8M_UR50D --outdb results\duckdb\embeddings.duckdb --batch 4

# inspect + train (keep qubits tiny)
python -m src query --db results\duckdb\embeddings.duckdb --limit 10
python -m src train-qsvc --db results\duckdb\embeddings.duckdb --table embeddings --rule median-length --feature-cap 8 --out results\metrics\qsvc.json

# Train + Predict
python -m src train-qsvc --db results/duckdb/embeddings.duckdb --table embeddings --rule median-length --feature-cap 8 --out results/metrics/qsvc.json --model-out results/models/qsvc.joblib
python -m src predict data/demo.fa --model results/models/qsvc.joblib --backend esm2_t6_8M_UR50D --out results/predictions.csv


### Fold (ESM Atlas)

```bash
# Fold one or more sequences from FASTA; writes PDBs to results/structures
python -m src fold data/train.fa --limit 1 --sanitize
# Example output:
# [ok]   seqA: wrote seqA.pdb
# Folded 1/1 sequence(s) into results/structures

### Multi-omics workflow (synthetic demo)

```bash
# ingest labelled multi-omics features (synthetic flare/remission cohort)
python -m src ingest-omics data/ibd_multiomics_synthetic.csv --id-col sample_id --label-col clinical_status \
    --outdb results/duckdb/ibd.duckdb --overwrite

# train logistic baseline on stored features (records accuracy, F1, ROC-AUC, confusion matrix)
python -m src train-omics --db results/duckdb/ibd.duckdb \
    --out results/metrics/omics_classifier.json \
    --model-out results/models/omics_classifier.joblib \
    --pred-out results/predictions_omics.csv

# generate figures + tables from multi-omics predictions (correlation heatmap, PCA, confusion matrix)
python -m src report --db results/duckdb/ibd.duckdb --pred results/predictions_omics.csv \
    --figdir results/figures --tabledir results/tables
```

`train-qsvc-quantum` records classical RBF-SVC baselines alongside quantum-kernel metrics and kernel diagnostics. Sample tuning for the synthetic dataset:

```bash
python -m src train-qsvc-quantum --db results/duckdb/ibd.duckdb --table embeddings \
    --labels-csv data/ibd_multiomics_synthetic.csv --label-col clinical_status \
    --pca-components 3 --reps 1 --shots 1024 --backend statevector \
    --per-class-limit 120 --C 100 --test-size 0.2 --random-state 123 \
    --out results/metrics/qsvc_quantum.json --model-out results/models/qsvc_quantum.joblib
```

> **Note:** The bundled multi-omics file is synthetic and meant to validate the CLI flow. Replace it with a real cohort (e.g., IBDMDB) to pursue the project’s research goal.

### Multi-omics workflow (IBDMDB cohort)

    python -m src ingest-omics data/ibd_multiomics_ibdmdb.csv --id-col sample_id --label-col label \n        --outdb results/duckdb/final_multiomics_ibdmdb.duckdb --table multiomics --overwrite

    python -m src train-omics --db results/duckdb/final_multiomics_ibdmdb.duckdb --table multiomics \n        --random-state 0 --cv-splits 5 \n        --out results/metrics/final_omics_classifier_ibdmdb.json \n        --model-out results/models/final_omics_classifier_ibdmdb.joblib \n        --pred-out results/predictions_final_omics_ibdmdb.csv

    python -m src report --db results/duckdb/final_multiomics_ibdmdb.duckdb \n        --pred results/predictions_final_omics_ibdmdb.csv \n        --figdir results/figures_final_ibdmdb --tabledir results/tables_final_ibdmdb

> These commands produce the real-cohort metrics (F1_macro ≈ 0.80, ROC-AUC ≈ 0.93) summarised below.

### Final metrics

| Pipeline | Dataset | Key metrics | Artifacts |
|----------|---------|-------------|-----------|
| Classical QSVC | ESM embeddings (dummy backend, feature cap = 32) | test accuracy 0.656 (n_train=1080, n_test=360) | results/metrics/final_qsvc.json; results/models/final_qsvc.joblib; results/predictions_final_qsvc.csv |
| Multi-omics logistic baseline | IBDMDB multi-omics cohort | F1_macro = 0.797, ROC_AUC = 0.927 (5-fold CV) | results/metrics/final_omics_classifier_ibdmdb.json; results/models/final_omics_classifier_ibdmdb.joblib; results/predictions_final_omics_ibdmdb.csv; figures in results/figures_final_ibdmdb/; tables in results/tables_final_ibdmdb/ |
| Quantum kernel QSVC | ESM embeddings (classes 1 vs 2, PCA=4) | quantum F1_macro = 0.603, accuracy = 0.604 (baseline F1=0.700, acc=0.708) | results/metrics/final_qsvc_quantum_ec12.json; results/models/final_qsvc_quantum_ec12.joblib; results/models/final_qsvc_quantum_ec12.joblib.preds.csv |
| Multi-omics logistic baseline (demo) | Synthetic flare/remission cohort | F1_macro = 1.000 (5-fold CV, toy) | results/metrics/final_omics_classifier.json; results/models/final_omics_classifier.joblib; results/predictions_final_omics.csv; figures in results/figures_final/; tables in results/tables_final/ |

### Reproducing the final runs

1. Protein embedding + classical QSVC (dummy backend):

        python -m src embed data/train.fa --backend dummy --outdb results/duckdb/final_embeddings.duckdb --batch 16
        python -m src train-qsvc --db results/duckdb/final_embeddings.duckdb --table embeddings --rule median-length --feature-cap 32 --out results/metrics/final_qsvc.json --model-out results/models/final_qsvc.joblib
        python -m src predict data/heldout.fa --model results/models/final_qsvc.joblib --backend dummy --out results/predictions_final_qsvc.csv

2. Multi-omics ingest + logistic classifier + report (IBDMDB cohort):

        python -m src ingest-omics data/ibd_multiomics_ibdmdb.csv --id-col sample_id --label-col label --outdb results/duckdb/final_multiomics_ibdmdb.duckdb --table multiomics --overwrite
        python -m src train-omics --db results/duckdb/final_multiomics_ibdmdb.duckdb --table multiomics --random-state 0 --cv-splits 5 --out results/metrics/final_omics_classifier_ibdmdb.json --model-out results/models/final_omics_classifier_ibdmdb.joblib --pred-out results/predictions_final_omics_ibdmdb.csv
        python -m src report --db results/duckdb/final_multiomics_ibdmdb.duckdb --pred results/predictions_final_omics_ibdmdb.csv --figdir results/figures_final_ibdmdb --tabledir results/tables_final_ibdmdb

3. Quantum kernel QSVC on ESM embeddings (classes 1 vs 2):

        python -m src train-qsvc-quantum --db results/duckdb/uniprot.duckdb --table embeddings_ec12 --labels-csv data/labels_ec12.csv --label-col ec_top --pca-components 4 --angle-scale-grid 0.5 --angle-scale-grid 1.0 --reps-grid 1 --reps-grid 2 --entanglement-grid linear --entanglement-grid full --per-class-limit 120 --C 10 --test-size 0.2 --random-state 123 --out results/metrics/final_qsvc_quantum_ec12.json --model-out results/models/final_qsvc_quantum_ec12.joblib

> Note: The embeddings_ec12 DuckDB table and data/labels_ec12.csv helper file are derived from the provided UniProt dataset; regenerate them with the helper script in scripts/prep_uniprot_ec.py if needed.

> Optional: To reproduce the synthetic demo, rerun step 2 with data/ibd_multiomics_synthetic.csv and the original synthetic output paths (yields the toy F1 ≈ 1.0).


