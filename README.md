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

