# ibd-quantum-esm
Hybrid quantum–classical pipeline for IBD multi-omics using Qiskit (QSVC + QuantumKernel) and protein mapping with ESM.

## Quick start
```bash
conda env create -f environment.yml
conda activate ibd-quantum-esm
python -m src.cli hello
pytest -q
