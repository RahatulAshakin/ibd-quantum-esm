# src/ibdqlib/esm_embed.py
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Tuple
import numpy as np

def read_fasta(path: str) -> List[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    cur_id = None
    cur_seq = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    pairs.append((cur_id, "".join(cur_seq)))
                cur_id = line[1:].split()[0]
                cur_seq = []
            else:
                cur_seq.append(line)
    if cur_id is not None:
        pairs.append((cur_id, "".join(cur_seq)))
    return pairs

# ------------------------
# Embedders
# ------------------------

def _dummy_embedder() -> tuple[Callable[[List[str], int], np.ndarray], int, str]:
    dim = 64
    aa = "ACDEFGHIKLMNPQRSTVWY"

    def _embed(seqs: List[str], batch: int = 64) -> np.ndarray:
        out = []
        for s in seqs:
            counts = np.array([s.count(c) for c in aa], dtype=np.float32)
            # L2-normalize; if empty sequence, keep zeros
            if counts.sum() > 0:
                counts = counts / np.linalg.norm(counts)
            # tile to 64 dims
            v = np.tile(counts, int(np.ceil(dim / len(aa))))[:dim].astype(np.float32)
            out.append(v)
        return np.vstack(out)

    return _embed, dim, "dummy"

def _esm2_embedder(model_name: str) -> tuple[Callable[[List[str], int], np.ndarray], int, str]:
    import torch
    import esm  # fair-esm

    model_name_lc = model_name.lower()
    if model_name_lc == "esm2_t6_8m_ur50d":
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        dim = 320
    else:
        raise ValueError(f"Unsupported ESM model '{model_name}'. Try 'esm2_t6_8M_UR50D'.")

    model.eval()
    model.to("cpu")
    batch_converter = alphabet.get_batch_converter()
    cls_idx, eos_idx, pad_idx = alphabet.cls_idx, alphabet.eos_idx, alphabet.padding_idx

    def _embed(seqs: List[str], batch: int = 64) -> np.ndarray:
        vecs = []
        with torch.no_grad():
            for i in range(0, len(seqs), batch):
                chunk = [("seq", s) for s in seqs[i : i + batch]]
                _, _, toks = batch_converter(chunk)
                toks = toks.to("cpu")
                out = model(toks, repr_layers=[model.num_layers], return_contacts=False)
                reps = out["representations"][model.num_layers]
                for j in range(reps.size(0)):
                    rep_j = reps[j]
                    tok_j = toks[j]
                    mask = (tok_j != pad_idx) & (tok_j != cls_idx) & (tok_j != eos_idx)
                    v = rep_j[mask].mean(0).cpu().numpy().astype(np.float32)
                    vecs.append(v)
        return np.vstack(vecs)

    return _embed, dim, "esm2_t6_8M_UR50D"

def get_embedder(backend: str):
    """
    Return (embed_fn, dim, backend_used).
    backend: "dummy" or "esm2_t6_8M_UR50D"
    """
    b = (backend or "dummy").lower()
    if b == "dummy":
        return _dummy_embedder()
    if b.startswith("esm2"):
        return _esm2_embedder(b)
    raise ValueError(f"Unknown backend '{backend}'. Use 'dummy' or 'esm2_t6_8M_UR50D'.")
