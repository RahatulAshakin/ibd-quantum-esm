# scripts/prep_uniprot_ec.py
from __future__ import annotations
import argparse, re, random, os
from pathlib import Path
from typing import Dict, Tuple, List
import pandas as pd

def parse_args():
    ap = argparse.ArgumentParser(description="Prepare UniProt EC dataset: labels + stratified split.")
    ap.add_argument("--fasta", default="data/uniprot_enzymes.fa", help="Concatenated FASTA")
    ap.add_argument("--map",   default="data/accession_ec.tsv",   help="UniProt accession→EC mapping (TSV)")
    ap.add_argument("--per-class", type=int, default=300, help="Samples per EC top class (1..6)")
    ap.add_argument("--train-frac", type=float, default=0.8, help="Train fraction per class")
    ap.add_argument("--outdir", default="data", help="Output directory for split FASTAs and labels")
    return ap.parse_args()

def get_accession_from_header(hdr: str) -> str:
    # Handles headers like:
    # >sp|A0A061I403|FICD_CRIGR ...
    # >tr|Q9XYZ1|NAME_SPEC ...
    # >P12345 some desc ...
    hdr = hdr.strip()
    if hdr.startswith(">"):
        hdr = hdr[1:]
    parts = hdr.split("|")
    if len(parts) >= 2 and (parts[0] in ("sp", "tr")):
        return parts[1]
    # fallback: first token
    return hdr.split()[0]

def read_fasta_pairs(path: str) -> List[Tuple[str, str]]:
    pairs = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        seq_id = None
        buf = []
        for line in f:
            if line.startswith(">"):
                if seq_id is not None and buf:
                    pairs.append((seq_id, "".join(buf).replace(" ", "").upper()))
                seq_id = get_accession_from_header(line.strip())
                buf = []
            else:
                buf.append(re.sub(r"[^A-Za-z]", "", line))
        if seq_id is not None and buf:
            pairs.append((seq_id, "".join(buf).replace(" ", "").upper()))
    return pairs

def top_ec_class(ec_field: str) -> int | None:
    # ec_field may contain multiple ECs separated by "; "
    if not isinstance(ec_field, str) or not ec_field:
        return None
    first = ec_field.split(";")[0].strip()
    m = re.match(r"^(\d)\.", first)
    return int(m.group(1)) if m else None

def main():
    args = parse_args()
    random.seed(1337)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load mapping
    df = pd.read_csv(args.map, sep="\t")
    cols = {c.lower(): c for c in df.columns}
    # UniProt fields are usually "Entry" and "EC number"
    acc_col = cols.get("accession") or cols.get("entry") or list(df.columns)[0]
    ec_col  = cols.get("ec") or cols.get("ec number") or list(df.columns)[1]
    df = df[[acc_col, ec_col]].rename(columns={acc_col: "accession", ec_col: "ec_raw"})
    df["ec_top"] = df["ec_raw"].map(top_ec_class)
    df = df.dropna(subset=["ec_top"])
    df["ec_top"] = df["ec_top"].astype(int)
    df = df[df["ec_top"].between(1, 6)]

    ec_map: Dict[str, int] = dict(zip(df["accession"], df["ec_top"]))

    # Read FASTA and keep those with labels
    pairs = read_fasta_pairs(args.fasta)
    keep: List[Tuple[str, str, int]] = []
    for acc, seq in pairs:
        ec = ec_map.get(acc)
        if ec is not None and 50 <= len(seq) <= 512:
            keep.append((acc, seq, ec))

    # Balance per class
    by_class: Dict[int, List[Tuple[str, str, int]]] = {c: [] for c in range(1, 7)}
    for acc, seq, ec in keep:
        by_class[ec].append((acc, seq, ec))
    for c in by_class:
        random.shuffle(by_class[c])

    selected: List[Tuple[str, str, int]] = []
    for c in range(1, 7):
        bucket = by_class[c]
        take = min(len(bucket), args.per_class)
        selected.extend(bucket[:take])

    # Stratified split
    train, heldout = [], []
    for c in range(1, 7):
        bucket = [(a, s, e) for (a, s, e) in selected if e == c]
        n = len(bucket)
        k = max(1, int(round(n * args.train_frac)))
        train.extend(bucket[:k])
        heldout.extend(bucket[k:])

    # Write FASTAs
    def write_fa(path: Path, items: List[Tuple[str, str, int]]):
        with open(path, "w", encoding="utf-8") as w:
            for acc, seq, ec in items:
                w.write(f">{acc}\n{seq}\n")

    write_fa(outdir / "train.fa", train)
    write_fa(outdir / "heldout.fa", heldout)

    # Write labels
    df_train = pd.DataFrame([{"seq_id": a, "ec_top": e, "length": len(s)} for a, s, e in train])
    df_held  = pd.DataFrame([{"seq_id": a, "ec_top": e, "length": len(s)} for a, s, e in heldout])
    df_train.to_csv(outdir / "labels_train.csv", index=False)
    df_held.to_csv(outdir / "labels_heldout.csv", index=False)

    # Small summary
    def counts(df_):
        return df_.groupby("ec_top").size().rename("n").reset_index()

    print("Prepared dataset")
    print(" per-class train counts:\n", counts(df_train).to_string(index=False))
    print(" per-class heldout counts:\n", counts(df_held).to_string(index=False))
    print(f"Wrote: {outdir/'train.fa'}, {outdir/'heldout.fa'}")
    print(f"Wrote: {outdir/'labels_train.csv'}, {outdir/'labels_heldout.csv'}")

if __name__ == "__main__":
    main()
