# src/ibdqlib/esmatlas.py
from __future__ import annotations
import re
from pathlib import Path
from typing import Tuple
import requests

# Standard 20 aa + ambiguous codes often accepted by ESM services
ALLOWED = set("ACDEFGHIKLMNPQRSTVWYXBZJ")

def _clean(seq: str) -> tuple[str, list[str]]:
    seq = re.sub(r"[^A-Za-z]", "", seq).upper()
    bad = sorted({ch for ch in seq if ch not in ALLOWED})
    cleaned = "".join(ch if ch in ALLOWED else "X" for ch in seq)
    return cleaned, bad

def _post_raw_text(url: str, seq: str, timeout: int):
    # Raw body, text/plain is what many examples use
    headers = {
        "Content-Type": "text/plain; charset=utf-8",
        "Accept": "chemical/x-pdb, text/plain;q=0.9, */*;q=0.1",
    }
    return requests.post(url, data=seq.encode("utf-8"), headers=headers, timeout=timeout)

def _post_json(url: str, key: str, seq: str, timeout: int):
    headers = {"Accept": "chemical/x-pdb, application/json;q=0.9, */*;q=0.1"}
    return requests.post(url, json={key: seq}, headers=headers, timeout=timeout)

def _post_form(url: str, key: str, seq: str, timeout: int):
    headers = {"Accept": "chemical/x-pdb, text/plain;q=0.9, */*;q=0.1"}
    return requests.post(url, data={key: seq}, headers=headers, timeout=timeout)

def fold_sequence(seq_id: str, seq: str, outdir: str, sanitize: bool = False, timeout: int = 60) -> Tuple[bool, str | None]:
    """
    Call ESM Atlas API and write <outdir>/<seq_id>.pdb on success.
    Returns (ok, message). On ok=True, message may include a note about sanitization.
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Clean/sanitize
    raw = re.sub(r"[^A-Za-z]", "", seq).upper()
    if sanitize:
        seq_use, bad = _clean(raw)
        note = f" (replaced {''.join(bad)} -> 'X')" if bad else ""
    else:
        bad = sorted({ch for ch in raw if ch not in ALLOWED})
        if bad:
            return False, f"Invalid tokens: {bad}. Try --sanitize."
        seq_use, note = raw, ""

    urls = [
        "https://api.esmatlas.com/foldSequence/v1/pdb",
        "https://api.esmatlas.com/foldSequence/v1/pdb/",
    ]

    attempts = []
    for url in urls:
        # 1) raw text
        try:
            r = _post_raw_text(url, seq_use, timeout)
            if r.status_code == 200 and r.text and len(r.text) > 50:
                out_path = Path(outdir) / f"{seq_id}.pdb"
                out_path.write_text(r.text, encoding="utf-8")
                return True, f"wrote {out_path.name}{note}".strip()
            else:
                # Try to read JSON error; otherwise show short body
                try:
                    errj = r.json()
                    msg = errj.get("detail") or errj.get("message") or str(errj)
                except Exception:
                    body = (r.text or "").strip()
                    snippet = (body[:120] if body else "").replace("\n", " ")
                    msg = f"HTTP {r.status_code}: {snippet or 'empty response'}"
                attempts.append(f"POST {url} [raw text] -> {msg}")
        except Exception as e:
            attempts.append(f"POST {url} [raw text] -> network error: {e}")

        # 2) JSON: {"sequence": "..."}
        try:
            r = _post_json(url, "sequence", seq_use, timeout)
            if r.status_code == 200 and r.text and len(r.text) > 50:
                out_path = Path(outdir) / f"{seq_id}.pdb"
                out_path.write_text(r.text, encoding="utf-8")
                return True, f"wrote {out_path.name}{note}".strip()
            else:
                try:
                    errj = r.json()
                    msg = errj.get("detail") or errj.get("message") or str(errj)
                except Exception:
                    body = (r.text or "").strip()
                    snippet = (body[:120] if body else "").replace("\n", " ")
                    msg = f"HTTP {r.status_code}: {snippet or 'empty response'}"
                attempts.append(f"POST {url} [json:sequence] -> {msg}")
        except Exception as e:
            attempts.append(f"POST {url} [json:sequence] -> network error: {e}")

        # 3) JSON: {"seq": "..."}
        try:
            r = _post_json(url, "seq", seq_use, timeout)
            if r.status_code == 200 and r.text and len(r.text) > 50:
                out_path = Path(outdir) / f"{seq_id}.pdb"
                out_path.write_text(r.text, encoding="utf-8")
                return True, f"wrote {out_path.name}{note}".strip()
            else:
                try:
                    errj = r.json()
                    msg = errj.get("detail") or errj.get("message") or str(errj)
                except Exception:
                    body = (r.text or "").strip()
                    snippet = (body[:120] if body else "").replace("\n", " ")
                    msg = f"HTTP {r.status_code}: {snippet or 'empty response'}"
                attempts.append(f"POST {url} [json:seq] -> {msg}")
        except Exception as e:
            attempts.append(f"POST {url} [json:seq] -> network error: {e}")

        # 4) Form: sequence=...
        try:
            r = _post_form(url, "sequence", seq_use, timeout)
            if r.status_code == 200 and r.text and len(r.text) > 50:
                out_path = Path(outdir) / f"{seq_id}.pdb"
                out_path.write_text(r.text, encoding="utf-8")
                return True, f"wrote {out_path.name}{note}".strip()
            else:
                try:
                    errj = r.json()
                    msg = errj.get("detail") or errj.get("message") or str(errj)
                except Exception:
                    body = (r.text or "").strip()
                    snippet = (body[:120] if body else "").replace("\n", " ")
                    msg = f"HTTP {r.status_code}: {snippet or 'empty response'}"
                attempts.append(f"POST {url} [form:sequence] -> {msg}")
        except Exception as e:
            attempts.append(f"POST {url} [form:sequence] -> network error: {e}")

    # If all attempts failed:
    return False, "; ".join(attempts)[:800]
