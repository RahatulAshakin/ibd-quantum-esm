import subprocess
import sys
import duckdb
from pathlib import Path

def test_query_reads_duckdb(tmp_path: Path):
    db = tmp_path / "e.duckdb"
    con = duckdb.connect(str(db))
    con.execute("""
        create table embeddings(
          seq_id text, length integer, backend text, dim integer, embedding float[]
        );
    """)
    con.execute("insert into embeddings values ('foo1', 10, 'dummy', 64, [0.0,1.0,2.0])")
    con.execute("insert into embeddings values ('foo2', 12, 'dummy', 64, [3.0,4.0,5.0])")
    con.close()

    r = subprocess.run(
        [sys.executable, "-m", "src", "query", "--db", str(db), "--limit", "1", "--no-stats"],
        text=True,
        capture_output=True,
    )
    assert r.returncode == 0
    # Should show total rows and a preview with seq_id
    assert "rows=2" in r.stdout
    assert "seq_id=foo1" in r.stdout or "seq_id=foo2" in r.stdout
