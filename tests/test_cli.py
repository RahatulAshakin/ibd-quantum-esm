import subprocess
import sys


def run_cli(args):
    # Always use the current interpreter and the package entry: python -m src
    return subprocess.run(
        [sys.executable, "-m", "src"] + args,
        text=True,
        capture_output=True,
    )


def test_help_option():
    r = run_cli(["--help"])
    # Typer/Click prints help to stdout and exits 0
    assert r.returncode == 0
    assert ("Usage:" in r.stdout) or ("Commands" in r.stdout)


def test_unexpected_argument():
    # Ask for a non-existent command to force an error
    r = run_cli(["nope"])
    assert r.returncode != 0
    # Click error message
    assert "No such command" in r.stderr
