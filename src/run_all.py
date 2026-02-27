# src/run_all.py
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]  # repo root

def run(relpath: str):
    script = ROOT / relpath
    print(f"\n==> {sys.executable} {script}")
    subprocess.check_call([sys.executable, str(script)])

def main():
    run("src/extract_entities.py")
    run("src/extract_triples.py")
    run("src/build_graph.py")
    print("\nDone. Check outputs/")

if __name__ == "__main__":
    main()