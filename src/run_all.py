import subprocess

def run(cmd):
    print(f"\n==> {' '.join(cmd)}")
    subprocess.check_call(cmd)

def main():
    run(["python", "src/extract_entities.py"])
    run(["python", "src/extract_triples.py"])
    run(["python", "src/build_graph.py"])
    print("\nDone. Check outputs/")

if __name__ == "__main__":
    main()