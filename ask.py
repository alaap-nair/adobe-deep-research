import os
import runpy
import sys


def main() -> None:
    # This workspace folder is separate from the actual codebase in ~/bio-graph.
    # We keep the CLI entrypoint here so `python ask.py "..."` works from this directory.
    workspace_dir = os.path.dirname(__file__)
    ivy_root = os.path.abspath(os.path.join(workspace_dir, "..", ".."))
    bio_graph_dir = os.path.join(ivy_root, "bio-graph")
    sys.path.insert(0, bio_graph_dir)
    runpy.run_path(os.path.join(bio_graph_dir, "ask.py"), run_name="__main__")


if __name__ == "__main__":
    main()

