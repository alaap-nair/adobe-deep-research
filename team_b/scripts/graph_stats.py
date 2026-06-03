"""
graph_stats.py -- Quick KG health metrics. Used for before/after dedup comparison.

Usage:
    python scripts/graph_stats.py
    python scripts/graph_stats.py --json
"""

import argparse
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from neo4j import GraphDatabase

from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


def connect():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def gather(driver) -> dict:
    out = {}
    with driver.session() as session:
        out["nodes"] = session.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
        out["relationships"] = session.run(
            "MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS c"
        ).single()["c"]

        # Average degree
        if out["nodes"]:
            avg_deg = session.run(
                "MATCH (n:Entity) "
                "OPTIONAL MATCH (n)-[r:RELATES_TO]-() "
                "WITH n, count(r) AS deg "
                "RETURN avg(deg) AS d"
            ).single()
            out["avg_degree"] = float(avg_deg["d"] or 0.0) if avg_deg else 0.0
        else:
            out["avg_degree"] = 0.0

        # Top 10 entities by surface-form count (from original_names)
        records = session.run(
            "MATCH (n:Entity) "
            "RETURN n.name AS name, size(coalesce(n.original_names, [])) AS surface_forms "
            "ORDER BY surface_forms DESC, name ASC LIMIT 10"
        )
        out["top_surface_forms"] = [
            {"name": r["name"], "surface_forms": r["surface_forms"]} for r in records
        ]

        # Distinct relation labels (the open-vocabulary `relation` property)
        rels = session.run(
            "MATCH ()-[r:RELATES_TO]->() RETURN r.relation AS rel"
        )
        rel_counts: Counter = Counter()
        for record in rels:
            rel_counts[record["rel"]] += 1
        out["distinct_relations"] = len(rel_counts)
        out["top_relations"] = rel_counts.most_common(10)

        # If schema labels exist (W3B), show node-type breakdown.
        labels = session.run(
            "CALL db.labels() YIELD label RETURN label"
        )
        node_type_labels = [r["label"] for r in labels if r["label"] != "Entity"]
        out["node_type_labels"] = node_type_labels
        if node_type_labels:
            type_counts = {}
            for label in node_type_labels:
                count = session.run(
                    f"MATCH (n:Entity:`{label}`) RETURN count(n) AS c"
                ).single()["c"]
                type_counts[label] = count
            out["nodes_by_type"] = dict(
                sorted(type_counts.items(), key=lambda kv: -kv[1])
            )
    return out


def render(stats: dict) -> str:
    lines = []
    lines.append(f"Nodes:                 {stats['nodes']}")
    lines.append(f"Relationships:         {stats['relationships']}")
    lines.append(f"Avg degree:            {stats['avg_degree']:.2f}")
    lines.append(f"Distinct relations:    {stats['distinct_relations']}")
    lines.append("")
    lines.append("Top entities by surface-form count:")
    for row in stats["top_surface_forms"]:
        lines.append(f"  {row['surface_forms']:>3}  {row['name']}")
    lines.append("")
    lines.append("Top 10 relation labels:")
    for rel, count in stats["top_relations"]:
        lines.append(f"  {count:>4}  {rel}")
    if stats.get("node_type_labels"):
        lines.append("")
        lines.append("Nodes by schema type (W3B labels):")
        for label, count in stats.get("nodes_by_type", {}).items():
            lines.append(f"  {count:>4}  {label}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    args = parser.parse_args()

    driver = connect()
    try:
        stats = gather(driver)
    finally:
        driver.close()

    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print(render(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
