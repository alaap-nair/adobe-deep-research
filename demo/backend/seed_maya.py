"""
seed_maya.py -- Pre-ingest Maya's three episodes into live Graphiti.

Run ONCE before the presentation. Each episode is a real Graphiti `add_episode`
call under group_id="maya": Graphiti runs the LLM to extract entities + edges and
links them to Maya's personal subgraph in Neo4j. You can inspect the result in the
Neo4j Browser (http://localhost:7474) with:

    MATCH (n {group_id: "maya"}) RETURN n LIMIT 100

Usage:
    python -m demo.backend.seed_maya          # ingest all three episodes
    python -m demo.backend.seed_maya --clear  # wipe Maya's group first, then ingest
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone

from graphiti_core.nodes import EpisodeType

from demo.backend.graphiti_setup import GROUP_ID, build_graphiti

# Episode bodies mirror demo/web/public/snapshot.json so the live run and the
# frozen replay tell the same story. Text is grounded in OpenStax Biology Ch.8.
EPISODES = [
    {
        "name": "Maya — Calvin cycle question",
        "reference_time": datetime(2026, 6, 14, 9, 12, tzinfo=timezone.utc),
        "source": EpisodeType.text,
        "source_description": "Maya's query",
        "body": (
            "How does the Calvin cycle fix carbon? In the stroma, RuBisCO catalyzes a "
            "reaction between CO2 and RuBP, producing 3-PGA, which the Calvin cycle "
            "reduces to G3P, a sugar precursor."
        ),
    },
    {
        "name": "Maya — light reactions lecture notes",
        "reference_time": datetime(2026, 6, 14, 15, 47, tzinfo=timezone.utc),
        "source": EpisodeType.text,
        "source_description": "Maya's uploaded lecture notes (light_reactions.md)",
        "body": (
            "Light-dependent reactions: pigments such as chlorophyll a in the thylakoid "
            "membrane absorb light. Photosystem II splits water, releasing oxygen and "
            "electrons. Electrons travel down the electron transport chain to photosystem I, "
            "which reduces NADP to NADPH. The proton gradient drives ATP synthase to make "
            "ATP. Net output: ATP and NADPH (plus O2 as a byproduct)."
        ),
    },
    {
        "name": "Maya — linking the two stages",
        "reference_time": datetime(2026, 6, 15, 11, 3, tzinfo=timezone.utc),
        "source": EpisodeType.text,
        "source_description": "Maya's query",
        "body": (
            "What links the light reactions to the Calvin cycle? The ATP and NADPH produced "
            "by the light reactions power the Calvin cycle: ATP and NADPH reduce 3-PGA to G3P."
        ),
    },
]


async def main(clear: bool) -> None:
    graphiti = build_graphiti()
    try:
        await graphiti.build_indices_and_constraints()

        if clear:
            print(f"Clearing existing group_id='{GROUP_ID}' ...")
            # Remove only Maya's partition; leave the rest of the graph intact.
            await graphiti.driver.execute_query(
                "MATCH (n {group_id: $gid}) DETACH DELETE n", gid=GROUP_ID
            )

        for i, ep in enumerate(EPISODES, start=1):
            print(f"[{i}/{len(EPISODES)}] Ingesting: {ep['name']} ...")
            await graphiti.add_episode(
                name=ep["name"],
                episode_body=ep["body"],
                source=ep["source"],
                source_description=ep["source_description"],
                reference_time=ep["reference_time"],
                group_id=GROUP_ID,
            )
        print(
            f"\nDone. Inspect Maya's subgraph in Neo4j Browser:\n"
            f'  MATCH (n {{group_id: "{GROUP_ID}"}}) RETURN n LIMIT 100'
        )
    finally:
        await graphiti.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed Maya's Graphiti episodes.")
    parser.add_argument("--clear", action="store_true", help="Wipe Maya's group first.")
    args = parser.parse_args()
    asyncio.run(main(args.clear))
