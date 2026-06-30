"""
bio_schema.py -- Domain schema for Molecular Biology & Genetics.

Defines allowed node types and relationship types to constrain
LLM extraction and reduce noise in the knowledge graph.
"""

# ── Allowed Node Types ───────────────────────────────────────────────────────
# Each key is the canonical type label; values are examples for prompt context.

NODE_TYPES = {
    "Gene": "e.g., BRCA1, p53, lac operon, rpoB",
    "Protein": "e.g., hemoglobin, insulin, RNA polymerase, ribosome, enzyme",
    "Molecule": "e.g., ATP, glucose, mRNA, tRNA, DNA, amino acid, nucleotide",
    "CellStructure": "e.g., nucleus, ribosome, mitochondria, endoplasmic reticulum, membrane",
    "BiologicalProcess": "e.g., transcription, translation, glycolysis, DNA replication, gene expression",
    "Pathway": "e.g., Calvin cycle, electron transport chain, signal transduction",
    "Organism": "e.g., E. coli, Drosophila, human, bacteriophage T2",
    "Disease": "e.g., sickle cell anemia, cancer, cystic fibrosis",
    "Technique": "e.g., PCR, gel electrophoresis, CRISPR, DNA sequencing, cloning",
    "GenomicElement": "e.g., promoter, enhancer, codon, intron, exon, telomere, plasmid",
}

# ── Allowed Relationship Types ───────────────────────────────────────────────
# Short verb phrases the LLM should use as relation labels.

RELATION_TYPES = [
    "encodes",
    "is encoded by",
    "transcribes",
    "is transcribed from",
    "translates",
    "is translated into",
    "regulates",
    "inhibits",
    "activates",
    "catalyzes",
    "binds to",
    "produces",
    "is produced by",
    "degrades",
    "is component of",
    "contains",
    "interacts with",
    "is involved in",
    "causes",
    "prevents",
    "mutates",
    "repairs",
    "replicates",
    "modifies",
    "transports",
    "is located in",
    "is type of",
    "requires",
    "uses",
    "converts",
]


def build_schema_prompt_block() -> str:
    """Format the schema as a prompt block for the extraction LLM."""
    lines = [
        "DOMAIN SCHEMA — use these types to categorize entities and relationships.\n",
        "Allowed Node Types:",
    ]
    for node_type, examples in NODE_TYPES.items():
        lines.append(f"  - {node_type}: {examples}")

    lines.append("\nAllowed Relationship Types:")
    lines.append(f"  {', '.join(RELATION_TYPES)}")

    lines.append(
        "\nRules:"
        "\n  - Classify each head and tail entity as one of the Node Types above."
        "\n  - Use a relationship from the Allowed list when possible."
        "\n  - If no allowed relationship fits, use a short verb phrase (1-3 words)."
        "\n  - Do NOT extract raw numbers, figure references, or generic phrases as entities."
        "\n  - Each entity should be a specific biological concept, not a sentence fragment."
    )

    return "\n".join(lines)


def get_node_type_list() -> list[str]:
    """Return the list of allowed node type names."""
    return list(NODE_TYPES.keys())


def get_relation_type_list() -> list[str]:
    """Return the list of allowed relation types."""
    return list(RELATION_TYPES)


if __name__ == "__main__":
    print(build_schema_prompt_block())
