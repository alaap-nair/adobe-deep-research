"""
domain_schema.py -- Schema constraints for biology KG extraction (W3B).

Constrains the LLM extraction to a fixed vocabulary of node and relation types.
Triples whose head_type/tail_type/relation_type fall outside these sets are
dropped during validation, eliminating noisy extractions like raw numeric
measurements with no biological grounding.

Domains in scope: OpenStax Biology 2e Ch. 8 (Photosynthesis), Ch. 14
(DNA Structure & Function), Ch. 15 (Genes & Proteins), Ch. 24 (Fungi).
"""

from __future__ import annotations

# Allowed node types. Each LLM-extracted entity must declare one of these.
NODE_TYPES: dict[str, str] = {
    "Organism": "A whole organism: cyanobacteria, algae, fungi, plants, animals.",
    "Molecule": "A chemical compound or substrate: ATP, NADPH, CO2, H2O, glucose, RuBP, G3P, chlorophyll.",
    "Enzyme": "A biological catalyst: RuBisCO, ATP synthase.",
    "CellularStructure": (
        "A cellular component or sub-structure: thylakoid membrane, chloroplast, "
        "stroma, photosystem, hyphae, mycelium, cell wall."
    ),
    "Process": (
        "A biological process or activity: photosynthesis, glycolysis, "
        "carbon fixation, decomposition, respiration."
    ),
    "Pathway": "A named sequence of reactions: Calvin cycle, light-dependent reactions, citric acid cycle.",
    "TaxonomicGroup": "A taxonomic classification: Ascomycota, Basidiomycota, Glomeromycota, Chytridiomycota.",
    "EcologicalRole": "A functional ecological role: decomposer, mutualist, pathogen, parasite, producer.",
    "ChemicalReaction": "A specific reaction: photolysis of water, oxidation of NADH, electron transfer.",
    "Gene": (
        "A unit of heredity: a sequence of DNA that codes for a functional "
        "product (protein or RNA). Examples: a specific gene, an open reading frame."
    ),
    "GeneticElement": (
        "A nucleic-acid sequence element with a defined role: codon, anticodon, "
        "intron, exon, promoter, terminator, telomere, plasmid, chromosome, "
        "origin of replication, Okazaki fragment, primer."
    ),
}

# Allowed relation types (verbs). Each LLM-extracted triple must declare one.
RELATION_TYPES: dict[str, str] = {
    "CATALYZES": "An enzyme accelerates a reaction or process.",
    "PRODUCES": "A process or organism yields a molecule or product.",
    "CONSUMES": "A process or organism takes in a molecule as input.",
    "OCCURS_IN": "A process or reaction takes place in a cellular structure or location.",
    "PART_OF": "A structural or hierarchical containment relationship.",
    "DECOMPOSES": "An organism or process breaks down organic matter.",
    "MUTUALISTIC_WITH": "Two organisms maintain a mutually beneficial association.",
    "PARASITIC_ON": "An organism lives at the expense of a host.",
    "INHIBITS": "One entity suppresses or blocks another.",
    "LOCATED_IN": "A static location relationship (entity resides in structure).",
    "CONVERTS_TO": "A molecule or form is transformed into another.",
    "CONSISTS_OF": "An entity is composed of named parts or stages.",
    "BELONGS_TO": "Taxonomic membership: organism in taxonomic group.",
    "ABSORBS": "An entity takes up energy, light, or matter.",
    "RELEASES": "An entity emits energy, gas, or matter.",
    "REGULATES": "One entity modulates another's activity (positive or negative).",
    "ENCODES": (
        "A gene or DNA sequence specifies a protein, RNA, or trait "
        "(gene -> product)."
    ),
    "TRANSCRIBES_TO": (
        "DNA is copied into an RNA transcript (DNA template -> mRNA, tRNA, rRNA)."
    ),
    "TRANSLATES_TO": (
        "An mRNA codon sequence is decoded by a ribosome into a polypeptide "
        "(mRNA -> protein)."
    ),
    "REPLICATES": (
        "A DNA strand or chromosome is copied to produce a complementary daughter "
        "strand (DNA -> DNA copy)."
    ),
    "BINDS": (
        "One molecule physically attaches to another (e.g., enzyme to substrate, "
        "protein to DNA, ribosome to mRNA)."
    ),
    "PAIRS_WITH": (
        "Complementary base pairing between nucleotides (A-T in DNA, A-U in RNA, "
        "G-C in both)."
    ),
}


# Relation types treated as *functional* (single-valued): for a given head
# entity, at most one tail is currently true. When a new episode asserts a
# different tail for the same (head, relation_type), the older edge is taken to
# be contradicted and invalidated (phase-3 temporal conflict resolution).
#
# Kept deliberately conservative -- only relations where a second distinct tail
# is genuinely a contradiction rather than an additional true fact. e.g. a
# structure has one location (LOCATED_IN) and an organism one taxonomic group
# (BELONGS_TO), whereas a process legitimately PRODUCES many molecules.
# Overridable via env KG_FUNCTIONAL_RELATIONS (comma-separated).
import os as _os

FUNCTIONAL_RELATIONS: set[str] = {
    r.strip()
    for r in _os.getenv("KG_FUNCTIONAL_RELATIONS", "LOCATED_IN,BELONGS_TO").split(",")
    if r.strip()
}


def is_functional_relation(relation_type: str | None) -> bool:
    return relation_type in FUNCTIONAL_RELATIONS


def is_valid_node_type(name: str) -> bool:
    return name in NODE_TYPES


def is_valid_relation_type(name: str) -> bool:
    return name in RELATION_TYPES


def node_types_block() -> str:
    """Bullet list of allowed node types + descriptions, for the extraction prompt."""
    return "\n".join(f"- {k}: {v}" for k, v in NODE_TYPES.items())


def relation_types_block() -> str:
    """Bullet list of allowed relation types + descriptions, for the extraction prompt."""
    return "\n".join(f"- {k}: {v}" for k, v in RELATION_TYPES.items())


__all__ = [
    "NODE_TYPES",
    "RELATION_TYPES",
    "FUNCTIONAL_RELATIONS",
    "is_valid_node_type",
    "is_valid_relation_type",
    "is_functional_relation",
    "node_types_block",
    "relation_types_block",
]
