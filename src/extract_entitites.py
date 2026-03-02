# GLiNER step

import os
from gliner import GLiNER

NODE_TYPES = [
    "Protein",
    "Enzyme",
    "Molecule",
    "Cellular Process",
    "Organelle",
    "Cell Type",
    "Biological Pathway"
]

def extract_entities(text):
    model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

    entities = model.predict_entities(text, NODE_TYPES)

    unique_entities = list(set([e["text"] for e in entities]))

    return unique_entities