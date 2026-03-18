"""
Output schema and validation for extracted triples and pipeline output.
Use to verify quality of pipeline outputs.
Supports both programmatic validation (ALLOWED_RELATIONS, REQUIRED_KEYS)
and JSON Schema validation (schema/triples_schema.json, schema/pipeline_output_schema.json).
"""

import json
import os

ALLOWED_RELATIONS = [
    "occurs_in",
    "produces",
    "converts_to",
    "uses",
    "requires",
    "inhibits",
    "activates",
    "transports_to",
    "donates_electrons_to",
    "accepts_electrons_from",
]

REQUIRED_KEYS = ("head", "relation", "tail", "evidence")

# Schema directory (project root / schema)
_SCHEMA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "schema")


def _load_json_schema(name):
    path = os.path.join(_SCHEMA_DIR, name)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_triples_with_schema(triples):
    """
    Validate that a list of triples conforms to schema/triples_schema.json.
    Returns (valid_triples, errors). If jsonschema is not installed or schema missing, uses validate_triples().
    """
    try:
        import jsonschema
    except ImportError:
        return validate_triples(triples)
    schema = _load_json_schema("triples_schema.json")
    if not schema:
        return validate_triples(triples)
    payload = {"triples": triples}
    try:
        jsonschema.validate(instance=payload, schema=schema)
        return triples, []
    except jsonschema.ValidationError as e:
        return [], [str(e)]


def validate_pipeline_output(entities, triples):
    """
    Validate full pipeline output (entities + triples) against schema/pipeline_output_schema.json.
    Returns (True, []) if valid, or (False, list of error messages).
    """
    try:
        import jsonschema
    except ImportError:
        return True, []
    schema = _load_json_schema("pipeline_output_schema.json")
    if not schema:
        return True, []
    payload = {"entities": list(entities) if entities else [], "triples": triples}
    try:
        jsonschema.validate(instance=payload, schema=schema)
        return True, []
    except jsonschema.ValidationError as e:
        return False, [str(e)]
    except jsonschema.SchemaError:
        return True, []


def validate_triple(triple, index=0):
    """Validate a single triple. Returns (is_valid, error_message or None)."""
    if not isinstance(triple, dict):
        return False, f"[{index}] triple is not a dict"
    for key in REQUIRED_KEYS:
        if key not in triple:
            return False, f"[{index}] missing key: {key}"
        if not isinstance(triple[key], str):
            return False, f"[{index}] {key} must be a string"
    if not (triple["head"] or triple["head"].strip()):
        return False, f"[{index}] head cannot be empty"
    if not (triple["tail"] or triple["tail"].strip()):
        return False, f"[{index}] tail cannot be empty"
    rel = (triple.get("relation") or "").strip()
    if rel not in ALLOWED_RELATIONS:
        return False, f"[{index}] relation '{rel}' not in allowed list"
    return True, None


def validate_triples(triples):
    """
    Validate a list of triples. Returns (valid_triples, errors).
    valid_triples: list of triples that pass validation.
    errors: list of error strings for invalid triples.
    """
    if not isinstance(triples, list):
        return [], [f"triples must be a list, got {type(triples).__name__}"]
    valid = []
    errors = []
    for i, t in enumerate(triples):
        ok, err = validate_triple(t, i)
        if ok:
            valid.append(t)
        else:
            errors.append(err)
    return valid, errors
