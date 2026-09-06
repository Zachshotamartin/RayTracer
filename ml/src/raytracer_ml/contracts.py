"""Versioned public JSON contracts; array semantics are validated separately."""

from jsonschema import Draft202012Validator

SHA = {"type": "string", "pattern": "^[a-f0-9]{64}$"}
IDENTIFIER = {"type": "string", "pattern": "^[a-zA-Z0-9_-]{1,128}$"}
EXAMPLE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "RayTracer training example v1",
    "type": "object",
    "required": [
        "schema_version",
        "id",
        "configuration",
        "group",
        "split",
        "cohort",
        "frame",
        "scene",
        "scene_sha256",
        "path",
        "sha256",
        "reference",
        "reference_sha256",
        "input_seed",
        "target_seed",
        "samples",
        "reference_samples",
        "scale",
        "stats",
        "reference_stats",
    ],
    "properties": {
        "schema_version": {"const": 1},
        **{k: IDENTIFIER for k in ["id", "configuration", "group", "cohort"]},
        "split": {"enum": ["train", "val", "test"]},
        **{k: SHA for k in ["scene_sha256", "sha256", "reference_sha256"]},
        **{k: {"type": "integer", "minimum": 0} for k in ["frame", "input_seed", "target_seed"]},
        **{k: {"type": "integer", "minimum": 1} for k in ["samples", "reference_samples"]},
        "scale": {"enum": [1, 2]},
        "scene": {"type": "object", "required": ["schema_version", "camera"]},
        **{k: {"type": "string", "minLength": 1} for k in ["path", "reference"]},
        **{k: {"type": "object"} for k in ["stats", "reference_stats"]},
    },
}
EXAMPLE_VALIDATOR = Draft202012Validator(EXAMPLE_SCHEMA)
