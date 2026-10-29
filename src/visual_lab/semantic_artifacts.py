"""Verify the fixed semantic editor's model/data/decoder contract."""

import hashlib


def validate_semantic_configuration(config, checkpoint_bytes, manifest_bytes):
    if (config.get("selected_on") != "validation"
            or config.get("test_model_performance_measured") is not False):
        raise ValueError("Semantic configuration must have been selected without test performance.")
    if (config.get("model") != "SemanticDilatedEditor" or config.get("architecture") != "dilated"
            or config.get("dataset") != "instruction_edits_v1"
            or config.get("decoder") != "three_class_argmax"
            or config.get("class_mapping") != {"background": 0, "circle": 1, "square": 2}):
        raise ValueError("Unsupported semantic model, dataset, class mapping, or decoder.")
    if hashlib.sha256(checkpoint_bytes).hexdigest() != config["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from the validation-selected semantic checkpoint.")
    if hashlib.sha256(manifest_bytes).hexdigest() != config["manifest_sha256"]:
        raise ValueError("Manifest differs from the selected semantic configuration.")
