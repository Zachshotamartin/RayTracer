import json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch
from .models import build_model
from .train import load_checkpoint
from .preprocessing import channel_names, model_schema
from .io import write_json, digest


def export_model(checkpoint, output, width=None, height=None, precision="fp32"):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    state = load_checkpoint(checkpoint)
    model = build_model(state["config"]["model"]).cpu().eval()
    model.load_state_dict(state["model"])
    if (width is None) != (height is None) or (
        width is not None and (width < 1 or height < 1 or width * height > 16777216)
    ):
        raise ValueError("Provide both positive fixed dimensions within 16 megapixels")
    if precision not in ("fp32", "mixed-fp16"):
        raise ValueError("Unknown export precision")
    if precision == "mixed-fp16":
        if state["config"]["model"].get("kind") not in ("guided", "refine"):
            raise ValueError("Mixed precision requires a detail model with FP32 HDR arithmetic")
        for name in ("encoder", "head", "upscale", "history_gate"):
            if hasattr(model, name):
                getattr(model, name).half()
    torch.manual_seed(901)
    temporal = state["config"]["model"].get("temporal", False)
    schema = model_schema(state["config"]["model"])
    channels = channel_names(schema, temporal)
    x = torch.rand(1, len(channels), height or 36, width or 64)
    x[:, 15] = 4
    x[:, 16] = 1
    x[:, 10:12] = 1
    if schema == 2:
        x[:, 26] = 1
    torch.onnx.export(
        model,
        x,
        str(output),
        input_names=["features"],
        output_names=["radiance"],
        dynamic_axes=None
        if width is not None
        else {
            "features": {2: "height", 3: "width"},
            "radiance": {2: "out_height", 3: "out_width"},
        },
        opset_version=18,
        dynamo=False,
    )
    graph = onnx.load(str(output))
    for key, value in {
        "rt_schema": str(schema),
        "rt_channels": json.dumps(channels),
        "rt_temporal": "1" if temporal else "0",
        "rt_scale": str(state["config"]["model"].get("scale", 1)),
        "rt_domain": "diffuse-pinhole",
        "rt_checkpoint_sha256": digest(checkpoint),
        "rt_precision": precision,
        "rt_tile_halo": "32"
        if state["config"]["model"].get("kind") in ("guided", "refine") and width is None
        else "0",
    }.items():
        entry = graph.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.checker.check_model(graph)
    onnx.save(graph, str(output))
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    errors = []
    sizes = [(height, width)] if width is not None else [(36, 64), (35, 61), (72, 128)]
    for h, w in sizes:
        x = torch.rand(1, len(channels), h, w)
        x[:, 15] = 8
        x[:, 16] = 1
        x[:, 10:12] = 1
        if schema == 2:
            x[:, 26] = 1
        with torch.inference_mode():
            expected = model(x).numpy()
        actual = session.run(None, {"features": x.numpy()})[0]
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=2e-3 if precision == "mixed-fp16" else 2e-4,
            atol=2e-4 if precision == "mixed-fp16" else 2e-5,
        )
        errors.append(float(np.max(np.abs(actual - expected))))
    metadata = {
        "schema_version": schema,
        "precision": precision,
        "fixed_shape": [height, width] if width is not None else None,
        "channels": channels,
        "model": state["config"]["model"],
        "sha256": digest(output),
        "checkpoint_sha256": digest(checkpoint),
        "manifest_sha256": state["manifest_sha256"],
        "parity_max_absolute_error": max(errors),
        "domain": "diffuse pinhole scenes; unsupported primary pixels preserve raw RGB",
        "providers_tested": session.get_providers(),
        "onnxruntime": ort.__version__,
    }
    write_json(output.with_suffix(".json"), metadata)
    return metadata
