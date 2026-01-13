#!/usr/bin/env python3
"""Export Docling models to ONNX format for gamma-crown verification.

Models that need export:
- DocumentFigureClassifier (4.07M) - EfficientNet-based classifier
- docling-layout-egret-large (31.2M) - RT-DETR layout detector
- docling-layout-egret-xlarge (62.7M) - RT-DETR layout detector
- granite-docling-258M (258M) - VLM (SigLIP encoder + Granite LLM)
- CodeFormulaV2 (300M) - VLM for formula recognition

SmolDocling-256M already has ONNX exports.

Usage:
    python scripts/export_docling_to_onnx.py --model DocumentFigureClassifier
    python scripts/export_docling_to_onnx.py --all
"""

import argparse
import os
import sys
from pathlib import Path

# Check for required packages
try:
    import torch
    import onnx
except ImportError:
    print("Installing required packages...")
    os.system("pip install torch onnx transformers optimum[exporters]")
    import torch
    import onnx

def export_document_figure_classifier():
    """Export DocumentFigureClassifier (EfficientNet-based)."""
    from transformers import AutoModelForImageClassification, AutoImageProcessor

    model_path = "models/docling/DocumentFigureClassifier"
    output_path = f"{model_path}/model.onnx"

    print(f"Exporting DocumentFigureClassifier to {output_path}")

    # Load model
    model = AutoModelForImageClassification.from_pretrained(model_path)
    processor = AutoImageProcessor.from_pretrained(model_path)

    model.eval()

    # Create dummy input (3 channel image)
    dummy_input = torch.randn(1, 3, 224, 224)

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "logits": {0: "batch_size"}
        },
        opset_version=14
    )

    # Verify
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"Exported and verified: {output_path}")

def export_layout_model(model_name: str):
    """Export RT-DETR layout detection model."""
    from transformers import AutoModelForObjectDetection, AutoImageProcessor

    model_path = f"models/docling/{model_name}"
    output_path = f"{model_path}/model.onnx"

    print(f"Exporting {model_name} to {output_path}")

    # Load model
    model = AutoModelForObjectDetection.from_pretrained(model_path)
    processor = AutoImageProcessor.from_pretrained(model_path)

    model.eval()

    # RT-DETR expects specific input size
    dummy_input = torch.randn(1, 3, 640, 640)

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["pixel_values"],
        output_names=["logits", "pred_boxes"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "logits": {0: "batch_size"},
            "pred_boxes": {0: "batch_size"}
        },
        opset_version=14
    )

    print(f"Exported: {output_path}")

def export_vlm_encoder(model_name: str):
    """Export VLM vision encoder only (full VLM export is complex)."""
    from transformers import AutoModel, AutoProcessor

    model_path = f"models/docling/{model_name}"
    output_path = f"{model_path}/vision_encoder.onnx"

    print(f"Exporting {model_name} vision encoder to {output_path}")

    try:
        # Try to load as VLM and extract vision encoder
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

        if hasattr(model, 'vision_tower') or hasattr(model, 'vision_encoder'):
            vision_encoder = getattr(model, 'vision_tower', None) or model.vision_encoder
            vision_encoder.eval()

            # SigLIP expects 384x384 or 512x512
            dummy_input = torch.randn(1, 3, 512, 512)

            torch.onnx.export(
                vision_encoder,
                dummy_input,
                output_path,
                input_names=["pixel_values"],
                output_names=["vision_features"],
                dynamic_axes={
                    "pixel_values": {0: "batch_size"},
                    "vision_features": {0: "batch_size"}
                },
                opset_version=14
            )
            print(f"Exported vision encoder: {output_path}")
        else:
            print(f"Could not find vision encoder in {model_name}")

    except Exception as e:
        print(f"Error exporting {model_name}: {e}")
        print("VLM export may require optimum-cli or manual export")

def export_with_optimum(model_name: str):
    """Use optimum-cli for complex model export."""
    model_path = f"models/docling/{model_name}"
    output_path = f"{model_path}/onnx"

    print(f"Exporting {model_name} with optimum-cli...")

    cmd = f"optimum-cli export onnx --model {model_path} {output_path}"
    result = os.system(cmd)

    if result == 0:
        print(f"Exported to: {output_path}")
    else:
        print(f"Export failed for {model_name}")

def main():
    parser = argparse.ArgumentParser(description="Export Docling models to ONNX")
    parser.add_argument("--model", type=str, help="Model name to export")
    parser.add_argument("--all", action="store_true", help="Export all models")
    parser.add_argument("--optimum", action="store_true", help="Use optimum-cli for export")
    args = parser.parse_args()

    if args.all:
        models = [
            "DocumentFigureClassifier",
            "docling-layout-egret-large",
            "docling-layout-egret-xlarge",
            "granite-docling-258M",
            "CodeFormulaV2"
        ]
    elif args.model:
        models = [args.model]
    else:
        parser.print_help()
        return

    for model in models:
        print(f"\n{'='*60}")
        print(f"Processing: {model}")
        print('='*60)

        if args.optimum:
            export_with_optimum(model)
        elif model == "DocumentFigureClassifier":
            export_document_figure_classifier()
        elif model.startswith("docling-layout"):
            export_layout_model(model)
        elif model in ["granite-docling-258M", "CodeFormulaV2"]:
            export_vlm_encoder(model)
        else:
            print(f"Unknown model: {model}")

if __name__ == "__main__":
    main()
