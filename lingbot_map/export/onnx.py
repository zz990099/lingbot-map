import argparse
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn

from lingbot_map.models.gct_stream import GCTStream


DEFAULT_OUTPUTS = (
    "pose_enc",
    "depth",
    "depth_conf",
    "world_points",
    "world_points_conf",
    "cam_points",
    "cam_points_conf",
)


def _ensure_onnx_export_dependencies():
    missing = []
    for package_name in ("onnx", "onnxscript"):
        try:
            __import__(package_name)
        except ImportError:
            missing.append(package_name)
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"Missing ONNX export dependencies: {joined}. "
            f"Install them with `pip install {' '.join(missing)}`."
        )


class ONNXExportWrapper(nn.Module):
    def __init__(
        self,
        model: GCTStream,
        output_names: Sequence[str],
        num_frame_for_scale: int,
        num_frame_per_block: int,
        sliding_window_size: int | None = None,
    ):
        super().__init__()
        self.model = model
        self.output_names = list(output_names)
        self.num_frame_for_scale = num_frame_for_scale
        self.num_frame_per_block = num_frame_per_block
        self.sliding_window_size = sliding_window_size

    def forward(self, images: torch.Tensor):
        predictions = self.model(
            images,
            num_frame_for_scale=self.num_frame_for_scale,
            sliding_window_size=self.sliding_window_size,
            num_frame_per_block=self.num_frame_per_block,
            causal_inference=False,
        )
        return tuple(predictions[name] for name in self.output_names)


def _available_outputs(model: GCTStream) -> list[str]:
    outputs = []
    if model.camera_head is not None:
        outputs.append("pose_enc")
    if model.depth_head is not None:
        outputs.extend(["depth", "depth_conf"])
    if model.point_head is not None:
        outputs.extend(["world_points", "world_points_conf"])
    if model.local_point_head is not None:
        outputs.extend(["cam_points", "cam_points_conf"])
    return outputs


def _resolve_output_names(model: GCTStream, requested: Iterable[str] | None) -> list[str]:
    available = _available_outputs(model)
    if requested is None:
        return available
    requested = list(requested)
    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError(f"Requested outputs are not enabled on this model: {missing}")
    return requested


def _dynamic_axes_for_outputs(output_names: Sequence[str]) -> dict[str, dict[int, str]]:
    dynamic_axes: dict[str, dict[int, str]] = {
        "images": {0: "batch", 1: "frames", 3: "height", 4: "width"},
    }
    for name in output_names:
        if name == "pose_enc":
            dynamic_axes[name] = {0: "batch", 1: "frames"}
        else:
            dynamic_axes[name] = {0: "batch", 1: "frames", 3: "out_height", 4: "out_width"}
    return dynamic_axes


def build_model_for_onnx(
    *,
    model_path: str | None,
    image_size: int,
    patch_size: int,
    embed_dim: int,
    patch_embed: str,
    enable_camera: bool,
    enable_depth: bool,
    enable_point: bool,
    enable_local_point: bool,
    num_scale_frames: int,
    kv_cache_sliding_window: int,
    camera_num_iterations: int,
    device: torch.device,
) -> GCTStream:
    model = GCTStream(
        img_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        patch_embed=patch_embed,
        enable_camera=enable_camera,
        enable_depth=enable_depth,
        enable_point=enable_point,
        enable_local_point=enable_local_point,
        enable_3d_rope=False,
        enable_camera_3d_rope=False,
        num_frame_for_scale=num_scale_frames,
        kv_cache_scale_frames=num_scale_frames,
        kv_cache_sliding_window=kv_cache_sliding_window,
        use_sdpa=True,
        use_gradient_checkpoint=False,
        camera_num_iterations=camera_num_iterations,
    ).to(device)

    if model_path:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)
        model.load_state_dict(state_dict, strict=False)

    if hasattr(model.aggregator, "rope"):
        model.aggregator.rope = None
    if hasattr(model.aggregator, "position_getter"):
        model.aggregator.position_getter = None
    for block in getattr(model.aggregator, "frame_blocks", []):
        if hasattr(block, "attn") and hasattr(block.attn, "rope"):
            block.attn.rope = None
    for block in getattr(model.aggregator, "global_blocks", []):
        if hasattr(block, "attn") and hasattr(block.attn, "rope"):
            block.attn.rope = None

    model.eval()
    model.clean_kv_cache()
    model.set_export_mode(True)
    return model


def export_model_to_onnx(
    model: GCTStream,
    output_path: str | Path,
    sample_images: torch.Tensor,
    *,
    output_names: Sequence[str] | None = None,
    num_frame_for_scale: int = 1,
    num_frame_per_block: int = 1,
    sliding_window_size: int | None = None,
    opset_version: int = 18,
) -> list[str]:
    _ensure_onnx_export_dependencies()
    resolved_output_names = _resolve_output_names(model, output_names)
    wrapper = ONNXExportWrapper(
        model=model,
        output_names=resolved_output_names,
        num_frame_for_scale=num_frame_for_scale,
        num_frame_per_block=num_frame_per_block,
        sliding_window_size=sliding_window_size,
    ).eval()

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (sample_images,),
            str(output_path),
            export_params=True,
            do_constant_folding=True,
            opset_version=opset_version,
            dynamo=False,
            input_names=["images"],
            output_names=resolved_output_names,
            dynamic_axes=_dynamic_axes_for_outputs(resolved_output_names),
        )
    return resolved_output_names


def export_checkpoint_to_onnx(
    *,
    model_path: str | None,
    output_path: str | Path,
    image_size: int,
    patch_size: int,
    embed_dim: int,
    patch_embed: str,
    batch_size: int,
    num_frames: int,
    enable_camera: bool,
    enable_depth: bool,
    enable_point: bool,
    enable_local_point: bool,
    num_scale_frames: int,
    kv_cache_sliding_window: int,
    camera_num_iterations: int,
    opset_version: int,
    device: str,
) -> list[str]:
    torch_device = torch.device(device)
    model = build_model_for_onnx(
        model_path=model_path,
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        patch_embed=patch_embed,
        enable_camera=enable_camera,
        enable_depth=enable_depth,
        enable_point=enable_point,
        enable_local_point=enable_local_point,
        num_scale_frames=num_scale_frames,
        kv_cache_sliding_window=kv_cache_sliding_window,
        camera_num_iterations=camera_num_iterations,
        device=torch_device,
    )
    sample_images = torch.rand(
        batch_size,
        num_frames,
        3,
        image_size,
        image_size,
        device=torch_device,
    )
    try:
        return export_model_to_onnx(
            model,
            output_path,
            sample_images,
            num_frame_for_scale=min(num_scale_frames, num_frames),
            num_frame_per_block=min(num_scale_frames, num_frames),
            opset_version=opset_version,
        )
    finally:
        model.set_export_mode(False)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export LingBot-Map to ONNX.")
    parser.add_argument("--model_path", type=str, default=None, help="Checkpoint path. If omitted, exports random weights.")
    parser.add_argument("--output_path", type=str, required=True, help="Output ONNX file path.")
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--embed_dim", type=int, default=1024)
    parser.add_argument("--patch_embed", type=str, default="dinov2_vitl14_reg")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_frames", type=int, default=2)
    parser.add_argument("--num_scale_frames", type=int, default=1)
    parser.add_argument("--kv_cache_sliding_window", type=int, default=64)
    parser.add_argument("--camera_num_iterations", type=int, default=4)
    parser.add_argument("--opset_version", type=int, default=18)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--disable_camera", action="store_true")
    parser.add_argument("--disable_depth", action="store_true")
    parser.add_argument("--disable_point", action="store_true")
    parser.add_argument("--enable_local_point", action="store_true")
    return parser


def main():
    args = _build_arg_parser().parse_args()
    output_names = export_checkpoint_to_onnx(
        model_path=args.model_path,
        output_path=args.output_path,
        image_size=args.image_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        patch_embed=args.patch_embed,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        enable_camera=not args.disable_camera,
        enable_depth=not args.disable_depth,
        enable_point=not args.disable_point,
        enable_local_point=args.enable_local_point,
        num_scale_frames=args.num_scale_frames,
        kv_cache_sliding_window=args.kv_cache_sliding_window,
        camera_num_iterations=args.camera_num_iterations,
        opset_version=args.opset_version,
        device=args.device,
    )
    print(f"Exported outputs: {', '.join(output_names)}")
    print(f"Saved ONNX model to {args.output_path}")
