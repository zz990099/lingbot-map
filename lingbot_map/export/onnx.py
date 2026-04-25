import argparse
import json
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


def _default_selected_feature_groups(num_groups: int) -> list[int]:
    if num_groups != 24:
        return sorted({max(0, round((num_groups - 1) * ratio)) for ratio in (0.2, 0.48, 0.74, 1.0)})
    return [4, 11, 17, 23]


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


class PatchEmbedExportWrapper(nn.Module):
    def __init__(self, model: GCTStream, num_frame_for_scale: int):
        super().__init__()
        self.aggregator = model.aggregator
        self.num_frame_for_scale = num_frame_for_scale

    def forward(self, images: torch.Tensor):
        tokens, batch_size, _, seq_len, tokens_per_frame, channels = self.aggregator._embed_images(
            images,
            num_frame_for_scale=self.num_frame_for_scale,
        )
        return tokens.view(batch_size, seq_len, tokens_per_frame, channels)


class FrameGlobalGroupExportWrapper(nn.Module):
    def __init__(self, model: GCTStream, group_idx: int, num_frame_for_scale: int, num_frame_per_block: int):
        super().__init__()
        self.aggregator = model.aggregator
        self.group_idx = group_idx
        self.num_frame_for_scale = num_frame_for_scale
        self.num_frame_per_block = num_frame_per_block

    def forward(self, tokens: torch.Tensor):
        batch_size, seq_len, tokens_per_frame, channels = tokens.shape
        frame_tokens = self.aggregator.frame_blocks[self.group_idx](
            tokens.view(batch_size * seq_len, tokens_per_frame, channels),
            pos=None,
            enable_ulysses_cp=False,
        )
        global_tokens = self.aggregator.global_blocks[self.group_idx](
            frame_tokens.view(batch_size, seq_len * tokens_per_frame, channels),
            pos=None,
            enable_ulysses_cp=False,
            num_patches=tokens_per_frame - self.aggregator.num_special_tokens,
            num_special=self.aggregator.num_special_tokens,
            num_frames=seq_len,
            enable_3d_rope=False,
            kv_cache=None,
            global_idx=self.group_idx,
            num_frame_per_block=self.num_frame_per_block,
            num_frame_for_scale=self.num_frame_for_scale,
            num_register_tokens=self.aggregator.num_register_tokens,
        )
        frame_tokens = frame_tokens.view(batch_size, seq_len, tokens_per_frame, channels)
        global_tokens = global_tokens.view(batch_size, seq_len, tokens_per_frame, channels)
        group_features = torch.cat([frame_tokens, global_tokens], dim=-1)
        return global_tokens, group_features


class CameraHeadExportWrapper(nn.Module):
    def __init__(self, model: GCTStream, num_frame_for_scale: int, num_frame_per_block: int):
        super().__init__()
        self.camera_head = model.camera_head
        self.num_frame_for_scale = num_frame_for_scale
        self.num_frame_per_block = num_frame_per_block

    def forward(self, final_group_features: torch.Tensor):
        return self.camera_head(
            [final_group_features],
            causal_inference=False,
            num_frame_for_scale=self.num_frame_for_scale,
            num_frame_per_block=self.num_frame_per_block,
        )[-1]


class DenseHeadExportWrapper(nn.Module):
    def __init__(self, head: nn.Module, patch_start_idx: int):
        super().__init__()
        self.head = head
        self.patch_start_idx = patch_start_idx

    def forward(
        self,
        feature_0: torch.Tensor,
        feature_1: torch.Tensor,
        feature_2: torch.Tensor,
        feature_3: torch.Tensor,
        images: torch.Tensor,
    ):
        return self.head(
            [feature_0, feature_1, feature_2, feature_3],
            images=images,
            patch_start_idx=self.patch_start_idx,
        )


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


def _export_module_to_onnx(
    module: nn.Module,
    args: tuple[torch.Tensor, ...],
    output_path: Path,
    *,
    input_names: Sequence[str],
    output_names: Sequence[str],
):
    torch.onnx.export(
        module.eval(),
        args,
        str(output_path),
        export_params=True,
        do_constant_folding=True,
        opset_version=18,
        dynamo=False,
        input_names=list(input_names),
        output_names=list(output_names),
    )


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


def export_split_checkpoint_to_onnx(
    *,
    model_path: str | None,
    output_dir: str | Path,
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
    device: str,
) -> dict:
    _ensure_onnx_export_dependencies()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    sample_tokens = torch.rand(
        batch_size,
        num_frames,
        model.aggregator.patch_start_idx + (image_size // patch_size) ** 2,
        embed_dim,
        device=torch_device,
    )
    sample_group_features = torch.rand(
        batch_size,
        num_frames,
        sample_tokens.shape[2],
        embed_dim * 2,
        device=torch_device,
    )
    selected_feature_groups = _default_selected_feature_groups(len(model.aggregator.frame_blocks))
    manifest = {
        "export_layout": "split",
        "patch_start_idx": model.aggregator.patch_start_idx,
        "num_special_tokens": model.aggregator.num_special_tokens,
        "selected_feature_groups": selected_feature_groups,
        "rope_disabled_for_export": True,
        "files": {},
    }

    try:
        patch_wrapper = PatchEmbedExportWrapper(model, min(num_scale_frames, num_frames))
        patch_path = output_dir / "patch_embed.onnx"
        _export_module_to_onnx(
            patch_wrapper,
            (sample_images,),
            patch_path,
            input_names=["images"],
            output_names=["tokens"],
        )
        manifest["files"]["patch_embed"] = patch_path.name

        group_files = []
        for group_idx in range(len(model.aggregator.frame_blocks)):
            group_wrapper = FrameGlobalGroupExportWrapper(
                model,
                group_idx=group_idx,
                num_frame_for_scale=min(num_scale_frames, num_frames),
                num_frame_per_block=min(num_scale_frames, num_frames),
            )
            group_path = output_dir / f"frame_global_group_{group_idx:02d}.onnx"
            _export_module_to_onnx(
                group_wrapper,
                (sample_tokens,),
                group_path,
                input_names=["tokens"],
                output_names=["next_tokens", "group_features"],
            )
            group_files.append(group_path.name)
        manifest["files"]["frame_global_groups"] = group_files

        if model.camera_head is not None:
            camera_wrapper = CameraHeadExportWrapper(
                model,
                num_frame_for_scale=min(num_scale_frames, num_frames),
                num_frame_per_block=min(num_scale_frames, num_frames),
            )
            camera_path = output_dir / "camera_head.onnx"
            _export_module_to_onnx(
                camera_wrapper,
                (sample_group_features,),
                camera_path,
                input_names=["final_group_features"],
                output_names=["pose_enc"],
            )
            manifest["files"]["camera_head"] = camera_path.name

        dense_feature_inputs = (
            sample_group_features,
            sample_group_features,
            sample_group_features,
            sample_group_features,
            sample_images,
        )
        if model.depth_head is not None:
            depth_path = output_dir / "depth_head.onnx"
            _export_module_to_onnx(
                DenseHeadExportWrapper(model.depth_head, model.aggregator.patch_start_idx),
                dense_feature_inputs,
                depth_path,
                input_names=["feature_0", "feature_1", "feature_2", "feature_3", "images"],
                output_names=["depth", "depth_conf"],
            )
            manifest["files"]["depth_head"] = depth_path.name

        if model.point_head is not None:
            point_path = output_dir / "point_head.onnx"
            _export_module_to_onnx(
                DenseHeadExportWrapper(model.point_head, model.aggregator.patch_start_idx),
                dense_feature_inputs,
                point_path,
                input_names=["feature_0", "feature_1", "feature_2", "feature_3", "images"],
                output_names=["world_points", "world_points_conf"],
            )
            manifest["files"]["point_head"] = point_path.name

        if model.local_point_head is not None:
            local_point_path = output_dir / "local_point_head.onnx"
            _export_module_to_onnx(
                DenseHeadExportWrapper(model.local_point_head, model.aggregator.patch_start_idx),
                dense_feature_inputs,
                local_point_path,
                input_names=["feature_0", "feature_1", "feature_2", "feature_3", "images"],
                output_names=["cam_points", "cam_points_conf"],
            )
            manifest["files"]["local_point_head"] = local_point_path.name

        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest
    finally:
        model.set_export_mode(False)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export LingBot-Map to ONNX.")
    parser.add_argument("--model_path", type=str, default=None, help="Checkpoint path. If omitted, exports random weights.")
    parser.add_argument("--layout", type=str, choices=("split", "full"), default="split")
    parser.add_argument("--output_path", type=str, default=None, help="Output ONNX file path for full export mode.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for split export mode.")
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
    common_kwargs = dict(
        model_path=args.model_path,
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
        device=args.device,
    )
    if args.layout == "split":
        if args.output_dir is None:
            raise ValueError("--output_dir is required when --layout split")
        manifest = export_split_checkpoint_to_onnx(
            output_dir=args.output_dir,
            **common_kwargs,
        )
        print(f"Saved split ONNX export to {args.output_dir}")
        print(json.dumps(manifest, indent=2))
        return

    if args.output_path is None:
        raise ValueError("--output_path is required when --layout full")
    output_names = export_checkpoint_to_onnx(
        output_path=args.output_path,
        opset_version=args.opset_version,
        **common_kwargs,
    )
    print(f"Exported outputs: {', '.join(output_names)}")
    print(f"Saved ONNX model to {args.output_path}")
