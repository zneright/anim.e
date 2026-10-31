"""Workflow template generator for ComfyUI AnimateDiff video generation."""

from typing import Any
from backend.app.core.config import settings

CHECKPOINT_NAME = "Counterfeit-V3.0_fix_fp16.safetensors"
VAE_NAME = "vae-ft-mse-840000-ema-pruned.safetensors"
MOTION_MODULE_NAME = "mm_sd_v15_v2.ckpt"

PRESET_CONFIGS: dict[str, dict[str, Any]] = {
    "fast_draft": {
        "width": 384,
        "height": 384,
        "frame_count": 8,
        "fps": 8,
        "steps": 15,
        "cfg_scale": 7.0,
        "sampler_name": "euler_ancestral",
        "scheduler": "normal",
    },
    "standard_loop": {
        "width": 384,
        "height": 384,
        "frame_count": 16,
        "fps": 8,
        "steps": 16,
        "cfg_scale": 7.0,
        "sampler_name": "euler_ancestral",
        "scheduler": "normal",
    },
    "high_res_square": {
        "width": 512,
        "height": 512,
        "frame_count": 8,
        "fps": 8,
        "steps": 16,
        "cfg_scale": 7.0,
        "sampler_name": "euler_ancestral",
        "scheduler": "normal",
    },
    "experimental_portrait": {
        "width": 384,
        "height": 512,
        "frame_count": 8,
        "fps": 8,
        "steps": 16,
        "cfg_scale": 7.0,
        "sampler_name": "euler_ancestral",
        "scheduler": "normal",
    },
}


class WorkflowBuilder:
    """Builds validated ComfyUI prompt API graphs matching Phase 1 benchmarks."""

    @staticmethod
    def build_txt2vid_workflow(
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        frame_count: int,
        fps: int,
        steps: int,
        cfg_scale: float,
        seed: int,
        sampler_name: str = "euler_ancestral",
        scheduler: str = "normal",
        output_prefix: str = "anim_e_out",
    ) -> dict[str, Any]:
        return {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": CHECKPOINT_NAME},
            },
            "2": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": VAE_NAME},
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": prompt, "clip": ["1", 1]},
            },
            "4": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": negative_prompt, "clip": ["1", 1]},
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": width, "height": height, "batch_size": frame_count},
            },
            "6": {
                "class_type": "ADE_AnimateDiffLoaderGen1",
                "inputs": {
                    "model": ["1", 0],
                    "model_name": MOTION_MODULE_NAME,
                    "beta_schedule": "autoselect",
                },
            },
            "7": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg_scale,
                    "sampler_name": sampler_name,
                    "scheduler": scheduler,
                    "denoise": 1.0,
                    "model": ["6", 0],
                    "positive": ["3", 0],
                    "negative": ["4", 0],
                    "latent_image": ["5", 0],
                },
            },
            "8": {
                "class_type": "VAEDecodeTiled",
                "inputs": {
                    "samples": ["7", 0],
                    "vae": ["2", 0],
                    "tile_size": 384,
                    "overlap": 64,
                    "temporal_size": 8,
                    "temporal_overlap": 4,
                },
            },
            "9": {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "images": ["8", 0],
                    "frame_rate": fps,
                    "loop_count": 0,
                    "filename_prefix": output_prefix,
                    "format": "video/h264-mp4",
                    "pix_fmt": "yuv420p",
                    "crf": 19,
                    "save_metadata": True,
                    "trim_to_audio": False,
                    "pingpong": False,
                    "save_output": True,
                },
            },
        }
