# anim.e Studio — Phase 1 Hardware Benchmark & Verification Report

- **Phase 1 Automated Status:** **`PASS (Automated Technical Validation Passed — Pending User Visual Inspection)`**
- **Human Visual Quality Inspection Status:** **`PENDING USER MANUAL VERIFICATION`**
- **Report Timestamp:** `2026-11-01T02:57:39+0800`

## 1. Installation & Runtime Environment Verification
- **GPU Model (Measured):** `NVIDIA GeForce RTX 3050 Laptop GPU` (Compute Capability `8.6`, `4096 MiB` Dedicated VRAM)
- **NVIDIA Driver Version:** `610.78`
- **Embedded Python Version:** `3.12.10` (`engine/ComfyUI_windows_portable/python_embeded/python.exe`)
- **PyTorch & CUDA Runtime:** `torch 2.14.1+cu126` | CUDA `12.6` | `cuda_available=True`
- **ComfyUI Version & Git Commit:** `v0.43.0 (cu126)` (Commit `426f5eea52a7ece1dd599fff4cb6cd1f352ddd24`)
- **Python Dependencies:** `opencv-python-headless 4.11.0` | `imageio-ffmpeg 0.6.0`
- **FFmpeg Binary & H.264 Support:** `C:\Users\Renz Jericho Buday\Anim.e\engine\ComfyUI_windows_portable\python_embeded\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe` (`libx264_supported=True`)

## 2. Disk Space Budget (Measured Across Phase 1 Lifecycle)
| Measurement Stage | Free Space / Size on `C:\` | Notes |
| :--- | :--- | :--- |
| **1. Pre-Installation Free Space (`C:\`)** | `37.95 GiB` | Measured during initial pre-flight audit before any Phase 1 downloads |
| **2. Benchmark-Start Free Space (`C:\`)** | `24.51 GiB` | Measured after ComfyUI extraction, `.7z` cleanup, custom nodes, and 3 model downloads |
| **3. Post-Benchmark Free Space (`C:\`)** | **`24.27 GiB`** | Measured after benchmark execution (Safety Floor: `15.00 GiB`) |
| **`engine/` Directory Footprint** | `9.55 GiB` | ComfyUI Portable + Custom Nodes + 3 Model Files |
| **`data/` Directory Footprint** | `0.00 GiB` | Generated benchmark outputs, companion files, and logs |
| **Total Net Disk Space Consumed** | `13.68 GiB` | `Pre-Install (37.95 GiB) - Post-Benchmark (24.27 GiB)` |

## 3. Custom Nodes & Cryptographic Model Inventory

### Installed Custom Nodes
| Custom Node Repository | Pinned Git Commit SHA | License |
| :--- | :--- | :--- |
| `ComfyUI-AnimateDiff-Evolved` | `9257651221002dcba0a12f9cff37e1944e58fb60` | Apache-2.0 |
| `ComfyUI-VideoHelperSuite` | `4d907bee61e92c2e65af3bd6383a4e4d356126d1` | GPL-3.0 |

### Model Inventory & Full SHA-256 Cryptographic Verification
| Model Filename | Role | Size (MiB) | Full Local SHA-256 | Upstream HF LFS SHA-256 Authentication | Format / Header Check |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Counterfeit-V3.0_fix_fp16.safetensors` | checkpoint (SD 1.5 FP16 Pruned) | `2033.85 MiB` | `a54c944e4c04e9d9ca43468ef5b90ea0408bb6264829a4980de79a768df7179f` | **VERIFIED** (`a54c944e4c04e9d9ca43468ef5b90ea0408bb6264829a4980de79a768df7179f`) | Valid safetensors (1145 tensors) |
| `vae-ft-mse-840000-ema-pruned.safetensors` | vae (SD 1.5 FP16 Pruned) | `319.14 MiB` | `735e4c3a447a3255760d7f86845f09f937809baa529c17370d83e4c3758f3c75` | **VERIFIED** (`735e4c3a447a3255760d7f86845f09f937809baa529c17370d83e4c3758f3c75`) | Valid safetensors (250 tensors) |
| `mm_sd_v15_v2.ckpt` | motion_module (AnimateDiff v2 SD 1.5) | `1733.67 MiB` | `69ed0f5fef82b110aca51bcab73b21104242bc65d6ab4b8b2a2a94d31cad1bf0` | **VERIFIED** (`69ed0f5fef82b110aca51bcab73b21104242bc65d6ab4b8b2a2a94d31cad1bf0`) | Exact byte size & full SHA-256 verified |

## 4. Empirical Benchmark Ladder Results

> **Note on Validation Scope:** The `Status` column below reflects **automated technical validation** (ComfyUI execution completion, valid container, exact resolution, exact frame count, non-black pixel statistics, and inter-frame pixel delta). **Human visual inspection** of animation smoothness and artistic quality must be performed manually by opening the generated files.

| Rung | Automated Status | Resolution & Frames | Steps / Sampler | Duration | Baseline -> Peak VRAM | Avg / Peak GPU Util | Max Temp & Power | Primary Output, Companion Files & Technical Validation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Rung 0: Basic SD 1.5 Sanity Test (384x384, 1 frame)** | `PASSED` | `384x384` (1f @ 1 FPS) | `15` / `euler_ancestral` | `14.33s` | `63` -> **`2445 MiB`** / `4096 MiB` | `31.8%` / `100%` | `57°C` / `42.6W` | `data\outputs\poc\rung0_sanity_384x384_1f.png` (212.7 KB, tech_valid=True, 384x384, mean_px=64.75, visual=`PENDING_USER_REVIEW`) |
| **Rung 1: First AnimateDiff Video (384x384, 8 frames @ 8 FPS)** | `PASSED` | `384x384` (8f @ 8 FPS) | `15` / `euler_ancestral` | `39.62s` | `141` -> **`3151 MiB`** / `4096 MiB` | `64.4%` / `100%` | `64°C` / `44.9W` | `data\outputs\poc\rung1_video_384x384_8f.mp4` (77.8 KB, tech_valid=True, 384x384, 8f @ 8.0 FPS = 1.0s, motion_diff=9.18, visual=`PENDING_USER_REVIEW`) |
| **Rung 2: Conservative 16-Frame Video (384x384, 16 frames @ 8 FPS)** | `PASSED` | `384x384` (16f @ 8 FPS) | `16` / `euler_ancestral` | `58.59s` | `143` -> **`3343 MiB`** / `4096 MiB` | `79.5%` / `100%` | `69°C` / `44.9W` | `data\outputs\poc\rung2_video_384x384_16f.mp4` (103.4 KB, tech_valid=True, 384x384, 16f @ 8.0 FPS = 2.0s, motion_diff=9.26, visual=`PENDING_USER_REVIEW`) |
| **Rung 3a: Portrait Video (384x512, 8 frames @ 8 FPS)** | `PASSED` | `384x512` (8f @ 8 FPS) | `16` / `euler_ancestral` | `45.41s` | `143` -> **`3313 MiB`** / `4096 MiB` | `73.3%` / `100%` | `70°C` / `44.8W` | `data\outputs\poc\rung3a_video_384x512_8f.mp4` (125.7 KB, tech_valid=True, 384x512, 8f @ 8.0 FPS = 1.0s, motion_diff=22.39, visual=`PENDING_USER_REVIEW`) |
| **Rung 3b: Square Standard Video (512x512, 8 frames @ 8 FPS)** | `PASSED` | `512x512` (8f @ 8 FPS) | `16` / `euler_ancestral` | `56.58s` | `145` -> **`3409 MiB`** / `4096 MiB` | `78.9%` / `100%` | `71°C` / `45.0W` | `data\outputs\poc\rung3b_video_512x512_8f.mp4` (152.9 KB, tech_valid=True, 512x512, 8f @ 8.0 FPS = 1.0s, motion_diff=9.32, visual=`PENDING_USER_REVIEW`) |

## 5. Evidence-Based Findings, Errors & Safe Default Recommendations
- **Post-Shutdown GPU VRAM Used:** `0 MiB`
- **Rung 0 (Static Sanity) Passed:** `True`
- **Rung 1 (8-Frame Video Baseline) Passed:** `True`
- **Total Passed Rungs:** `5` of `5` configured rungs (`4` video rungs passed).
- **Recorded Failures / Errors:** None on executed rungs.

### Recommended Default Generation Settings (Strictly from Verified Passing Rungs)
- **Verified Configuration (`rung1`):** Resolution `384x384`, `8` frames @ `8` FPS, `15` steps, sampler `euler_ancestral` (Measured Peak VRAM: `3151 MiB`, Duration: `39.62s`).
- **Verified Configuration (`rung2`):** Resolution `384x384`, `16` frames @ `8` FPS, `16` steps, sampler `euler_ancestral` (Measured Peak VRAM: `3343 MiB`, Duration: `58.59s`).
- **Verified Configuration (`rung3a`):** Resolution `384x512`, `8` frames @ `8` FPS, `16` steps, sampler `euler_ancestral` (Measured Peak VRAM: `3313 MiB`, Duration: `45.41s`).
- **Verified Configuration (`rung3b`):** Resolution `512x512`, `8` frames @ `8` FPS, `16` steps, sampler `euler_ancestral` (Measured Peak VRAM: `3409 MiB`, Duration: `56.58s`).
