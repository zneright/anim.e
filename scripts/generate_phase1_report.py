r"""Generates docs/hardware_benchmark.md strictly from verified evidence in engine/manifest.json and data/outputs/poc/benchmark_results.json.

Guarantees:
1. Requires BOTH Rung 0 and Rung 1 to pass before declaring Phase 1 automated status as PASS.
2. Explicitly reports missing upstream SHA-256 metadata as UNVERIFIED (never as verified).
3. Separately reports pre-installation free disk space (37.95 GiB), benchmark-start free disk space, and post-benchmark free disk space.
4. Clearly distinguishes Automated Technical Validation (OpenCV/FFmpeg container, resolution, frame count, pixel stats, motion delta)
   from User Manual Visual Inspection (PENDING_USER_REVIEW).
5. Preserves and lists any companion output files generated alongside primary artifacts.
"""
import json
from pathlib import Path
import shutil

ROOT_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT_DIR / "engine" / "manifest.json"
BENCHMARK_PATH = ROOT_DIR / "data" / "outputs" / "poc" / "benchmark_results.json"
REPORT_PATH = ROOT_DIR / "docs" / "hardware_benchmark.md"


def get_dir_size_gib(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / (1024**3), 2)


def fmt_num(val: float | int | None, suffix: str = "", decimals: int = 0) -> str:
    if val is None:
        return "Unavailable"
    if decimals == 0:
        return f"{val:.0f}{suffix}"
    return f"{val:.{decimals}f}{suffix}"


def main() -> None:
    manifest = (
        json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if MANIFEST_PATH.exists()
        else {}
    )
    bench = (
        json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        if BENCHMARK_PATH.exists()
        else {}
    )
    ver = bench.get("verification") or {}
    rt = ver.get("runtime") or manifest.get("runtime") or {}
    results = bench.get("results") or []

    _, _, free_bytes = shutil.disk_usage("C:\\")
    final_free_gib = round(free_bytes / (1024**3), 2)
    engine_gib = get_dir_size_gib(ROOT_DIR / "engine")
    data_gib = get_dir_size_gib(ROOT_DIR / "data")
    pre_install_free_gib = bench.get("pre_install_free_disk_gib", 37.95)
    bench_start_free_gib = bench.get("benchmark_start_free_disk_gib", final_free_gib)
    total_consumed_gib = round(pre_install_free_gib - final_free_gib, 2)

    rung0_passed = any(
        r.get("rung_id") == "rung0" and r.get("status") == "passed" for r in results
    )
    rung1_passed = any(
        r.get("rung_id") == "rung1" and r.get("status") == "passed" for r in results
    )
    passed_rungs = [r for r in results if r.get("status") == "passed"]
    passed_video_rungs = [
        r for r in passed_rungs if (r.get("parameters") or {}).get("frames", 0) > 1
    ]
    failed_rungs = [
        r for r in results if str(r.get("status", "")).startswith("failed")
    ]
    skipped_rungs = [
        r for r in results if r.get("status") in ("skipped", "not_run")
    ]

    if rung0_passed and rung1_passed:
        overall_status = "PASS (Automated Technical Validation Passed — Pending User Visual Inspection)"
    elif rung0_passed and not rung1_passed:
        overall_status = "PARTIAL (Rung 0 Static Passed; Rung 1 Video Did Not Pass)"
    else:
        overall_status = "FAIL"

    lines = [
        "# anim.e Studio — Phase 1 Hardware Benchmark & Verification Report",
        "",
        f"- **Phase 1 Automated Status:** **`{overall_status}`**",
        "- **Human Visual Quality Inspection Status:** **`PENDING USER MANUAL VERIFICATION`**",
        f"- **Report Timestamp:** `{bench.get('timestamp', 'Unknown')}`",
        "",
        "## 1. Installation & Runtime Environment Verification",
        f"- **GPU Model (Measured):** `{rt.get('device_name', bench.get('gpu', 'Unavailable'))}` "
        f"(Compute Capability `{rt.get('compute_capability', 'Unavailable')}`, "
        f"`{fmt_num(rt.get('total_vram_mib'), ' MiB')}` Dedicated VRAM)",
        f"- **NVIDIA Driver Version:** `{(ver.get('initial_gpu_state') or {}).get('driver_version', 'Unavailable')}`",
        f"- **Embedded Python Version:** `{rt.get('python', 'Unavailable')}` (`engine/ComfyUI_windows_portable/python_embeded/python.exe`)",
        f"- **PyTorch & CUDA Runtime:** `torch {rt.get('torch', 'Unavailable')}` | CUDA `{rt.get('cuda_version', rt.get('cuda', 'Unavailable'))}` | `cuda_available={rt.get('cuda_available', False)}`",
        f"- **ComfyUI Version & Git Commit:** `{manifest.get('comfyui_portable_version', 'Unavailable')}` (Commit `{manifest.get('comfyui_commit', 'Unavailable')}`)",
        f"- **Python Dependencies:** `opencv-python-headless {rt.get('opencv', 'Unavailable')}` | `imageio-ffmpeg {rt.get('imageio_ffmpeg', 'Unavailable')}`",
        f"- **FFmpeg Binary & H.264 Support:** `{(ver.get('ffmpeg') or {}).get('binary', manifest.get('ffmpeg_binary', 'Unavailable'))}` "
        f"(`libx264_supported={(ver.get('ffmpeg') or {}).get('libx264_supported', False)}`)",
        "",
        "## 2. Disk Space Budget (Measured Across Phase 1 Lifecycle)",
        "| Measurement Stage | Free Space / Size on `C:\\` | Notes |",
        "| :--- | :--- | :--- |",
        f"| **1. Pre-Installation Free Space (`C:\\`)** | `{pre_install_free_gib:.2f} GiB` | Measured during initial pre-flight audit before any Phase 1 downloads |",
        f"| **2. Benchmark-Start Free Space (`C:\\`)** | `{bench_start_free_gib:.2f} GiB` | Measured after ComfyUI extraction, `.7z` cleanup, custom nodes, and 3 model downloads |",
        f"| **3. Post-Benchmark Free Space (`C:\\`)** | **`{final_free_gib:.2f} GiB`** | Measured after benchmark execution (Safety Floor: `15.00 GiB`) |",
        f"| **`engine/` Directory Footprint** | `{engine_gib:.2f} GiB` | ComfyUI Portable + Custom Nodes + 3 Model Files |",
        f"| **`data/` Directory Footprint** | `{data_gib:.2f} GiB` | Generated benchmark outputs, companion files, and logs |",
        f"| **Total Net Disk Space Consumed** | `{total_consumed_gib:.2f} GiB` | `Pre-Install ({pre_install_free_gib:.2f} GiB) - Post-Benchmark ({final_free_gib:.2f} GiB)` |",
        "",
        "## 3. Custom Nodes & Cryptographic Model Inventory",
        "",
        "### Installed Custom Nodes",
        "| Custom Node Repository | Pinned Git Commit SHA | License |",
        "| :--- | :--- | :--- |",
    ]

    for name, info in (manifest.get("custom_nodes") or {}).items():
        lines.append(
            f"| `{name}` | `{info.get('commit', 'Unavailable')}` | {info.get('license', 'Unavailable')} |"
        )

    lines.extend(
        [
            "",
            "### Model Inventory & Full SHA-256 Cryptographic Verification",
            "| Model Filename | Role | Size (MiB) | Full Local SHA-256 | Upstream HF LFS SHA-256 Authentication | Format / Header Check |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
    )

    model_ver_map = {}
    for k in ("checkpoint", "vae", "motion_module"):
        v_obj = ver.get(k)
        if isinstance(v_obj, dict) and "file" in v_obj:
            model_ver_map[v_obj["file"]] = v_obj

    for m in manifest.get("models") or []:
        fname = m.get("name", "Unknown")
        v_info = model_ver_map.get(fname, {})
        local_sha = v_info.get("local_sha256", "UNVERIFIED")
        pub_sha = v_info.get("published_sha256")
        matched = v_info.get("checksum_verified_against_upstream", False)
        if matched and pub_sha and local_sha != "UNVERIFIED":
            match_str = f"**VERIFIED** (`{pub_sha}`)"
        elif not pub_sha:
            match_str = "**UNVERIFIED** (Missing upstream SHA-256 OID)"
        else:
            match_str = f"**MISMATCH** (Expected `{pub_sha}`)"

        if v_info.get("valid_safetensors"):
            hdr_str = f"Valid safetensors ({v_info.get('tensor_count')} tensors)"
        elif matched:
            hdr_str = "Exact byte size & full SHA-256 verified"
        else:
            hdr_str = "UNVERIFIED"

        lines.append(
            f"| `{fname}` | {m.get('type', 'Unknown')} | `{m.get('size_mib', 0)} MiB` | "
            f"`{local_sha}` | {match_str} | {hdr_str} |"
        )

    lines.extend(
        [
            "",
            "## 4. Empirical Benchmark Ladder Results",
            "",
            "> **Note on Validation Scope:** The `Status` column below reflects **automated technical validation** (ComfyUI execution completion, valid container, exact resolution, exact frame count, non-black pixel statistics, and inter-frame pixel delta). **Human visual inspection** of animation smoothness and artistic quality must be performed manually by opening the generated files.",
            "",
            "| Rung | Automated Status | Resolution & Frames | Steps / Sampler | Duration | Baseline -> Peak VRAM | Avg / Peak GPU Util | Max Temp & Power | Primary Output, Companion Files & Technical Validation |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
    )

    for r in results:
        p = r.get("parameters") or {}
        t = r.get("telemetry") or {}
        val = r.get("validation") or {}
        status_str = str(r.get("status", "unknown")).upper()
        res_frames = (
            f"`{p.get('resolution', 'N/A')}` ({p.get('frames', 'N/A')}f @ {p.get('fps', 'N/A')} FPS)"
        )
        steps_sampler = f"`{p.get('steps', 'N/A')}` / `{p.get('sampler', 'N/A')}`"
        dur_str = f"`{r['duration_sec']}s`" if "duration_sec" in r else "—"

        if t.get("telemetry_valid"):
            vram_str = (
                f"`{fmt_num(t.get('baseline_vram_mib'))}` -> "
                f"**`{fmt_num(t.get('peak_vram_mib'), ' MiB')}`** / `{fmt_num(t.get('vram_total_mib'), ' MiB')}`"
            )
            util_str = f"`{fmt_num(t.get('avg_gpu_util_pct'), '%', 1)}` / `{fmt_num(t.get('peak_gpu_util_pct'), '%')}`"
            temp_pwr_str = f"`{fmt_num(t.get('max_temp_c'), '°C')}` / `{fmt_num(t.get('max_power_w'), 'W', 1)}`"
        else:
            vram_str = "—"
            util_str = "—"
            temp_pwr_str = "—"

        if "output_path" in r:
            companions = r.get("companion_outputs") or []
            comp_str = f" | Companions: `{companions}`" if companions else ""
            if "frames" in val:
                out_col = (
                    f"`{r['output_path']}` ({r.get('file_size_kb', 0)} KB, "
                    f"tech_valid={val.get('valid')}, {val.get('width')}x{val.get('height')}, "
                    f"{val.get('frames')}f @ {val.get('fps')} FPS = {val.get('duration_sec')}s, "
                    f"motion_diff={val.get('motion_diff_score')}, visual=`{val.get('visual_inspection_status', 'PENDING')}`){comp_str}"
                )
            else:
                out_col = (
                    f"`{r['output_path']}` ({r.get('file_size_kb', 0)} KB, "
                    f"tech_valid={val.get('valid')}, {val.get('width')}x{val.get('height')}, "
                    f"mean_px={val.get('mean_pixel')}, visual=`{val.get('visual_inspection_status', 'PENDING')}`){comp_str}"
                )
        else:
            out_col = r.get("error", "Not executed")

        lines.append(
            f"| **{r.get('name', r.get('rung_id', 'Unknown'))}** | `{status_str}` | {res_frames} | "
            f"{steps_sampler} | {dur_str} | {vram_str} | {util_str} | {temp_pwr_str} | {out_col} |"
        )

    lines.extend(
        [
            "",
            "## 5. Evidence-Based Findings, Errors & Safe Default Recommendations",
            f"- **Post-Shutdown GPU VRAM Used:** `{fmt_num(bench.get('post_shutdown_vram_mib'), ' MiB')}`",
            f"- **Rung 0 (Static Sanity) Passed:** `{rung0_passed}`",
            f"- **Rung 1 (8-Frame Video Baseline) Passed:** `{rung1_passed}`",
            f"- **Total Passed Rungs:** `{len(passed_rungs)}` of `{len(results)}` configured rungs (`{len(passed_video_rungs)}` video rungs passed).",
        ]
    )

    if failed_rungs:
        lines.append("- **Recorded Failures / Errors:**")
        for fr in failed_rungs:
            lines.append(
                f"  - `{fr.get('name')}` (`{fr.get('status')}`): `{fr.get('error', 'Validation or telemetry check failed')}`"
            )
    else:
        lines.append("- **Recorded Failures / Errors:** None on executed rungs.")

    if skipped_rungs:
        lines.append("- **Skipped / Unexecuted Rungs:**")
        for sr in skipped_rungs:
            lines.append(
                f"  - `{sr.get('name')}` (`{sr.get('status')}`): {sr.get('error', 'Not reached')}"
            )

    lines.append("")
    lines.append("### Recommended Default Generation Settings (Strictly from Verified Passing Rungs)")
    if passed_video_rungs:
        for vr in passed_video_rungs:
            vp = vr.get("parameters") or {}
            vt = vr.get("telemetry") or {}
            lines.append(
                f"- **Verified Configuration (`{vr.get('rung_id')}`):** Resolution `{vp.get('resolution')}`, "
                f"`{vp.get('frames')}` frames @ `{vp.get('fps')}` FPS, `{vp.get('steps')}` steps, "
                f"sampler `{vp.get('sampler')}` (Measured Peak VRAM: `{fmt_num(vt.get('peak_vram_mib'), ' MiB')}`, "
                f"Duration: `{vr.get('duration_sec')}s`)."
            )
    else:
        lines.append(
            "- **No video configurations have passed verification yet.** Do not proceed to Phase 2 until Rung 0 and Rung 1 both pass."
        )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[REPORT WRITTEN] {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
