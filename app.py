#!/usr/bin/env python3
"""Streamlit demonstration UI for the Coronary Artery Image Processing System.

The core processing engine lives in src/pipeline.py and is independent of this UI.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st
import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.pipeline import CoronaryPipeline
from src.io_utils import read_image, to_uint8_vis, to_uint16, validate_16bit
from src.config import DEFAULT_CONFIG
from benchmark import get_hardware_info, load_frames, run_benchmark


st.set_page_config(
    page_title="Coronary Artery Processor",
    page_icon="❤️",
    layout="wide",
)

st.title("Cardiology Coronary Artery Image Processing System")
st.caption(
    "Research / educational demonstration — not a clinically validated diagnostic device."
)

@st.cache_resource
def get_pipeline():
    pipe = CoronaryPipeline(DEFAULT_CONFIG)
    pipe.warm_up((512, 512))
    return pipe


def load_uploaded(uploaded) -> np.ndarray | None:
    if uploaded is None:
        return None
    try:
        encoded = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("unsupported or corrupted image")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    except Exception as e:
        st.error(f"Failed to read image: {e}")
        return None


def main():
    pipe = get_pipeline()

    with st.sidebar:
        st.header("Input")
        uploaded = st.file_uploader(
            "Upload 16-bit grayscale PNG/TIFF",
            type=["png", "tif", "tiff"],
        )
        use_sample = st.checkbox("Use sample synthetic image", value=True)
        sample_idx = st.selectbox("Sample index", list(range(5)), index=0)

        st.header("Info")
        st.markdown(
            """
            **Pipeline stages**
            1. 16-bit validation & normalization  
            2. Noise reduction  
            3. Background estimation & subtraction  
            4. Anatomical structure attenuation  
            5. Multi-scale vessel enhancement  
            6. Local contrast (CLAHE)  
            7. Fusion & output  
            """
        )
        run_perf = st.button("Run latency benchmark", use_container_width=True)

    img = None
    if uploaded is not None:
        img = load_uploaded(uploaded)
        use_sample = False
    elif use_sample:
        path = ROOT / "data" / "raw" / f"synthetic_angio_{sample_idx:02d}.png"
        if path.exists():
            img = read_image(path)
        else:
            st.error("Sample not found. Add a documented 16-bit sample instead of using placeholder data.")
            return

    if img is None:
        st.info("Upload an image or enable the sample to begin.")
        return

    ok, msg = validate_16bit(img)
    st.write(f"**Input:** shape={img.shape}, dtype={img.dtype}, validation={msg}")

    with st.spinner("Processing…"):
        result = pipe.process(img)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("1. Original / Raw")
        st.image(to_uint8_vis(result["raw"]), use_container_width=True, clamp=True)
    with col2:
        st.subheader("2. Processed (BG suppressed)")
        st.image(to_uint8_vis(result["processed"]), use_container_width=True, clamp=True)
    with col3:
        st.subheader("3. Enhanced Coronary Vessels")
        st.image(to_uint8_vis(result["enhanced"]), use_container_width=True, clamp=True)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Frame latency", f"{result['latency_ms']:.2f} ms")
    working_h, working_w = result["working_shape"]
    metric_cols[1].metric("Working resolution", f"{working_w} x {working_h}")
    metric_cols[2].metric("Input bit depth", "16-bit" if img.dtype == np.uint16 else str(img.dtype))
    metric_cols[3].metric("Requirement", "PASS" if result["latency_ms"] <= 36.0 else "OVER BUDGET")
    st.caption(
        "Latency is measured from frame entry through final enhanced output; UI rendering is excluded."
    )

    with st.expander("Anatomical suppression diagnostic"):
        st.image(to_uint8_vis(result["anatomy_mask"]), caption="Estimated bright rib/spine/background attenuation mask", use_container_width=True)

    processed_png = cv2.imencode(".png", to_uint16(result["processed"], scale_from_01=True))[1].tobytes()
    enhanced_png = cv2.imencode(".png", to_uint16(result["enhanced"], scale_from_01=True))[1].tobytes()
    st.download_button("Download processed 16-bit PNG", processed_png, "processed_16bit.png", "image/png")
    st.download_button("Download enhanced 16-bit PNG", enhanced_png, "enhanced_16bit.png", "image/png")

    if run_perf:
        with st.spinner("Running 60 measured frames..."):
            frames = load_frames(ROOT / "data" / "raw", max_frames=5)
            benchmark = run_benchmark(frames, warmup=8, iterations=60)
        st.subheader("Measured performance")
        perf_cols = st.columns(5)
        perf_cols[0].metric("Minimum", f"{benchmark['min_ms']:.2f} ms")
        perf_cols[1].metric("Average", f"{benchmark['mean_ms']:.2f} ms")
        perf_cols[2].metric("Maximum", f"{benchmark['max_ms']:.2f} ms")
        perf_cols[3].metric("P95", f"{benchmark['p95_ms']:.2f} ms")
        perf_cols[4].metric("FPS", f"{benchmark['fps']:.1f}")
        status = "PASS" if benchmark["max_ms"] <= 36.0 else "FAIL"
        st.write(f"**Strict maximum-latency status:** {status}  |  Hardware: {get_hardware_info().get('cpu', 'unknown')}")

    st.markdown("---")
    st.markdown(
        "**Disclaimer:** This system is an engineering demonstration for research "
        "and educational purposes only. It is not intended for clinical diagnosis "
        "or patient care."
    )


if __name__ == "__main__":
    main()
