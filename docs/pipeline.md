# Pipeline Stages

1. **16-bit validation** — reject empty / non-2-D arrays; accept uint16 / float.
2. **Denoise** — 3×3 Gaussian (bilateral optional but slower).
3. **Robust windowing** — 1st–99th percentile → [0,1].
4. **Anatomical suppression** — large-scale Gaussian illumination plus a fast box-filter bright-structure estimate attenuate ribs, spine and lung/soft-tissue background. The estimated attenuation mask is returned for auditability.
5. **Vessel enhancement** — multi-scale LoG (σ = 1.0, 1.8), max response, morphological close.
6. **Contrast enhancement** — fast histogram equalization in the real-time path; CLAHE remains available as an opt-in offline mode.
7. **Fusion** — weighted combination of CLAHE residual and vessel map.
8. **Output** — raw, processed, enhanced, vessel map, anatomy mask, and measured latency.

## Real-time mode

For 512×512 and larger inputs, the default configuration processes a bounded
256×256 working frame and resizes the outputs back to the source dimensions.
This is intentional: it provides predictable CPU latency while preserving the
display contract. The benchmark accepts a run only when the measured maximum,
not merely the average, is at most 36 ms.

## Evaluation boundaries

The default benchmark measures warmed-up `CoronaryPipeline.process()` calls,
including validation, preprocessing, enhancement, fusion, and resizing, but
excluding file decoding, Streamlit rendering, and disk/network I/O. The
annotated dataset evaluator is a separate contrast proxy and must not be
interpreted as sensitivity, specificity, Dice, or clinical validation.
