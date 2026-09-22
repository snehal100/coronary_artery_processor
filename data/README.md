# Dataset

## Synthetic development set

- Files: `raw/synthetic_angio_00.png` … `synthetic_angio_04.png`
- Format: 16-bit grayscale PNG, 512×512
- Content: procedurally generated background (spine, ribs, soft tissue) + dark tubular vessels + Gaussian noise
- Purpose: unit tests, benchmarks, demo UI
- License: generated for this project; free to use

## Annotated development set

The repository also contains `raw/data/stenosis/` and `raw/data/syntax/`.
Each has COCO-style `train`, `val`, and `test` folders with 512x512 PNG
images and JSON annotations. These checked-in files are 8-bit RGB PNGs; the
loader converts them to grayscale. They are not a 16-bit clinical acquisition
set. Verify provenance and permissions before redistributing them.

Run the engineering proxy evaluation with:

```bash
python scripts/evaluate_dataset.py --split test --limit 100
```

This compares annotated-region contrast before and after processing. It is not
a clinical accuracy score, and the annotations do not provide rib, spine, lung,
or complete coronary segmentation ground truth.

## Adding real data

Place any 16-bit grayscale PNG or TIFF coronary angiography frames into `raw/`.  
The pipeline and benchmark will pick them up automatically.

**Do not commit private patient data.**
