# Query-Interaction Intervention Audit: TMLR Artifact

This package reproduces the archived H4-D, Gate C, T1, and T2 scientific
aggregates used in the paper from compact, per-image analysis-ready tables.
It performs no model inference and requires no GPU.

The terms `frozen` and `fixed` in filenames or historical status fields denote
project-internal locked records. They do not imply an externally timestamped
preregistration.

## Scientific scope

The package supports the following bounded claims:

- a reproducible hard-mask attention sensitivity on the original discovery
  population;
- failure of the project-recorded Gate C operator-transport conjunction;
- localization of hard-minus-mass differences before matching-sensitive
  readout in T1;
- T2 pairwise equivalence classifications together with failure of the hard
  positive control.

The package does not support a general harmful-query-edge claim, an
intervention-invariant causal mechanism, population-wide prevalence, or a
static regularization recommendation.

## Quick reproduction

From the artifact root:

```powershell
python -B code/p1_artifact_manifest.py --root . --verify
python -B code/p1_reproduce_from_analysis_ready.py `
  --data-root analysis_ready `
  --output-root reproduced_run `
  --expected-root expected
```

Expected terminal status:

```text
PASS_ANALYSIS_READY_EXACT_REPRODUCTION
```

The validation tolerance is `1e-12`. On the frozen package used to prepare
this artifact, the maximum absolute numeric differences were:

| Result | Maximum absolute difference |
|---|---:|
| H4-D | 2.2090619264392153e-18 |
| Gate C | 1.3877787807814457e-17 |
| T1 | 0 |
| T2 | 4.440892098500626e-16 |

## Requirements

The CPU-only reproduction script requires:

- Python 3.12;
- NumPy 1.26.x.

The original execution environment was:

- Python 3.12.13;
- PyTorch 2.10.0+cu128;
- torchvision 0.25.0+cu128;
- NumPy 1.26.4;
- SciPy 1.17.1;
- NVIDIA RTX 4090 for the original model runs.

PyTorch, SciPy, model weights, image pixels, and a GPU are not required to
reproduce the reported aggregates from `analysis_ready/`.

## Directory layout

```text
analysis_ready/   Per-image rows required by the frozen statistics
audit/            Integrity audits, population/pair manifests, and dose mappings
code/             Compact reproducer, manifest verifier, and reference aggregators
configs/          Frozen statistical and intervention configurations
expected/         Frozen aggregate JSON files
reproduced/       Validation outputs generated during P1
```

`reproduced_selftest/` is an optional local self-test output and is excluded
from the frozen manifest.

## Analysis-ready transformation

The original raw receipts contain large prediction-by-ground-truth quality
matrices. Collectively they exceed the TMLR 100MB supplementary-material
limit after ordinary ZIP compression.

`code/p1_export_analysis_ready.py` transforms those receipts into per-image
rows containing only values used by the frozen estimands and statistics:

- H4-D local, fixed/rematched/native, focal/spillover/matching/selection, and
  control effects;
- Gate C target/control effects for every frozen dose;
- T1 hard-minus-mapped-mass contrasts for all eight readouts;
- T2 M/P/D/H effects, proportional curves, realized message norms, and
  population strata.

The transformation reuses optimal assignments and image-set utilities already
stored in the integrity-checked receipts. It does not refit, rematch, select
images, read outcomes during mapping, or run a model. The compact rows were
validated against the archived aggregates at `1e-12` tolerance.

The export script is included for provenance, but it requires the full private
source workspace and is not needed for ordinary artifact reproduction.

## Data provenance

The discovery and Gate C population comes from the recorded Open Images
validation-derived D population with COCO-category mapping. The T2
confirmation population comes from the public COCO 2017 validation split and
was selected without reading model or intervention outcomes.

Image pixels are not redistributed in this package. The analysis-ready files
contain opaque image identifiers, image SHA-256 values, frozen strata, and
derived numerical effects.

Relevant public sources:

- Open Images: <https://storage.googleapis.com/openimages/web/index.html>
- COCO: <https://cocodataset.org/>

Users who wish to rerun model inference must obtain those datasets under their
respective terms.

## Model provenance

The original study used:

- DETR-R50-500, checkpoint SHA-256
  `e632da11ec76ae67bac2f8579fbed3724e08dead7d200ca13e019b197784eadc`;
- DINO-R50 4-scale 12-epoch, checkpoint SHA-256
  `0bcd6b0c33d60ed33461ce6f02ce5797a819c7c02eb7e15b76adfb6df307955a`.

Model weights are not included. Additional provenance is recorded in
`DATA_AND_MODEL_PROVENANCE.md`.

## Statistical unit and uncertainty

- H4-D and Gate C use image-level stratified bootstrap intervals;
- T1 uses paired-image, within-model max-studentized simultaneous intervals;
- T2 uses paired-image, stratum-resampled max-studentized simultaneous
  intervals over the frozen contrast family;
- models are analyzed separately;
- T2 equivalence bands are 0.005 for local quality and 0.001 for spillover and
  fixed-set utility.

The T2 terminal label `T2_INCONCLUSIVE_PRECISION` must not be interpreted as
ordinary low precision alone. The pairwise comparisons were classified within
the frozen bands, but the hard-deletion positive control failed in both
models, so mechanism classification was excluded.

## V/F firewall

No V/F data, result, receipt, pixel, or derived statistic is included.
All included core aggregates record `V_F_read = false` or the corresponding
split-specific false flag.

## Known exclusions

- Full raw quality matrices are excluded for size and licensing reasons.
- Model weights and image pixels are excluded.
- T0B raw input-only receipts are excluded. The package includes the T0B
  overlap output, D-capture integrity record, R-pilot integrity record and
  manifest, configuration, and reference audit code. These materials verify
  the reported output and its hashes, but the T0B overlap calculation cannot
  be rerun without the excluded raw input-only receipts.
- N1 raw receipts are excluded because N1 is used only as a historical
  actionability boundary in the paper; its archived aggregate, integrity audit,
  and configurations are included.
- T0 is an input/estimand audit rather than a new model-effect result. Its
  frozen audit and mapping dependencies are included, while the main
  scientific reproduction is performed for H4-D, Gate C, T1, and T2.

## Integrity

`ARTIFACT_MANIFEST.json` records the byte size and SHA-256 of every packaged
package file except itself and optional self-test output. Verify it before
running the statistical reproduction.
