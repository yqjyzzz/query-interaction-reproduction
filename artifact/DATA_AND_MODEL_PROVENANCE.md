# Data and Model Provenance

## Data populations

### Original discovery / Gate C population

- Source family: Open Images validation;
- mapped label space: frozen COCO-category mapping;
- analysis population: frozen D population;
- complete paired images: 710 per model;
- population manifest used by Gate C:
  `audit/H4_D_FULL_GT_MANIFEST.json` in the full internal archive;
- image pixels are not included in the TMLR artifact.

The public artifact retains only opaque image identifiers, image hashes,
strata, and derived analysis values.

### T2 confirmation population

- Source: COCO 2017 validation;
- population status:
  `FROZEN_T2_CONFIRMATION_POPULATION_NO_MODEL_OUTCOME`;
- initial analysis: 128 images per model;
- maximum frozen population: 256 images;
- same images for both models;
- model outputs and intervention outcomes were not used for population
  selection;
- source annotation SHA-256:
  `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`.

The frozen public-population manifest is included as
`audit/T2_CONFIRMATION_POPULATION_MANIFEST.json`.

## Checkpoints

### DETR

- model: DETR-R50-500;
- checkpoint SHA-256:
  `e632da11ec76ae67bac2f8579fbed3724e08dead7d200ca13e019b197784eadc`;
- weights are not redistributed.

### DINO

- model: DINO-R50, 4-scale, 12-epoch;
- checkpoint SHA-256:
  `0bcd6b0c33d60ed33461ce6f02ce5797a819c7c02eb7e15b76adfb6df307955a`;
- upstream source commit:
  `d84a491d41898b3befd8294d1cf2614661fc0953`;
- weights are not redistributed.

## Runtime

- Python 3.12.13;
- PyTorch 2.10.0+cu128;
- torchvision 0.25.0+cu128;
- NumPy 1.26.4;
- SciPy 1.17.1;
- original GPU: NVIDIA RTX 4090.

The compact aggregate reproduction requires only Python and NumPy and does
not load either model.

## Restricted and excluded material

- no V/F data or result is included;
- no image pixels are included;
- no model weights are included;
- no private authentication token, cloud credential, or user identity is
  included;
- full raw matrix receipts remain in the internal integrity archive.

