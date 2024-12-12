# PELICAN Network for ATLAS Top Tagging

A modified version of the Permutation Equivariant, Lorentz Invariant/Covariant Aggregator Network (PELICAN) adapted for the ATLAS Top Tagging Open Dataset.

Original PELICAN paper: [arXiv:2211.00454](https://arxiv.org/abs/2211.00454)

## Overview

This repository contains a modified version of PELICAN specifically adapted to work with the ATLAS Top Tagging Open Dataset. The key modification is the integration of per-event weights to handle the unique characteristics of the ATLAS dataset, particularly the reweighting of background jet pT spectrum to match the signal spectrum.

## ATLAS Top Tagging Dataset

Dataset: [ATLAS Top Tagging Open Dataset](http://opendata.cern.ch/record/15013)

Key features of the dataset:
- Generated with GEANT4-based detector simulation
- Uses state-of-the-art jet reconstruction methods
- Includes training weights to handle background pT spectrum reweighting
- Only public top tagging dataset with this level of simulation fidelity

### Dataset Characteristics

The dataset addresses two key challenges:
1. Unphysical bumps in background jet pT distribution
2. Need to prevent background sculpting in ML training

These are handled through the included training weights that:
- Reweight background jet pT spectrum to match signal spectrum
- Help prevent the network from learning pT-based discrimination
- Enable more robust top tagging performance

## Modifications from Original PELICAN

Key changes in this version:
1. Integration of per-event weights in loss calculation
   - Modified `trainclassifier` file: Changed loss function to `nn.crossentropy(reduction="none")`
   - Updated `src.trainer.trainer`: Added weight multiplication in train function
2. Adapted for ATLAS dataset structure and format
3. Handling of ATLAS-specific event weights

## Requirements

Same as original PELICAN:
* Python >=3.9
* PyTorch >=1.10
* h5py
* colorlog
* scikit-learn
* tensorboard (for --summarize)
* optuna (for hyperparameter optimization)
* psycopg2-binary (for distributed optuna)

## Usage

### Basic Training Command

```bash
python train_pelican_classifier.py \
    --datadir=/path/to/atlas/data \
    --target=is_signal \
    --nobj=80 \
    --nobj-avg=49 \
    --num-epoch=35 \
    --num-train=60000 \
    --num-valid=60000 \
    --batch-size=64 \
    --prefix=atlas_classifier \
    --optim=adamw \
    --activation=leakyrelu \
    --factorize \
    --lr-decay-type=warm \
    --lr-init=0.0025 \
    --lr-final=1e-6 \
    --drop-rate=0.05 \
    --drop-rate-out=0.05 \
    --weight-decay=0.005
```

### Key Arguments

All original PELICAN arguments plus:
- Standard PELICAN arguments remain valid
- Dataset should include the weights column from ATLAS
- No additional arguments needed as weights are handled automatically

### Data Format

Expected format:
- Input 4-momenta under 'Pmu'
- Classification labels under 'is_signal'
- ATLAS event weights included in dataset
- Files should be split into train.h5, valid.h5, and test.h5

## Output and Evaluation

Same as original PELICAN:
- Log files in `log/` folder
- Tensorboard summaries (if --summarize is used)
- Model checkpoints in `model/`
- Predictions and ROC curves in `predict/`
- Metrics in CSV files for best and final models

## Citation

If you use this code, please cite both:
1. Original PELICAN paper: [arXiv:2211.00454](https://arxiv.org/abs/2211.00454)
2. ATLAS Top Tagging Dataset: [ATLAS Collaboration (2021)](http://opendata.cern.ch/record/15013)


## Acknowledgments

- Original PELICAN authors
- ATLAS Collaboration for the dataset
- [Original PELICAN acknowledgments maintained]

## Authors

Pradyun Hebbar, LBNL

Based on PELICAN by:
- Alexander Bogatskiy, Flatiron Institute
- Jan T. Offermann, University of Chicago
- Timothy Hoffman, University of Chicago
- Xiaoyang Liu, University of Chicago
