<div align="center">

<h2>Adaptive Logit Weighting for Zero-Shot Quality Assessment with MLLMs</h2>

</div>

![pipline figure](mPLUG-Owl3/radar.png "Magic Gardens")

- Developed on **mPLUG-Owl3**

## Method

Our approach leverages mPLUG-Owl3 for zero-shot quality assessment through **Adaptive Logit Weighting**. The method consists of the following key components:

### Overview
The framework employs a multi-modal large language model (mPLUG-Owl3) to assess image quality without requiring task-specific training. Instead of relying on traditional fine-tuning approaches, we utilize the model's inherent language understanding capabilities to perform quality assessment through strategic prompt design and logit manipulation.

## Experimental Results

### Comparison with Training-Free SOTA Methods

<div align="center">

**Table 1: Comparison with training-free SOTA IQA methods. The best results are in bold.**

| Dataset | LIVE Challenge |  | KonIQ-10k |  | AGIQA-3K |  | KADID-10k |  | SPAQ |  |
|---------|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| Method | SRCC | PLCC | SRCC | PLCC | SRCC | PLCC | SRCC | PLCC | SRCC | PLCC |
| BIQI | 0.364 | 0.447 | 0.559 | 0.616 | 0.390 | 0.423 | 0.338 | 0.405 | 0.591 | 0.549 |
| BLIINDS-II | 0.090 | 0.107 | 0.585 | 0.598 | 0.454 | 0.510 | 0.224 | 0.313 | 0.317 | 0.326 |
| BRISQUE | 0.561 | 0.598 | 0.705 | 0.707 | 0.493 | 0.533 | 0.330 | 0.370 | 0.484 | 0.481 |
| NIQE | 0.463 | 0.491 | 0.551 | 0.488 | 0.529 | 0.520 | 0.379 | 0.389 | 0.703 | 0.671 |
| CLIP-IQA | 0.612 | 0.594 | 0.695 | 0.727 | 0.658 | 0.714 | 0.500 | 0.520 | 0.738 | 0.735 |
| MDFS | 0.482 | 0.536 | 0.733 | 0.712 | 0.672 | 0.676 | 0.598 | 0.594 | 0.741 | 0.718 |
| Q-Debias | 0.794 | 0.790 | 0.838 | 0.863 | 0.717 | 0.753 | 0.700 | 0.713 | 0.867 | 0.826 |
| Dog-IQA | 0.756 | 0.752 | 0.819 | 0.811 | **0.823** | 0.797 | 0.612 | 0.624 | 0.902 | 0.897 |
| **Ours** | **0.856** | **0.883** | **0.886** | **0.916** | 0.770 | **0.827** | **0.791** | **0.787** | **0.906** | **0.911** |

</div>

### Performance Comparison with Different Prompt Styles

<div align="center">

**Table 2: Performance on (SRCC+PLCC)/2 of popular methods implemented with the same prompts and based on mPLUG-Owl3. Best results are in bold.**

| Category | Dataset | Q-Bench style | Q-Align style | **Ours** |
|----------|---------|:-------------:|:-------------:|:--------:|
| **Artificial** | KADID-10k | 0.649 | 0.651 | **0.779** |
|  | LIVE | 0.751 | 0.777 | **0.887** |
|  | CSIQ | 0.771 | 0.748 | **0.828** |
| **UGC** | KonIQ-10k | 0.748 | 0.847 | **0.900** |
|  | SPAQ | 0.828 | 0.900 | **0.913** |
|  | LIVEC | 0.676 | 0.831 | **0.869** |
| **AIGC** | AGIQA-3K | 0.722 | 0.771 | **0.795** |

</div>

Our method consistently outperforms existing training-free approaches across multiple datasets and categories, demonstrating the effectiveness of the adaptive logit weighting strategy for zero-shot quality assessment.

## Usage

### Requirements
```bash
cd mPLUG-Owl3
pip install -r requirements.txt
```

### Configuration

The method supports both Image Quality Assessment (IQA) and Image Aesthetic Assessment (IAA). Configuration files are available:
- `mPLUG-Owl3/iqa.yml` - Configuration for quality assessment datasets
- `mPLUG-Owl3/iaa.yml` - Configuration for aesthetic assessment datasets
