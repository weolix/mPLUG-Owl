<div align="center">

<h2>Adaptive Logit Weighting for Zero-Shot Quality Assessment with MLLMs</h2>

</div>

![pipline figure](mPLUG-Owl3/radar.png "Magic Gardens")

- Developed on **mPLUG-Owl3**

## Method

Our approach leverages mPLUG-Owl3 for zero-shot quality assessment through **Adaptive Logit Weighting**. The method consists of the following key components:

### Overview
The framework employs a multi-modal large language model (mPLUG-Owl3) to assess image quality without requiring task-specific training. Instead of relying on traditional fine-tuning approaches, we utilize the model's inherent language understanding capabilities to perform quality assessment through strategic prompt design and logit manipulation.

### Adaptive Logit Weighting Strategy

1. **Quality-aware Prompting**: We design task-specific prompts that guide the model to generate quality-related responses:
   - **IQA (Image Quality Assessment)**: "Taking into account the details and the rationality of the image, how would you rate the quality of this image?"
   - **IAA (Image Aesthetic Assessment)**: "Considering its artistic composition, color harmony, and overall visual appeal, use an adjective to describe the aesthetic quality of this image?"

2. **Top-k Logit Extraction**: From the model's output logits at the last token position, we extract the top-k (k=100) highest probability tokens and their corresponding logit values.

3. **Semantic Embedding Construction**: We construct quality-aware embeddings by:
   - Defining positive quality words (e.g., "excellent", "superb", "outstanding", "stunning")
   - Defining negative quality words (e.g., "poor", "terrible", "awful", "blurry")
   - Computing mean embeddings for positive and negative word sets
   - Creating a quality direction vector: `val_embed = positive_embed - negative_embed`

4. **Adaptive Weighting**: For each top-k token:
   - Extract token embeddings using the model's embedding layer
   - Compute cosine similarity between token embeddings and the quality direction vector
   - Weight the corresponding logits using these similarity scores
   - Sum the weighted logits to obtain the final quality score

### Technical Implementation

The core algorithm can be summarized as:

```python
# Extract top-k logits and indices
topk_logits, topk_indices = torch.topk(last_token_logits, k=100, dim=-1)

# Get embeddings for top-k tokens
embedding_layer = model.get_input_embeddings()
topk_embeddings = embedding_layer(topk_indices)

# Compute similarity weights
weights = F.cosine_similarity(topk_embeddings, quality_embed, dim=-1)

# Generate final quality score
weighted_logits = topk_logits * weights
quality_score = torch.sum(weighted_logits, dim=-1)
```

This approach enables the model to automatically focus on quality-relevant tokens in the vocabulary space, providing interpretable and effective quality assessment without requiring domain-specific training data.

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

### Quick Start

```python
import torch
from modelscope import AutoConfig, AutoModel, AutoTokenizer
from PIL import Image

# Load model
model_path = 'iic/mPLUG-Owl3-7B-241101'
model = AutoModel.from_pretrained(
    model_path, 
    attn_implementation='flash_attention_2', 
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True
).cuda().eval()

tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = model.init_processor(tokenizer)

# Quality Assessment
from mPLUG-Owl3.owl3_zeroshot import MultimodalQualityEvaluator, get_embed

evaluator = MultimodalQualityEvaluator(task="IQA", model_path=model_path)
quality_embed = get_embed(evaluator, device="cuda", TASK="IQA")

# Evaluate an image
image = Image.open("your_image.jpg").convert('RGB')
image_data = [{"info": {"name": "test.jpg"}, "data": image}]

with torch.no_grad():
    outputs = evaluator(image_or_video=image_data)
    logits = outputs.logits
    # Apply adaptive logit weighting for quality score
    # (See owl3_zeroshot.py for complete implementation)
```

### Configuration

The method supports both Image Quality Assessment (IQA) and Image Aesthetic Assessment (IAA). Configuration files are available:
- `mPLUG-Owl3/iqa.yml` - Configuration for quality assessment datasets
- `mPLUG-Owl3/iaa.yml` - Configuration for aesthetic assessment datasets

### Implementation Details

For complete implementation details, please refer to `mPLUG-Owl3/owl3_zeroshot.py` which contains:
- `MultimodalQualityEvaluator` class for model initialization and inference
- `get_embed()` function for creating quality-aware embeddings  
- Dataset classes for various quality assessment benchmarks
- Evaluation utilities and metrics computation
