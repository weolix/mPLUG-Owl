# Continual Learning for Image Quality Assessment (IQA)

This module provides a complete framework for continual learning on image quality assessment tasks using lightweight models from torchvision.

## Features

- **Two Dataset Classes**: EVADataset and AVADataset with PyTorch DataLoader compatibility
- **Lightweight Models**: ResNet18, MobileNetV3-Small, EfficientNet-B0 from torchvision
- **Configurable Continual Learning**: Multiple strategies including Naive, EWC, L2 regularization, and PackNet
- **Feature Embedding Optimization**: Normalized features with similarity-based scoring
- **Cosine Similarity Loss**: Custom loss function with ranking component

## Quick Start

### 1. Run the Demo

To test the framework with synthetic data:

```bash
cd mPLUG-Owl3
python demo_continual_iqa.py
```

### 2. Train on Real Data

To train on your own datasets:

```bash
python train_continual_iqa.py \
    --eva_image_dir /path/to/eva/images \
    --eva_label_file /path/to/eva/labels.csv \
    --ava_image_dir /path/to/ava/images \
    --ava_label_file /path/to/ava/labels.json \
    --backbone resnet18 \
    --cl_strategy ewc \
    --epochs 10 \
    --batch_size 32
```

## Dataset Format

### EVA Dataset
- **Images**: JPEG files in a directory
- **Labels**: CSV file with columns: `image_id`, `score`
- Example CSV:
```csv
image_id,score
image_001,7.5
image_002,6.2
```

### AVA Dataset
- **Images**: JPEG files in a directory
- **Labels**: JSON file with structure:
```json
{
  "files": [
    {"image": "ava_001.jpg", "score": 8.1},
    {"image": "ava_002.jpg", "score": 5.7}
  ]
}
```

## Model Architecture

The framework uses lightweight backbone models with the following architecture:

1. **Backbone**: ResNet18/MobileNetV3/EfficientNet (pretrained on ImageNet)
2. **Feature Projection**: Maps backbone features to embedding space
3. **Positive Vector**: Learnable vector for similarity computation
4. **Scoring**: Cosine similarity between normalized embeddings and positive vector

## Continual Learning Strategies

### 1. Naive
- Simple sequential training without any continual learning techniques
- Good baseline for comparison

### 2. Elastic Weight Consolidation (EWC)
- Penalizes changes to important parameters from previous tasks
- Uses Fisher Information Matrix to identify important parameters
- Configurable λ parameter (default: 1000)

### 3. L2 Regularization
- Regularizes parameters towards previous task values
- Simple but effective approach
- Configurable λ parameter (default: 0.01)

### 4. PackNet
- Prunes network for each task to prevent interference
- Allocates different parameters for different tasks
- Configurable pruning ratio (default: 0.2)

## Configuration

Edit `config.yaml` to customize training parameters:

```yaml
model:
  backbone: "resnet18"
  embedding_dim: 256
  pretrained: true

training:
  epochs: 10
  batch_size: 32
  learning_rate: 0.0001

continual_learning:
  strategy: "ewc"
  strategy_params:
    ewc_lambda: 1000
```

## Usage Examples

### Basic Usage

```python
from image_quality_datasets import EVADataset, AVADataset
from lightweight_models import create_lightweight_model
from continual_learning import ContinualLearningFramework

# Create datasets
eva_dataset = EVADataset(image_dir="./eva_images", label_file="./eva_labels.csv")
ava_dataset = AVADataset(label_file="./ava_labels.json", img_root="./ava_images")

# Create model
model = create_lightweight_model(backbone='resnet18', embedding_dim=256)

# Create continual learning framework
cl_framework = ContinualLearningFramework(model=model, strategy='ewc')

# Training loop (simplified)
for task_id, dataset in enumerate([eva_dataset, ava_dataset]):
    cl_framework.before_task(task_id=task_id)
    # ... training code ...
    cl_framework.after_task(task_id=task_id, dataloader=dataloader)
```

### Custom Strategy

```python
from continual_learning import ContinualLearningStrategy

class MyCustomStrategy(ContinualLearningStrategy):
    def compute_loss(self, model, batch, task_id, criterion, **kwargs):
        # Implement your custom loss computation
        pass

# Use custom strategy
cl_framework = ContinualLearningFramework(model=model, strategy=MyCustomStrategy())
```

## Requirements

- torch >= 1.7.0
- torchvision >= 0.8.0
- numpy
- scipy
- scikit-learn
- Pillow
- tqdm
- tensorboard

## Files

- `image_quality_datasets.py`: Dataset classes for EVA and AVA
- `lightweight_models.py`: Lightweight backbone models
- `continual_learning.py`: Continual learning strategies
- `train_continual_iqa.py`: Main training script
- `demo_continual_iqa.py`: Demo with synthetic data
- `config.yaml`: Configuration file

## Output

The training script produces:
- **Checkpoints**: Model weights saved after each task
- **Logs**: TensorBoard logs with training metrics
- **Metrics**: SRCC and PLCC correlation scores

## Tips

1. **Data Preprocessing**: Images are automatically resized and normalized
2. **Memory Management**: Use smaller batch sizes if running out of memory
3. **Hyperparameter Tuning**: Adjust learning rate and continual learning parameters
4. **Model Selection**: Try different backbones for your specific use case
5. **Evaluation**: Monitor both current task performance and forgetting of previous tasks

## License

This code is part of the Q-Owl project. Please refer to the project's license.