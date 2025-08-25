"""
Demo script to test the continual learning framework with synthetic data
"""

import os
import torch
import numpy as np
from PIL import Image
import tempfile
import csv
import json
from torch.utils.data import DataLoader

from image_quality_datasets import EVADataset, AVADataset, get_default_transforms, collate_fn
from lightweight_models import create_lightweight_model
from continual_learning import ContinualLearningFramework


def create_synthetic_eva_dataset(num_samples=100, image_size=224):
    """Create synthetic EVA dataset for testing"""
    temp_dir = tempfile.mkdtemp()
    image_dir = os.path.join(temp_dir, 'images')
    os.makedirs(image_dir, exist_ok=True)
    
    # Create synthetic images and labels
    labels = []
    for i in range(num_samples):
        # Create random image
        image = Image.fromarray(np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8))
        image_path = os.path.join(image_dir, f'image_{i:04d}.jpg')
        image.save(image_path)
        
        # Create random quality score
        score = np.random.uniform(1.0, 10.0)
        labels.append({'image_id': f'image_{i:04d}', 'score': score})
    
    # Save labels to CSV
    label_file = os.path.join(temp_dir, 'labels.csv')
    with open(label_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['image_id', 'score'])
        writer.writeheader()
        writer.writerows(labels)
    
    return image_dir, label_file, temp_dir


def create_synthetic_ava_dataset(num_samples=100, image_size=224):
    """Create synthetic AVA dataset for testing"""
    temp_dir = tempfile.mkdtemp()
    image_dir = os.path.join(temp_dir, 'images')
    os.makedirs(image_dir, exist_ok=True)
    
    # Create synthetic images and labels
    data = {'files': []}
    for i in range(num_samples):
        # Create random image
        image = Image.fromarray(np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8))
        image_filename = f'ava_image_{i:04d}.jpg'
        image_path = os.path.join(image_dir, image_filename)
        image.save(image_path)
        
        # Create random quality score
        score = np.random.uniform(1.0, 10.0)
        data['files'].append({
            'image': image_filename,
            'score': score
        })
    
    # Save labels to JSON
    label_file = os.path.join(temp_dir, 'labels.json')
    with open(label_file, 'w', encoding='utf-8') as f:
        json.dump(data, f)
    
    return image_dir, label_file, temp_dir


def test_datasets():
    """Test dataset classes"""
    print("Testing dataset classes...")
    
    # Test EVA dataset
    eva_image_dir, eva_label_file, eva_temp_dir = create_synthetic_eva_dataset(50, 224)
    eva_dataset = EVADataset(
        image_dir=eva_image_dir,
        label_file=eva_label_file,
        transform=get_default_transforms(224, train=True)
    )
    
    print(f"EVA Dataset: {len(eva_dataset)} samples")
    print(f"Score range: {eva_dataset.vmin:.2f} - {eva_dataset.vmax:.2f}")
    
    # Test sample from EVA dataset
    image, score = eva_dataset[0]
    print(f"EVA sample shape: {image.shape}, score: {score.item():.2f}")
    
    # Test AVA dataset
    ava_image_dir, ava_label_file, ava_temp_dir = create_synthetic_ava_dataset(50, 224)
    ava_dataset = AVADataset(
        label_file=ava_label_file,
        img_root=ava_image_dir,
        transform=get_default_transforms(224, train=True)
    )
    
    print(f"AVA Dataset: {len(ava_dataset)} samples")
    print(f"Score range: {ava_dataset.vmin:.2f} - {ava_dataset.vmax:.2f}")
    
    # Test sample from AVA dataset
    image, score = ava_dataset[0]
    print(f"AVA sample shape: {image.shape}, score: {score.item():.2f}")
    
    return eva_dataset, ava_dataset, [eva_temp_dir, ava_temp_dir]


def test_model():
    """Test lightweight model"""
    print("\nTesting lightweight model...")
    
    model = create_lightweight_model(
        backbone='resnet18',
        embedding_dim=256,
        pretrained=False  # Faster for testing
    )
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    with torch.no_grad():
        outputs = model(dummy_input)
    
    print(f"Model outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    print(f"Embedding dimension: {model.get_embedding_dim()}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    return model


def test_continual_learning():
    """Test continual learning framework"""
    print("\nTesting continual learning framework...")
    
    # Create datasets
    eva_dataset, ava_dataset, temp_dirs = test_datasets()
    
    # Create model
    model = test_model()
    
    # Test different strategies
    strategies = ['naive', 'ewc', 'l2']
    
    for strategy in strategies:
        print(f"\nTesting {strategy} strategy...")
        
        # Create continual learning framework
        cl_framework = ContinualLearningFramework(
            model=model,
            strategy=strategy,
            strategy_kwargs={'ewc_lambda': 100} if strategy == 'ewc' else {}
        )
        
        # Test with both datasets
        datasets = [('EVA', eva_dataset), ('AVA', ava_dataset)]
        
        for task_id, (dataset_name, dataset) in enumerate(datasets):
            print(f"  Task {task_id}: {dataset_name}")
            
            # Prepare for task
            cl_framework.before_task(task_id=task_id)
            
            # Create small dataloader for testing
            dataloader = DataLoader(
                dataset,
                batch_size=8,
                shuffle=True,
                collate_fn=collate_fn
            )
            
            # Test a few batches
            criterion = torch.nn.MSELoss()
            
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= 2:  # Only test a few batches
                    break
                
                loss_dict = cl_framework.compute_loss(
                    batch=batch,
                    criterion=criterion,
                    task_id=task_id
                )
                
                print(f"    Batch {batch_idx}: Loss = {loss_dict['loss'].item():.4f}")
            
            # Finish task
            cl_framework.after_task(task_id=task_id, dataloader=dataloader)
    
    # Cleanup
    import shutil
    for temp_dir in temp_dirs:
        shutil.rmtree(temp_dir)
    
    print("\nContinual learning framework test completed successfully!")


def main():
    """Run all tests"""
    print("="*60)
    print("Running Continual Learning IQA Demo")
    print("="*60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Test datasets
        test_datasets()
        
        # Test model
        test_model()
        
        # Test continual learning
        test_continual_learning()
        
        print("\n" + "="*60)
        print("All tests completed successfully!")
        print("The continual learning framework is ready to use.")
        print("="*60)
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()