"""
Complete example of using the Continual Learning IQA framework
This script demonstrates the full workflow from data preparation to training and evaluation
"""

import os
import tempfile
import shutil
import numpy as np
from PIL import Image
import csv
import json
import torch
from torch.utils.data import DataLoader

# Import our modules
from image_quality_datasets import EVADataset, AVADataset, get_default_transforms, collate_fn
from lightweight_models import create_lightweight_model
from continual_learning import ContinualLearningFramework
from utils_continual_iqa import prepare_eva_dataset, prepare_ava_dataset, analyze_dataset


def create_sample_dataset(dataset_type='eva', num_samples=100, image_size=224):
    """Create a sample dataset for demonstration"""
    temp_dir = tempfile.mkdtemp()
    image_dir = os.path.join(temp_dir, 'images')
    os.makedirs(image_dir, exist_ok=True)
    
    print(f"Creating sample {dataset_type.upper()} dataset with {num_samples} images...")
    
    # Create synthetic images with different quality patterns
    for i in range(num_samples):
        # Create images with varying quality characteristics
        if i < num_samples // 3:
            # High quality: sharp, good contrast
            noise_level = 0.1
            blur_level = 0
        elif i < 2 * num_samples // 3:
            # Medium quality: some noise, slight blur
            noise_level = 0.3
            blur_level = 1
        else:
            # Low quality: noisy, blurred
            noise_level = 0.5
            blur_level = 2
        
        # Generate base image
        image_array = np.random.randint(50, 200, (image_size, image_size, 3), dtype=np.uint8)
        
        # Add noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * 50, image_array.shape)
            image_array = np.clip(image_array + noise, 0, 255).astype(np.uint8)
        
        # Add blur (simple box filter)
        if blur_level > 0:
            from scipy import ndimage
            image_array = ndimage.uniform_filter(image_array, size=(blur_level, blur_level, 1))
        
        image = Image.fromarray(image_array)
        image_filename = f'sample_{i:04d}.jpg'
        image.save(os.path.join(image_dir, image_filename))
        
        # Generate quality score based on quality level
        if i < num_samples // 3:
            score = np.random.uniform(7.0, 10.0)  # High quality
        elif i < 2 * num_samples // 3:
            score = np.random.uniform(4.0, 7.0)   # Medium quality
        else:
            score = np.random.uniform(1.0, 4.0)   # Low quality
    
    # Create label file
    if dataset_type == 'eva':
        label_file = os.path.join(temp_dir, 'labels.csv')
        prepare_eva_dataset(image_dir, label_file, score_range=(1.0, 10.0))
    else:  # ava
        label_file = os.path.join(temp_dir, 'labels.json')
        prepare_ava_dataset(image_dir, label_file, score_range=(1.0, 10.0))
    
    return image_dir, label_file, temp_dir


def train_simple_example():
    """Train a simple example with synthetic data"""
    print("="*60)
    print("CONTINUAL LEARNING IQA - COMPLETE EXAMPLE")
    print("="*60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create synthetic datasets
    print("\n1. Creating synthetic datasets...")
    eva_image_dir, eva_label_file, eva_temp_dir = create_sample_dataset('eva', num_samples=200)
    ava_image_dir, ava_label_file, ava_temp_dir = create_sample_dataset('ava', num_samples=200)
    
    # Analyze datasets
    print("\n2. Analyzing datasets...")
    analyze_dataset(eva_label_file, 'eva')
    analyze_dataset(ava_label_file, 'ava')
    
    # Create datasets
    print("\n3. Creating PyTorch datasets...")
    eva_dataset = EVADataset(
        image_dir=eva_image_dir,
        label_file=eva_label_file,
        transform=get_default_transforms(224, train=True)
    )
    
    ava_dataset = AVADataset(
        label_file=ava_label_file,
        img_root=ava_image_dir,
        transform=get_default_transforms(224, train=True)
    )
    
    print(f"EVA dataset: {len(eva_dataset)} samples")
    print(f"AVA dataset: {len(ava_dataset)} samples")
    
    # Create model
    print("\n4. Creating lightweight model...")
    model = create_lightweight_model(
        backbone='resnet18',
        embedding_dim=256,
        pretrained=True
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created: {total_params:,} total parameters, {trainable_params:,} trainable")
    
    # Test different continual learning strategies
    strategies = ['naive', 'ewc', 'l2']
    results = {}
    
    for strategy in strategies:
        print(f"\n5. Training with {strategy.upper()} strategy...")
        
        # Create fresh model for each strategy
        model = create_lightweight_model(
            backbone='resnet18',
            embedding_dim=256,
            pretrained=True
        ).to(device)
        
        # Create continual learning framework
        cl_framework = ContinualLearningFramework(
            model=model,
            strategy=strategy,
            strategy_kwargs={'ewc_lambda': 100} if strategy == 'ewc' else {}
        )
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Training settings
        epochs_per_task = 3  # Reduced for demo
        batch_size = 16
        
        datasets = [('EVA', eva_dataset), ('AVA', ava_dataset)]
        task_results = {}
        
        for task_id, (dataset_name, dataset) in enumerate(datasets):
            print(f"  Task {task_id}: {dataset_name}")
            
            # Prepare for task
            cl_framework.before_task(task_id=task_id)
            
            # Create dataloader
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn
            )
            
            # Train for a few epochs
            model.train()
            task_losses = []
            
            for epoch in range(epochs_per_task):
                epoch_loss = 0
                num_batches = 0
                
                for batch_idx, batch in enumerate(dataloader):
                    if batch_idx >= 5:  # Limit batches for demo
                        break
                    
                    images, targets = batch
                    images, targets = images.to(device), targets.to(device)
                    
                    optimizer.zero_grad()
                    
                    # Compute loss using continual learning framework
                    loss_dict = cl_framework.compute_loss(
                        batch=(images, targets),
                        criterion=torch.nn.MSELoss(),
                        task_id=task_id
                    )
                    
                    loss = loss_dict['loss']
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                
                avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
                task_losses.append(avg_loss)
                print(f"    Epoch {epoch+1}: Loss = {avg_loss:.4f}")
            
            # Evaluate on current task
            model.eval()
            predictions = []
            targets_list = []
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    if batch_idx >= 10:  # Limit for demo
                        break
                    
                    images, targets = batch
                    images, targets = images.to(device), targets.to(device)
                    
                    outputs = model(images)
                    pred_scores = outputs['similarity_scores']
                    
                    predictions.extend(pred_scores.cpu().numpy())
                    targets_list.extend(targets.cpu().numpy())
            
            # Calculate correlation
            from scipy.stats import spearmanr, pearsonr
            if len(predictions) > 1:
                srcc, _ = spearmanr(predictions, targets_list)
                plcc, _ = pearsonr(predictions, targets_list)
            else:
                srcc, plcc = 0, 0
            
            task_results[dataset_name] = {
                'final_loss': task_losses[-1] if task_losses else 0,
                'srcc': srcc,
                'plcc': plcc
            }
            
            print(f"    Final metrics: SRCC={srcc:.3f}, PLCC={plcc:.3f}")
            
            # Finish task
            cl_framework.after_task(task_id=task_id, dataloader=dataloader)
        
        results[strategy] = task_results
    
    # Print comparison
    print(f"\n6. Results Comparison:")
    print("="*80)
    print(f"{'Strategy':<10} {'Dataset':<10} {'SRCC':<10} {'PLCC':<10} {'Loss':<10}")
    print("-"*80)
    
    for strategy, strategy_results in results.items():
        for dataset_name, metrics in strategy_results.items():
            print(f"{strategy:<10} {dataset_name:<10} {metrics['srcc']:<10.3f} {metrics['plcc']:<10.3f} {metrics['final_loss']:<10.3f}")
    
    # Cleanup
    print(f"\n7. Cleaning up temporary files...")
    shutil.rmtree(eva_temp_dir)
    shutil.rmtree(ava_temp_dir)
    
    print("\nExample completed successfully!")
    print("The framework is ready for use with your own datasets.")


def test_feature_similarity():
    """Test the feature similarity mechanism"""
    print("\n" + "="*60)
    print("TESTING FEATURE SIMILARITY MECHANISM")
    print("="*60)
    
    # Create model
    model = create_lightweight_model(backbone='resnet18', embedding_dim=256, pretrained=False)
    
    # Create dummy images with different characteristics
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(dummy_images)
    
    print(f"Input shape: {dummy_images.shape}")
    print(f"Embeddings shape: {outputs['embeddings'].shape}")
    print(f"Normalized embeddings shape: {outputs['embeddings_normalized'].shape}")
    print(f"Similarity scores shape: {outputs['similarity_scores'].shape}")
    print(f"Direct scores shape: {outputs['direct_scores'].shape}")
    print(f"Positive vector shape: {outputs['positive_vector'].shape}")
    
    # Show similarity scores
    similarity_scores = outputs['similarity_scores'].numpy()
    print(f"\nSimilarity scores: {similarity_scores}")
    print(f"Score range: {similarity_scores.min():.3f} to {similarity_scores.max():.3f}")
    
    # Test normalization
    embeddings = outputs['embeddings']
    embeddings_norm = outputs['embeddings_normalized']
    
    original_norms = torch.norm(embeddings, p=2, dim=1)
    normalized_norms = torch.norm(embeddings_norm, p=2, dim=1)
    
    print(f"\nOriginal embedding norms: {original_norms.numpy()}")
    print(f"Normalized embedding norms: {normalized_norms.numpy()}")
    print(f"All norms ≈ 1.0: {torch.allclose(normalized_norms, torch.ones_like(normalized_norms), atol=1e-6)}")


if __name__ == '__main__':
    try:
        # Test the feature similarity mechanism
        test_feature_similarity()
        
        # Run the complete training example
        train_simple_example()
        
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install: pip install scipy")
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()