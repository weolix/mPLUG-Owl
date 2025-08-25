"""
Utility script for the Continual Learning IQA framework
Provides helper functions and tools for data preparation and model evaluation
"""

import os
import torch
import numpy as np
import json
import csv
from PIL import Image
# import matplotlib.pyplot as plt  # Optional dependency
from scipy.stats import spearmanr, pearsonr
import argparse

from image_quality_datasets import EVADataset, AVADataset, get_default_transforms
from lightweight_models import create_lightweight_model, create_multi_backbone_model
from continual_learning import ContinualLearningFramework


def prepare_eva_dataset(image_dir, output_csv, score_range=(1.0, 10.0)):
    """
    Prepare EVA dataset by creating a CSV file from images in a directory
    
    Args:
        image_dir: Directory containing images
        output_csv: Output CSV file path
        score_range: Range of scores to generate (for synthetic data)
    """
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'score'])
        
        for img_file in image_files:
            image_id = os.path.splitext(img_file)[0]  # Remove extension
            # Generate random score (replace with actual scores if available)
            score = np.random.uniform(score_range[0], score_range[1])
            writer.writerow([image_id, f"{score:.2f}"])
    
    print(f"Created EVA dataset CSV with {len(image_files)} images: {output_csv}")


def prepare_ava_dataset(image_dir, output_json, score_range=(1.0, 10.0)):
    """
    Prepare AVA dataset by creating a JSON file from images in a directory
    
    Args:
        image_dir: Directory containing images
        output_json: Output JSON file path
        score_range: Range of scores to generate (for synthetic data)
    """
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    data = {'files': []}
    for img_file in image_files:
        # Generate random score (replace with actual scores if available)
        score = np.random.uniform(score_range[0], score_range[1])
        data['files'].append({
            'image': img_file,
            'score': score
        })
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"Created AVA dataset JSON with {len(image_files)} images: {output_json}")


def analyze_dataset(dataset_path, dataset_type='eva'):
    """
    Analyze a dataset and print statistics
    
    Args:
        dataset_path: Path to dataset file (CSV for EVA, JSON for AVA)
        dataset_type: 'eva' or 'ava'
    """
    if dataset_type.lower() == 'eva':
        with open(dataset_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            scores = [float(row['score']) for row in reader]
    else:  # ava
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            scores = [float(item['score']) for item in data['files']]
    
    scores = np.array(scores)
    
    print(f"Dataset Analysis ({dataset_type.upper()}):")
    print(f"  Number of samples: {len(scores)}")
    print(f"  Score range: {scores.min():.2f} - {scores.max():.2f}")
    print(f"  Mean score: {scores.mean():.2f}")
    print(f"  Std deviation: {scores.std():.2f}")
    print(f"  Median score: {np.median(scores):.2f}")


def compare_models(eva_dataset_path=None, ava_dataset_path=None, eva_image_dir=None, ava_image_dir=None):
    """
    Compare different backbone models on the datasets
    """
    backbones = ['resnet18', 'mobilenet_v3_small', 'efficientnet_b0']
    results = {}
    
    print("Comparing backbone models...")
    print("=" * 50)
    
    for backbone in backbones:
        print(f"\nTesting {backbone}...")
        
        try:
            # Create model
            model = create_lightweight_model(
                backbone=backbone,
                embedding_dim=256,
                pretrained=False  # Faster for comparison
            )
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Test inference speed
            dummy_input = torch.randn(1, 3, 224, 224)
            
            import time
            start_time = time.time()
            with torch.no_grad():
                for _ in range(100):
                    _ = model(dummy_input)
            avg_inference_time = (time.time() - start_time) / 100
            
            results[backbone] = {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'inference_time': avg_inference_time
            }
            
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            print(f"  Avg inference time: {avg_inference_time*1000:.2f}ms")
            
        except Exception as e:
            print(f"  Error testing {backbone}: {e}")
            results[backbone] = None
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("Model Comparison Summary:")
    print("=" * 70)
    print(f"{'Model':<20} {'Parameters':<15} {'Inference (ms)':<15}")
    print("-" * 70)
    
    for backbone, result in results.items():
        if result:
            print(f"{backbone:<20} {result['total_params']:,<15} {result['inference_time']*1000:<15.2f}")
        else:
            print(f"{backbone:<20} {'Error':<15} {'Error':<15}")


def visualize_training_results(log_dir):
    """
    Visualize training results from tensorboard logs
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator
        
        print(f"Loading training logs from: {log_dir}")
        
        # This is a simplified version - in practice, you might want to use
        # tensorboard directly or a more sophisticated plotting library
        print("Use tensorboard to visualize training results:")
        print(f"tensorboard --logdir={log_dir}")
        
    except ImportError:
        print("Tensorboard not available. Install with: pip install tensorboard")


def evaluate_checkpoint(checkpoint_path, dataset_path, dataset_type='eva', image_dir=None):
    """
    Evaluate a saved model checkpoint
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model (assuming ResNet18 for simplicity)
    model = create_lightweight_model(backbone='resnet18', embedding_dim=256, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load dataset
    if dataset_type.lower() == 'eva':
        if not image_dir:
            print("Error: image_dir required for EVA dataset")
            return
        dataset = EVADataset(
            image_dir=image_dir,
            label_file=dataset_path,
            transform=get_default_transforms(224, train=False)
        )
    else:
        if not image_dir:
            print("Error: image_dir required for AVA dataset")
            return
        dataset = AVADataset(
            label_file=dataset_path,
            img_root=image_dir,
            transform=get_default_transforms(224, train=False)
        )
    
    # Evaluate
    from torch.utils.data import DataLoader
    from image_quality_datasets import collate_fn
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    predictions = []
    targets = []
    
    with torch.no_grad():
        for images, scores in dataloader:
            outputs = model(images)
            pred_scores = outputs['similarity_scores']
            
            predictions.extend(pred_scores.cpu().numpy())
            targets.extend(scores.cpu().numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Calculate metrics
    srcc, _ = spearmanr(predictions, targets)
    plcc, _ = pearsonr(predictions, targets)
    
    print(f"Evaluation Results:")
    print(f"  Dataset: {dataset_type.upper()}")
    print(f"  Samples: {len(predictions)}")
    print(f"  SRCC: {srcc:.4f}")
    print(f"  PLCC: {plcc:.4f}")
    
    # Print checkpoint info
    if 'eval_metrics' in checkpoint:
        print(f"  Original SRCC: {checkpoint['eval_metrics']['srcc']:.4f}")
        print(f"  Original PLCC: {checkpoint['eval_metrics']['plcc']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Continual Learning IQA Utilities')
    parser.add_argument('command', choices=['prepare_eva', 'prepare_ava', 'analyze', 'compare', 'evaluate'],
                       help='Command to run')
    parser.add_argument('--image_dir', type=str, help='Image directory')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--dataset_path', type=str, help='Dataset file path')
    parser.add_argument('--dataset_type', type=str, choices=['eva', 'ava'], default='eva',
                       help='Dataset type')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint file path')
    parser.add_argument('--log_dir', type=str, help='Log directory')
    
    args = parser.parse_args()
    
    if args.command == 'prepare_eva':
        if not args.image_dir or not args.output:
            print("Error: --image_dir and --output required for prepare_eva")
            return
        prepare_eva_dataset(args.image_dir, args.output)
    
    elif args.command == 'prepare_ava':
        if not args.image_dir or not args.output:
            print("Error: --image_dir and --output required for prepare_ava")
            return
        prepare_ava_dataset(args.image_dir, args.output)
    
    elif args.command == 'analyze':
        if not args.dataset_path:
            print("Error: --dataset_path required for analyze")
            return
        analyze_dataset(args.dataset_path, args.dataset_type)
    
    elif args.command == 'compare':
        compare_models()
    
    elif args.command == 'evaluate':
        if not args.checkpoint or not args.dataset_path:
            print("Error: --checkpoint and --dataset_path required for evaluate")
            return
        evaluate_checkpoint(args.checkpoint, args.dataset_path, args.dataset_type, args.image_dir)


if __name__ == '__main__':
    main()