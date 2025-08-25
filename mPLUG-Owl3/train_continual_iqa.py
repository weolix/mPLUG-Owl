"""
Main training script for continual learning with two image quality datasets
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import random

from image_quality_datasets import EVADataset, AVADataset, get_default_transforms, collate_fn
from lightweight_models import create_lightweight_model, create_multi_backbone_model
from continual_learning import ContinualLearningFramework


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def cosine_similarity_loss(predictions, targets, margin=0.1):
    """
    Cosine similarity loss with margin
    """
    # Normalize targets to [-1, 1] range for better similarity computation
    targets_normalized = 2 * (targets - targets.min()) / (targets.max() - targets.min()) - 1
    
    # Basic MSE loss for similarity scores
    mse_loss = nn.MSELoss()(predictions, targets_normalized)
    
    # Add ranking loss to maintain relative ordering
    batch_size = predictions.size(0)
    ranking_loss = 0
    count = 0
    
    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            if targets[i] > targets[j]:
                # predictions[i] should be > predictions[j]
                ranking_loss += torch.clamp(margin - (predictions[i] - predictions[j]), min=0)
            elif targets[i] < targets[j]:
                # predictions[j] should be > predictions[i]
                ranking_loss += torch.clamp(margin - (predictions[j] - predictions[i]), min=0)
            count += 1
    
    if count > 0:
        ranking_loss /= count
    
    return mse_loss + 0.1 * ranking_loss


def evaluate_model(model, dataloader, device, dataset_name=""):
    """Evaluate model performance"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
            images, targets = images.to(device), targets.to(device)
            
            outputs = model(images)
            predictions = outputs['similarity_scores']
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Calculate correlation metrics
    srcc, _ = spearmanr(all_predictions, all_targets)
    plcc, _ = pearsonr(all_predictions, all_targets)
    
    return {
        'srcc': srcc,
        'plcc': plcc,
        'predictions': all_predictions,
        'targets': all_targets
    }


def train_epoch(model, 
                cl_framework, 
                dataloader, 
                optimizer, 
                device, 
                task_id,
                epoch,
                writer=None):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    pbar = tqdm(dataloader, desc=f"Task {task_id}, Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        images, targets = batch
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Compute loss using continual learning framework
        loss_dict = cl_framework.compute_loss(
            batch=(images, targets),
            criterion=cosine_similarity_loss,
            task_id=task_id
        )
        
        loss = loss_dict['loss']
        predictions = loss_dict['predictions']
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_predictions.extend(predictions.detach().cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
        
        # Log to tensorboard
        if writer is not None:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar(f'Task_{task_id}/Loss/Train', loss.item(), global_step)
            
            if 'current_loss' in loss_dict:
                writer.add_scalar(f'Task_{task_id}/CurrentLoss/Train', loss_dict['current_loss'].item(), global_step)
            if 'ewc_loss' in loss_dict:
                writer.add_scalar(f'Task_{task_id}/EWCLoss/Train', loss_dict['ewc_loss'].item(), global_step)
            if 'l2_loss' in loss_dict:
                writer.add_scalar(f'Task_{task_id}/L2Loss/Train', loss_dict['l2_loss'].item(), global_step)
    
    # Calculate epoch metrics
    avg_loss = total_loss / len(dataloader)
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    srcc, _ = spearmanr(all_predictions, all_targets)
    plcc, _ = pearsonr(all_predictions, all_targets)
    
    return {
        'loss': avg_loss,
        'srcc': srcc,
        'plcc': plcc
    }


def main():
    parser = argparse.ArgumentParser(description='Continual Learning for Image Quality Assessment')
    parser.add_argument('--config', type=str, default='config.yaml', help='Configuration file')
    parser.add_argument('--eva_image_dir', type=str, help='EVA dataset image directory')
    parser.add_argument('--eva_label_file', type=str, help='EVA dataset label file')
    parser.add_argument('--ava_image_dir', type=str, help='AVA dataset image directory')
    parser.add_argument('--ava_label_file', type=str, help='AVA dataset label file')
    parser.add_argument('--backbone', type=str, default='resnet18', 
                       choices=['resnet18', 'mobilenet_v3_small', 'efficientnet_b0'],
                       help='Backbone model')
    parser.add_argument('--cl_strategy', type=str, default='naive',
                       choices=['naive', 'ewc', 'l2', 'packnet'],
                       help='Continual learning strategy')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs per task')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--image_size', type=int, default=224, help='Input image size')
    parser.add_argument('--embedding_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Save directory')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Log directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Initialize tensorboard writer
    writer = SummaryWriter(args.log_dir)
    
    # Create datasets (you'll need to provide actual paths)
    print("Creating datasets...")
    
    # For demo purposes, create mock datasets if paths not provided
    if args.eva_image_dir and args.eva_label_file:
        eva_train_dataset = EVADataset(
            image_dir=args.eva_image_dir,
            label_file=args.eva_label_file,
            image_size=args.image_size,
            transform=get_default_transforms(args.image_size, train=True)
        )
    else:
        print("Warning: EVA dataset paths not provided, skipping EVA dataset")
        eva_train_dataset = None
    
    if args.ava_image_dir and args.ava_label_file:
        ava_train_dataset = AVADataset(
            label_file=args.ava_label_file,
            img_root=args.ava_image_dir,
            image_size=args.image_size,
            transform=get_default_transforms(args.image_size, train=True)
        )
    else:
        print("Warning: AVA dataset paths not provided, skipping AVA dataset")
        ava_train_dataset = None
    
    # Create dataloaders
    datasets = []
    if eva_train_dataset is not None:
        datasets.append(('EVA', eva_train_dataset))
    if ava_train_dataset is not None:
        datasets.append(('AVA', ava_train_dataset))
    
    if not datasets:
        print("Error: No datasets available. Please provide dataset paths.")
        return
    
    # Create model
    print(f"Creating model with backbone: {args.backbone}")
    model = create_lightweight_model(
        backbone=args.backbone,
        embedding_dim=args.embedding_dim,
        pretrained=True
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")
    
    # Create continual learning framework
    cl_framework = ContinualLearningFramework(
        model=model,
        strategy=args.cl_strategy,
        strategy_kwargs={'ewc_lambda': 1000} if args.cl_strategy == 'ewc' else {}
    )
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Training loop for each task
    for task_id, (dataset_name, dataset) in enumerate(datasets):
        print(f"\n{'='*50}")
        print(f"Training Task {task_id}: {dataset_name}")
        print(f"Dataset size: {len(dataset)}")
        print(f"{'='*50}")
        
        # Prepare for new task
        cl_framework.before_task(task_id=task_id)
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
            pin_memory=True if device.type == 'cuda' else False
        )
        
        # Train for specified epochs
        for epoch in range(args.epochs):
            print(f"\nTask {task_id}, Epoch {epoch + 1}/{args.epochs}")
            
            # Train epoch
            train_metrics = train_epoch(
                model=model,
                cl_framework=cl_framework,
                dataloader=dataloader,
                optimizer=optimizer,
                device=device,
                task_id=task_id,
                epoch=epoch,
                writer=writer
            )
            
            print(f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"SRCC: {train_metrics['srcc']:.4f}, "
                  f"PLCC: {train_metrics['plcc']:.4f}")
            
            # Log epoch metrics
            writer.add_scalar(f'Task_{task_id}/SRCC/Train', train_metrics['srcc'], epoch)
            writer.add_scalar(f'Task_{task_id}/PLCC/Train', train_metrics['plcc'], epoch)
            writer.add_scalar(f'Task_{task_id}/Loss/Epoch', train_metrics['loss'], epoch)
        
        # Evaluate on current task after training
        eval_metrics = evaluate_model(model, dataloader, device, dataset_name)
        print(f"\nTask {task_id} Final Evaluation:")
        print(f"SRCC: {eval_metrics['srcc']:.4f}, PLCC: {eval_metrics['plcc']:.4f}")
        
        # Save model checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'task_id': task_id,
            'epoch': args.epochs,
            'eval_metrics': eval_metrics
        }
        torch.save(checkpoint, os.path.join(args.save_dir, f'task_{task_id}_{dataset_name}.pth'))
        
        # Finish task
        cl_framework.after_task(task_id=task_id, dataloader=dataloader)
        
        print(f"Task {task_id} completed and checkpoint saved.")
    
    print(f"\n{'='*50}")
    print("All tasks completed!")
    print(f"{'='*50}")
    
    # Test on all tasks after continual learning (to check forgetting)
    print("\nEvaluating on all tasks after continual learning:")
    for task_id, (dataset_name, dataset) in enumerate(datasets):
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
            pin_memory=True if device.type == 'cuda' else False
        )
        
        eval_metrics = evaluate_model(model, dataloader, device, dataset_name)
        print(f"Task {task_id} ({dataset_name}): SRCC: {eval_metrics['srcc']:.4f}, PLCC: {eval_metrics['plcc']:.4f}")
        
        # Log final metrics
        writer.add_scalar(f'Final/SRCC/{dataset_name}', eval_metrics['srcc'], 0)
        writer.add_scalar(f'Final/PLCC/{dataset_name}', eval_metrics['plcc'], 0)
    
    writer.close()
    print(f"\nLogs saved to: {args.log_dir}")
    print(f"Checkpoints saved to: {args.save_dir}")


if __name__ == '__main__':
    main()