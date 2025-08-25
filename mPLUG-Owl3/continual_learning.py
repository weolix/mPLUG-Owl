"""
Configurable Continual Learning Framework for Image Quality Assessment
Supports different continual learning strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Union
from abc import ABC, abstractmethod


class ContinualLearningStrategy(ABC):
    """Abstract base class for continual learning strategies"""
    
    @abstractmethod
    def before_task(self, model, task_id, **kwargs):
        """Called before training on a new task"""
        pass
    
    @abstractmethod
    def after_task(self, model, task_id, **kwargs):
        """Called after training on a task"""
        pass
    
    @abstractmethod
    def compute_loss(self, model, batch, task_id, **kwargs):
        """Compute the loss for the current batch"""
        pass


class NaiveContinualLearning(ContinualLearningStrategy):
    """Naive continual learning - just train sequentially without any special handling"""
    
    def __init__(self):
        super().__init__()
    
    def before_task(self, model, task_id, **kwargs):
        pass
    
    def after_task(self, model, task_id, **kwargs):
        pass
    
    def compute_loss(self, model, batch, task_id, criterion, **kwargs):
        images, targets = batch
        outputs = model(images)
        
        # Use similarity scores as predictions
        predictions = outputs['similarity_scores']
        loss = criterion(predictions, targets)
        
        return {
            'loss': loss,
            'predictions': predictions,
            'targets': targets,
            'outputs': outputs
        }


class EWCContinualLearning(ContinualLearningStrategy):
    """Elastic Weight Consolidation for continual learning"""
    
    def __init__(self, ewc_lambda=1000, fisher_estimation_samples=200):
        super().__init__()
        self.ewc_lambda = ewc_lambda
        self.fisher_estimation_samples = fisher_estimation_samples
        self.fisher_dict = {}
        self.optpar_dict = {}
    
    def before_task(self, model, task_id, **kwargs):
        # Store previous task parameters if not first task
        if task_id > 0:
            self._consolidate_weights(model, task_id - 1)
    
    def after_task(self, model, task_id, dataloader=None, **kwargs):
        # Estimate Fisher Information Matrix
        if dataloader is not None:
            self._estimate_fisher(model, dataloader, task_id)
    
    def _estimate_fisher(self, model, dataloader, task_id):
        """Estimate Fisher Information Matrix"""
        model.eval()
        fisher_dict = {}
        optpar_dict = {}
        
        # Initialize Fisher information
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher_dict[name] = torch.zeros_like(param)
                optpar_dict[name] = param.data.clone()
        
        # Estimate Fisher information
        num_samples = 0
        for batch_idx, batch in enumerate(dataloader):
            if num_samples >= self.fisher_estimation_samples:
                break
                
            images, targets = batch
            if torch.cuda.is_available():
                images, targets = images.cuda(), targets.cuda()
            
            model.zero_grad()
            outputs = model(images)
            loss = F.mse_loss(outputs['similarity_scores'], targets)
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_dict[name] += param.grad.data ** 2
            
            num_samples += images.size(0)
        
        # Normalize Fisher information
        for name in fisher_dict:
            fisher_dict[name] /= num_samples
        
        self.fisher_dict[task_id] = fisher_dict
        self.optpar_dict[task_id] = optpar_dict
    
    def _consolidate_weights(self, model, task_id):
        """Store current parameters as optimal for the task"""
        optpar_dict = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                optpar_dict[name] = param.data.clone()
        self.optpar_dict[task_id] = optpar_dict
    
    def compute_loss(self, model, batch, task_id, criterion, **kwargs):
        images, targets = batch
        outputs = model(images)
        
        # Use similarity scores as predictions
        predictions = outputs['similarity_scores']
        current_loss = criterion(predictions, targets)
        
        # Add EWC penalty for previous tasks
        ewc_loss = 0
        for prev_task_id in range(task_id):
            if prev_task_id in self.fisher_dict:
                for name, param in model.named_parameters():
                    if param.requires_grad and name in self.fisher_dict[prev_task_id]:
                        fisher = self.fisher_dict[prev_task_id][name]
                        optpar = self.optpar_dict[prev_task_id][name]
                        ewc_loss += (fisher * (param - optpar) ** 2).sum()
        
        total_loss = current_loss + self.ewc_lambda * ewc_loss
        
        return {
            'loss': total_loss,
            'current_loss': current_loss,
            'ewc_loss': ewc_loss,
            'predictions': predictions,
            'targets': targets,
            'outputs': outputs
        }


class L2ContinualLearning(ContinualLearningStrategy):
    """L2 regularization towards previous task parameters"""
    
    def __init__(self, l2_lambda=0.01):
        super().__init__()
        self.l2_lambda = l2_lambda
        self.previous_params = {}
    
    def before_task(self, model, task_id, **kwargs):
        if task_id > 0:
            # Store previous parameters
            prev_params = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    prev_params[name] = param.data.clone()
            self.previous_params[task_id - 1] = prev_params
    
    def after_task(self, model, task_id, **kwargs):
        pass
    
    def compute_loss(self, model, batch, task_id, criterion, **kwargs):
        images, targets = batch
        outputs = model(images)
        
        # Use similarity scores as predictions
        predictions = outputs['similarity_scores']
        current_loss = criterion(predictions, targets)
        
        # Add L2 penalty towards previous parameters
        l2_loss = 0
        if task_id > 0 and (task_id - 1) in self.previous_params:
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.previous_params[task_id - 1]:
                    prev_param = self.previous_params[task_id - 1][name]
                    l2_loss += ((param - prev_param) ** 2).sum()
        
        total_loss = current_loss + self.l2_lambda * l2_loss
        
        return {
            'loss': total_loss,
            'current_loss': current_loss,
            'l2_loss': l2_loss,
            'predictions': predictions,
            'targets': targets,
            'outputs': outputs
        }


class PackNetContinualLearning(ContinualLearningStrategy):
    """PackNet-style pruning for continual learning"""
    
    def __init__(self, prune_ratio=0.2):
        super().__init__()
        self.prune_ratio = prune_ratio
        self.masks = {}
        self.task_masks = {}
    
    def before_task(self, model, task_id, **kwargs):
        if task_id > 0:
            # Apply masks from previous tasks
            self._apply_masks(model)
    
    def after_task(self, model, task_id, **kwargs):
        # Create mask for current task
        self._create_mask(model, task_id)
    
    def _create_mask(self, model, task_id):
        """Create pruning mask for the current task"""
        # Calculate parameter importance (simple magnitude-based for now)
        param_importance = {}
        for name, param in model.named_parameters():
            if param.requires_grad and len(param.shape) > 1:  # Only for weight matrices
                param_importance[name] = param.abs()
        
        # Create mask for this task
        task_mask = {}
        for name, importance in param_importance.items():
            # Get available parameters (not used by previous tasks)
            available_mask = torch.ones_like(importance)
            for prev_task in range(task_id):
                if prev_task in self.task_masks and name in self.task_masks[prev_task]:
                    available_mask &= ~self.task_masks[prev_task][name]
            
            # Select top parameters for this task
            available_params = importance * available_mask.float()
            flat_params = available_params.flatten()
            k = int(len(flat_params) * self.prune_ratio)
            if k > 0:
                _, top_indices = torch.topk(flat_params, k)
                mask = torch.zeros_like(flat_params, dtype=torch.bool)
                mask[top_indices] = True
                task_mask[name] = mask.reshape(importance.shape)
            else:
                task_mask[name] = torch.zeros_like(importance, dtype=torch.bool)
        
        self.task_masks[task_id] = task_mask
    
    def _apply_masks(self, model):
        """Apply all task masks to freeze parameters"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                total_mask = torch.zeros_like(param, dtype=torch.bool)
                for task_id, task_mask in self.task_masks.items():
                    if name in task_mask:
                        total_mask |= task_mask[name]
                
                # Freeze masked parameters
                if name not in self.masks:
                    self.masks[name] = total_mask
                else:
                    self.masks[name] |= total_mask
    
    def compute_loss(self, model, batch, task_id, criterion, **kwargs):
        images, targets = batch
        outputs = model(images)
        
        # Use similarity scores as predictions
        predictions = outputs['similarity_scores']
        loss = criterion(predictions, targets)
        
        return {
            'loss': loss,
            'predictions': predictions,
            'targets': targets,
            'outputs': outputs
        }


class ContinualLearningFramework:
    """Main framework for continual learning"""
    
    def __init__(self, 
                 model,
                 strategy: Union[str, ContinualLearningStrategy] = 'naive',
                 strategy_kwargs: Optional[Dict] = None):
        self.model = model
        self.current_task = 0
        
        # Initialize strategy
        if isinstance(strategy, str):
            strategy_kwargs = strategy_kwargs or {}
            if strategy == 'naive':
                self.strategy = NaiveContinualLearning()
            elif strategy == 'ewc':
                self.strategy = EWCContinualLearning(**strategy_kwargs)
            elif strategy == 'l2':
                self.strategy = L2ContinualLearning(**strategy_kwargs)
            elif strategy == 'packnet':
                self.strategy = PackNetContinualLearning(**strategy_kwargs)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
        else:
            self.strategy = strategy
    
    def before_task(self, task_id=None, **kwargs):
        """Prepare for training on a new task"""
        if task_id is None:
            task_id = self.current_task
        
        self.strategy.before_task(self.model, task_id, **kwargs)
    
    def after_task(self, task_id=None, **kwargs):
        """Clean up after training on a task"""
        if task_id is None:
            task_id = self.current_task
        
        self.strategy.after_task(self.model, task_id, **kwargs)
        self.current_task += 1
    
    def compute_loss(self, batch, criterion, task_id=None, **kwargs):
        """Compute loss for the current batch"""
        if task_id is None:
            task_id = self.current_task
        
        return self.strategy.compute_loss(self.model, batch, task_id, criterion, **kwargs)
    
    def set_strategy(self, strategy: Union[str, ContinualLearningStrategy], **kwargs):
        """Change the continual learning strategy"""
        if isinstance(strategy, str):
            if strategy == 'naive':
                self.strategy = NaiveContinualLearning()
            elif strategy == 'ewc':
                self.strategy = EWCContinualLearning(**kwargs)
            elif strategy == 'l2':
                self.strategy = L2ContinualLearning(**kwargs)
            elif strategy == 'packnet':
                self.strategy = PackNetContinualLearning(**kwargs)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
        else:
            self.strategy = strategy