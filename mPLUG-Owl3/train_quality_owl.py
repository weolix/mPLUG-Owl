from dataset import ViewDecompositionDataset, dict_simply_collate
from .quality_plugowl3 import QualityOwl3Model

"""
1. 在数据集上跑一遍原始模型+高频词推理，得到高频词indices+权重
2. 在模型上加入quality brance，训练quality brance
"""

