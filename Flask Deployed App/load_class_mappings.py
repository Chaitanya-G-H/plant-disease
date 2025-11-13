"""
Utility module to load class mappings dynamically.
This allows the app to work with different numbers of classes without hardcoding.
"""

import os
import json
import torch
from torchvision import datasets
import torchvision.transforms as transforms

def load_class_mappings_from_dataset(dataset_path=None):
    """
    Load class mappings directly from the dataset.
    
    Args:
        dataset_path: Path to dataset folder. If None, tries to find it automatically.
    
    Returns:
        dict: Dictionary with 'class_to_idx', 'idx_to_class', 'num_classes', 'classes'
    """
    if dataset_path is None:
        # Try to find dataset path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        dataset_path = os.path.join(project_root, 'Model', 'Dataset')
        dataset_path = os.path.normpath(dataset_path)
    
    if not os.path.exists(dataset_path):
        # Fall back to hardcoded mappings if dataset not found
        return load_hardcoded_mappings()
    
    try:
        transform = transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        
        dataset = datasets.ImageFolder(dataset_path, transform=transform)
        class_to_idx = dataset.class_to_idx
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        
        return {
            'class_to_idx': class_to_idx,
            'idx_to_class': idx_to_class,
            'num_classes': len(dataset.classes),
            'classes': dataset.classes
        }
    except Exception as e:
        print(f"Warning: Could not load dataset from {dataset_path}: {e}")
        print("Falling back to hardcoded mappings.")
        return load_hardcoded_mappings()

def load_class_mappings_from_json(json_path=None):
    """
    Load class mappings from a JSON file.
    
    Args:
        json_path: Path to JSON file. If None, tries to find it automatically.
    
    Returns:
        dict: Dictionary with class mappings
    """
    if json_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(script_dir, 'class_mappings.json')
        json_path = os.path.normpath(json_path)
    
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert string keys back to integers
            idx_to_class = {int(k): v for k, v in data.get('idx_to_class', {}).items()}
            class_to_idx = data.get('class_to_idx', {})
            
            return {
                'class_to_idx': class_to_idx,
                'idx_to_class': idx_to_class,
                'num_classes': data.get('num_classes', len(idx_to_class)),
                'classes': list(class_to_idx.keys())
            }
        except Exception as e:
            print(f"Warning: Could not load JSON from {json_path}: {e}")
    
    return load_hardcoded_mappings()

def load_hardcoded_mappings():
    """
    Load hardcoded class mappings from CNN.py (fallback).
    
    Returns:
        dict: Dictionary with class mappings
    """
    try:
        import CNN
        idx_to_classes = CNN.idx_to_classes
        class_to_idx = {v: k for k, v in idx_to_classes.items()}
        
        return {
            'class_to_idx': class_to_idx,
            'idx_to_class': idx_to_classes,
            'num_classes': len(idx_to_classes),
            'classes': list(class_to_idx.keys())
        }
    except Exception as e:
        print(f"Error loading hardcoded mappings: {e}")
        raise

def get_class_mappings(prefer_dataset=True):
    """
    Get class mappings, trying different sources in order.
    
    Args:
        prefer_dataset: If True, prefer loading from dataset over JSON
    
    Returns:
        dict: Dictionary with class mappings
    """
    if prefer_dataset:
        # Try dataset first
        try:
            mappings = load_class_mappings_from_dataset()
            if mappings and mappings['num_classes'] > 0:
                return mappings
        except Exception as e:
            print(f"Could not load from dataset: {e}")
        
        # Try JSON second
        try:
            mappings = load_class_mappings_from_json()
            if mappings and mappings['num_classes'] > 0:
                return mappings
        except Exception as e:
            print(f"Could not load from JSON: {e}")
    else:
        # Try JSON first
        try:
            mappings = load_class_mappings_from_json()
            if mappings and mappings['num_classes'] > 0:
                return mappings
        except Exception as e:
            print(f"Could not load from JSON: {e}")
        
        # Try dataset second
        try:
            mappings = load_class_mappings_from_dataset()
            if mappings and mappings['num_classes'] > 0:
                return mappings
        except Exception as e:
            print(f"Could not load from dataset: {e}")
    
    # Fall back to hardcoded
    return load_hardcoded_mappings()

if __name__ == '__main__':
    # Test the module
    print("Testing class mapping loader...")
    mappings = get_class_mappings()
    print(f"Loaded {mappings['num_classes']} classes")
    print(f"Sample classes: {list(mappings['classes'])[:5]}")

