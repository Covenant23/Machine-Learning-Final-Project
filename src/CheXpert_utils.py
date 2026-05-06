"""
CheXpert Utilities Module
Contains all data loading, model, and training functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time


from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, roc_curve, precision_recall_curve, auc


# =============================================================================
# CONFIGURATION
# =============================================================================

CHEXPERT_CSV_PATH = '/home/cadenug/ML Final Project/datasets/ashery/chexpert/versions/1/'
CHEXPERT_IMG_PATH = '/home/cadenug/ML Final Project/datasets/ashery/chexpert/versions/1/'

SELECTED_DISEASES = [
    'Lung Opacity',
    'Pleural Effusion', 
    'Atelectasis',
    'Pneumothorax',
    'Pneumonia'
]

# =============================================================================
# DATASET CLASS
# =============================================================================

class CheXpertDataset(Dataset):
    """CheXpert Dataset with U-Zeros strategy"""
    
    def __init__(self, dataframe, root_dir, transform=None):
        """
        Args:
            dataframe: Pandas DataFrame with image paths and labels
            root_dir: Root directory for images
            transform: Torchvision transforms
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.diseases = SELECTED_DISEASES
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        # Get image path
        img_path = self.dataframe.iloc[idx]['Path']
        
        # Strip 'CheXpert-v1.0-small/' prefix and construct full path
        img_path = img_path.replace('CheXpert-v1.0-small/', '')
        full_path = os.path.join(self.root_dir, img_path)
        
        # Load image
        try:
            image = Image.open(full_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {full_path}: {e}")
            # Return a blank image if load fails
            image = Image.new('RGB', (224, 224))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get labels (multi-hot encoded)
        labels = []
        for disease in self.diseases:
            label_value = self.dataframe.iloc[idx][disease]
            labels.append(label_value)
        
        labels = torch.FloatTensor(labels)
        
        return image, labels


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def calculate_class_weights(train_dataset):
    """Calculate class weights using square root (gentler)"""
    
    print("\n" + "="*80)
    print("CALCULATING SQRT CLASS WEIGHTS FOR WEIGHTED BCE")
    print("="*80)
    
    # Load the dataframe from dataset
    df = train_dataset.dataframe
    total_samples = len(df)
    
    weights = []
    print(f"\n{'Disease':<20} {'Positive':>10} {'Original':>10} {'Sqrt':>10}")
    print("-" * 55)
    
    for disease in SELECTED_DISEASES:
        pos_count = (df[disease] == 1.0).sum()
        
        # Original weight (for comparison)
        orig_weight = total_samples / (2 * pos_count) if pos_count > 0 else 1.0
        
        # Square root weight (GENTLER)
        sqrt_weight = np.sqrt(total_samples / (2 * pos_count)) if pos_count > 0 else 1.0
        
        weights.append(sqrt_weight)
        print(f"{disease:<20} {pos_count:>10,} {orig_weight:>10.3f} {sqrt_weight:>10.3f}")
    
    weights = torch.FloatTensor(weights)
    print(f"\n Sqrt class weights calculated: {weights}")
    
    return weights

# =============================================================================
# MODEL CLASSES
# =============================================================================

class SimpleCNN(nn.Module):
    """
    Simple CNN baseline trained from scratch
    Lightweight architecture for comparison with transfer learning
    """
    def __init__(self, num_classes=5):
        super(SimpleCNN, self).__init__()
        
        # Convolutional blocks
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 224 -> 112
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112 -> 56
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56 -> 28
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28 -> 14
            
            # Block 5
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 14 -> 7
        )
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x  # No sigmoid - we'll use BCEWithLogitsLoss


class DenseNet121(nn.Module):
    """
    DenseNet-121 with pretrained ImageNet weights
    Transfer learning approach
    """
    def __init__(self, num_classes=5, pretrained=True):
        super(DenseNet121, self).__init__()
        
        # Load pretrained DenseNet-121
        self.model = models.densenet121(pretrained=pretrained)
        
        # Get number of features in the last layer
        num_features = self.model.classifier.in_features
        
        # Replace classifier for multi-label classification
        self.model.classifier = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.model(x)  # No sigmoid - we'll use BCEWithLogitsLoss


def get_model(model_name='densenet121', num_classes=5, pretrained=True):
    """
    Factory function to create models
    
    Args:
        model_name: 'simple_cnn' or 'densenet121'
        num_classes: Number of output classes
        pretrained: Use pretrained weights (only for densenet121)
    
    Returns:
        model: PyTorch model
    """
    if model_name == 'simple_cnn':
        model = SimpleCNN(num_classes=num_classes)
        print(f" Created SimpleCNN (trained from scratch)")
    elif model_name == 'densenet121':
        model = DenseNet121(num_classes=num_classes, pretrained=pretrained)
        if pretrained:
            print(f"Created DenseNet-121 (pretrained on ImageNet)")
        else:
            print(f"Created DenseNet-121 (random initialization)")
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def count_parameters(model):
    """Count trainable parameters in model"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def get_loss_function(loss_type='bce', class_weights=None, device='cuda'):
    """
    Get loss function
    Args:
        loss_type: 'bce', 'weighted_bce', or 'focal'
        class_weights: Tensor of class weights (for weighted_bce)
        device: Device to put weights on
    Returns:
        loss_fn: Loss function
    """
    if loss_type == 'bce':
        loss_fn = nn.BCEWithLogitsLoss()
        print(" Using Binary Cross-Entropy Loss")
        
    elif loss_type == 'weighted_bce':
        if class_weights is None:
            raise ValueError("class_weights required for weighted_bce")
        pos_weight = class_weights.to(device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Using Weighted BCE Loss")
        print(f"Weights: {pos_weight}")
        
    elif loss_type == 'focal':
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        print("Using Focal Loss (alpha=0.25, gamma=2.0)")
        
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    return loss_fn.to(device)  # ← ADD THIS to ensure loss is on correct device

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification
    
    Focal Loss addresses class imbalance by down-weighting easy examples
    and focusing on hard examples.
    
    Formula: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    where:
    - p_t is the model's estimated probability for the class
    - alpha: balancing factor for positive/negative examples (default: 0.25)
    - gamma: focusing parameter (default: 2.0)
      - gamma=0 → Focal Loss = BCE
      - gamma>0 → down-weights easy examples
    
    Args:
        alpha (float): Weighting factor for positive class (default: 0.25)
        gamma (float): Focusing parameter (default: 2.0)
        reduction (str): 'mean' or 'sum' (default: 'mean')
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits from model (before sigmoid), shape (batch_size, num_classes)
            targets: Ground truth labels, shape (batch_size, num_classes)
        
        Returns:
            Focal loss value
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)
        
        # Calculate binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate p_t
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Calculate focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Calculate alpha weight
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Combine: Focal Loss = alpha * focal_weight * BCE
        focal_loss = alpha_t * focal_weight * bce_loss
        
        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Train for one epoch
    
    Args:
        model: PyTorch model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device (cuda/cpu)
        epoch: Current epoch number
    
    Returns:
        avg_loss: Average loss for the epoch
    """
    model.train()
    running_loss = 0.0
    
    # Progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, (images, labels) in enumerate(pbar):
        # Move to device
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update running loss
        running_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = running_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, criterion, device, epoch):
    """
    Validate model
    
    Args:
        model: PyTorch model
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device (cuda/cpu)
        epoch: Current epoch number
    
    Returns:
        avg_loss: Average validation loss
        all_preds: All predictions (numpy array)
        all_labels: All labels (numpy array)
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    # Progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]  ")
    
    with torch.no_grad():
        for images, labels in pbar:
            # Move to device
            images = images.to(device)
            labels_gpu = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels_gpu)
            
            # Update running loss
            running_loss += loss.item()
            
            # Get predictions (apply sigmoid)
            preds = torch.sigmoid(outputs)
            
            # Store predictions and labels
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = running_loss / len(dataloader)
    
    # Concatenate all predictions and labels
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return avg_loss, all_preds, all_labels


def train_model(model, train_loader, val_loader, criterion, optimizer, 
                device, num_epochs=10, save_path='best_model.pth', 
                resume_from=None):
    """
    Full training loop with early stopping and checkpointing
    
    Args:
        model: PyTorch model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device (cuda/cpu)
        num_epochs: Number of epochs
        save_path: Path to save best model
        resume_from: Path to checkpoint to resume from (optional)
    
    Returns:
        history: Dictionary with training history
    """
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    start_epoch = 1
    
    history = {
        'train_loss': [],
        'val_loss': [],
    }
    
    # Resume from checkpoint if provided
    if resume_from and os.path.exists(resume_from):
        print(f"\n Loading checkpoint from {resume_from}")
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        history = checkpoint['history']
        patience_counter = checkpoint['patience_counter']
        print(f" Resumed from epoch {checkpoint['epoch']}")
        print(f" Best val loss so far: {best_val_loss:.4f}")
    
    print(f"\n{'='*80}")
    print(f"TRAINING START")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Epochs: {start_epoch} to {num_epochs}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"{'='*80}\n")
    
    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, 
                                     optimizer, device, epoch)
        
        # Validate
        val_loss, val_preds, val_labels = validate(model, val_loader, 
                                                    criterion, device, epoch)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{num_epochs} - {epoch_time:.1f}s")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        
        # Save checkpoint every epoch
        checkpoint_dir = os.path.dirname(save_path)
        checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pth')
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'history': history,
            'patience_counter': patience_counter
        }
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model (val_loss: {val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")
        
        print(f"  Checkpoint saved: {checkpoint_path}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n  Early stopping triggered after {epoch} epochs")
            break
        
        print(f"{'-'*80}\n")
    
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {save_path}")
    
    return history

# ============================================================================
# DATA AUGMENTATION TRANSFORMS
# ============================================================================

def get_augmentation_transforms(aug_type='minimal', input_size=224):
    """
    Get data augmentation transforms for training
    
    Args:
        aug_type (str): Type of augmentation
            - 'minimal': Basic augmentation (flip + small rotation)
            - 'geometric': Heavy geometric transformations
            - 'photometric': Heavy intensity/contrast transformations  
            - 'heavy': Combined geometric + photometric + advanced
        input_size (int): Input image size (default: 224)
    
    Returns:
        transform: Torchvision transform composition
    """
    
    # ImageNet normalization (always applied)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if aug_type == 'minimal':
        # Current baseline augmentation
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            normalize
        ])
        print(" Using MINIMAL augmentation:")
        print("   - Random horizontal flip (50%)")
        print("   - Random rotation (±10°)")
    
    elif aug_type == 'geometric':
        # Heavy geometric augmentation
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=25),  # Increased from ±10° to ±25°
            transforms.RandomAffine(
                degrees=0,  # Rotation handled above
                translate=(0.1, 0.1),  # ±10% translation
                scale=(0.9, 1.1),  # 90-110% zoom
                shear=5  # Small shear
            ),
            transforms.ToTensor(),
            normalize
        ])
        print(" Using GEOMETRIC augmentation:")
        print("   - Random horizontal flip (50%)")
        print("   - Random rotation (±25°)")
        print("   - Random translation (±10%)")
        print("   - Random zoom (90-110%)")
        print("   - Random shear (±5°)")
    
    elif aug_type == 'photometric':
        # Heavy photometric (intensity/contrast) augmentation
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),  # Keep minimal rotation
            transforms.ColorJitter(
                brightness=0.3,  # ±30% brightness
                contrast=0.3,    # ±30% contrast
                saturation=0,    # No saturation (grayscale X-rays)
                hue=0           # No hue shift
            ),
            transforms.ToTensor(),
            normalize
        ])
        print(" Using PHOTOMETRIC augmentation:")
        print("   - Random horizontal flip (50%)")
        print("   - Random rotation (±10°)")
        print("   - Random brightness (±30%)")
        print("   - Random contrast (±30%)")
    
    elif aug_type == 'heavy':
        # Combined: Geometric + Photometric + Advanced
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=25),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=5
            ),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0,
                hue=0
            ),
            transforms.RandomErasing(
                p=0.3,  # 30% chance to apply
                scale=(0.02, 0.1),  # Erase 2-10% of image
                ratio=(0.3, 3.3)
            ),
            transforms.ToTensor(),
            normalize
        ])
        print(" Using HEAVY (Combined) augmentation:")
        print("   - Random horizontal flip (50%)")
        print("   - Random rotation (±25°)")
        print("   - Random translation (±10%)")
        print("   - Random zoom (90-110%)")
        print("   - Random shear (±5°)")
        print("   - Random brightness (±30%)")
        print("   - Random contrast (±30%)")
        print("   - Random erasing (30% chance, 2-10% area)")
    
    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")
    
    return transform


def get_dataloaders_with_augmentation(train_df, val_df, test_df, 
                                     cache_dir, aug_type='minimal',
                                     batch_size=32, num_workers=4):
    """
    Create data loaders with specified augmentation
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe  
        test_df: Test dataframe
        cache_dir: Path to cached dataset
        aug_type: Augmentation type ('minimal', 'geometric', 'photometric', 'heavy')
        batch_size: Batch size
        num_workers: Number of worker processes
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Get augmentation transforms
    train_transform = get_augmentation_transforms(aug_type=aug_type)
    
    # Validation/test transforms (no augmentation, just resize + normalize)
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = CheXpertDataset(train_df, cache_dir, transform=train_transform)
    val_dataset = CheXpertDataset(val_df, cache_dir, transform=val_test_transform)
    test_dataset = CheXpertDataset(test_df, cache_dir, transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n Created dataloaders with {aug_type} augmentation")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val:   {len(val_dataset)} samples")
    print(f"   Test:  {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader

# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_model(model, dataloader, device, disease_names):
    """
    Evaluate model and compute metrics
    
    Args:
        model: Trained PyTorch model
        dataloader: DataLoader (test or validation)
        device: Device (cuda/cpu)
        disease_names: List of disease names
    
    Returns:
        results: Dictionary with all metrics
        all_preds: Predictions (for further analysis)
        all_labels: Ground truth labels
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    print(f"\n{'='*80}")
    print(f"EVALUATING MODEL")
    print(f"{'='*80}")
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Compute metrics
    results = {}
    
    print(f"\n Per-Disease Metrics (Threshold = 0.5):")
    print(f"{'Disease':<20} {'AUROC':>8} {'F1':>8} {'Precision':>10} {'Recall':>8}")
    print("-" * 65)
    
    aurocs = []
    f1s = []
    precisions = []
    recalls = []
    
    for i, disease in enumerate(disease_names):
        # AUROC
        auroc = roc_auc_score(all_labels[:, i], all_preds[:, i])
        aurocs.append(auroc)
        
        # Binary predictions (threshold = 0.5)
        preds_binary = (all_preds[:, i] > 0.5).astype(int)
        
        # F1, Precision, Recall
        f1 = f1_score(all_labels[:, i], preds_binary, zero_division=0)
        prec = precision_score(all_labels[:, i], preds_binary, zero_division=0)
        rec = recall_score(all_labels[:, i], preds_binary, zero_division=0)
        
        f1s.append(f1)
        precisions.append(prec)
        recalls.append(rec)
        
        # Store in results
        results[f'{disease}_auroc'] = auroc
        results[f'{disease}_f1'] = f1
        results[f'{disease}_precision'] = prec
        results[f'{disease}_recall'] = rec
        
        print(f"{disease:<20} {auroc:>8.4f} {f1:>8.4f} {prec:>10.4f} {rec:>8.4f}")
    
    # Macro averages
    results['macro_auroc'] = np.mean(aurocs)
    results['macro_f1'] = np.mean(f1s)
    results['macro_precision'] = np.mean(precisions)
    results['macro_recall'] = np.mean(recalls)
    
    print("-" * 65)
    print(f"{'MACRO AVERAGE':<20} {results['macro_auroc']:>8.4f} {results['macro_f1']:>8.4f} {results['macro_precision']:>10.4f} {results['macro_recall']:>8.4f}")
    
    return results, all_preds, all_labels


def optimize_thresholds(preds, labels, disease_names):
    """
    Find optimal threshold per disease (maximize F1)
    
    Args:
        preds: Predictions (N x num_diseases)
        labels: Ground truth labels (N x num_diseases)
        disease_names: List of disease names
    
    Returns:
        optimal_thresholds: Dictionary with optimal thresholds and metrics
    """
    print(f"\n{'='*80}")
    print(f"OPTIMIZING THRESHOLDS (Maximize F1)")
    print(f"{'='*80}")
    
    optimal_thresholds = {}
    
    print(f"\n{'Disease':<20} {'Optimal Threshold':>18} {'F1 @ 0.5':>10} {'F1 @ Optimal':>15} {'Improvement':>12}")
    print("-" * 80)
    
    for i, disease in enumerate(disease_names):
        best_f1 = 0
        best_thresh = 0.5
        f1_at_05 = f1_score(labels[:, i], (preds[:, i] > 0.5).astype(int), zero_division=0)
        
        # Try thresholds from 0.1 to 0.9
        for thresh in np.arange(0.05, 0.95, 0.05):
            preds_binary = (preds[:, i] > thresh).astype(int)
            f1 = f1_score(labels[:, i], preds_binary, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        improvement = ((best_f1 - f1_at_05) / f1_at_05 * 100) if f1_at_05 > 0 else 0
        
        optimal_thresholds[disease] = {
            'threshold': best_thresh,
            'f1_optimal': best_f1,
            'f1_at_0.5': f1_at_05,
            'improvement_pct': improvement
        }
        
        print(f"{disease:<20} {best_thresh:>18.2f} {f1_at_05:>10.4f} {best_f1:>15.4f} {improvement:>11.1f}%")
    
    return optimal_thresholds


def plot_roc_curves(all_results, disease_names, save_path=None):
    """
    Plot ROC curves for all models and diseases
    
    Args:
        all_results: Dict of {model_name: (preds, labels)}
        disease_names: List of disease names
        save_path: Path to save figure
    """
    n_diseases = len(disease_names)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    for i, disease in enumerate(disease_names):
        ax = axes[i]
        
        for model_idx, (model_name, (preds, labels)) in enumerate(all_results.items()):
            fpr, tpr, _ = roc_curve(labels[:, i], preds[:, i])
            auroc = roc_auc_score(labels[:, i], preds[:, i])
            
            ax.plot(fpr, tpr, color=colors[model_idx % len(colors)], 
                   label=f'{model_name} (AUC = {auroc:.3f})', linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.3)
        ax.set_xlabel('False Positive Rate', fontsize=10)
        ax.set_ylabel('True Positive Rate', fontsize=10)
        ax.set_title(f'{disease}', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Remove extra subplot
    if n_diseases < 6:
        fig.delaxes(axes[5])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n ROC curves saved to: {save_path}")
    
    plt.show()


def plot_pr_curves(all_results, disease_names, save_path=None):
    """
    Plot Precision-Recall curves for all models and diseases
    
    Args:
        all_results: Dict of {model_name: (preds, labels)}
        disease_names: List of disease names
        save_path: Path to save figure
    """
    n_diseases = len(disease_names)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    for i, disease in enumerate(disease_names):
        ax = axes[i]
        
        for model_idx, (model_name, (preds, labels)) in enumerate(all_results.items()):
            precision, recall, _ = precision_recall_curve(labels[:, i], preds[:, i])
            pr_auc = auc(recall, precision)
            
            ax.plot(recall, precision, color=colors[model_idx % len(colors)], 
                   label=f'{model_name} (AUC = {pr_auc:.3f})', linewidth=2)
        
        ax.set_xlabel('Recall', fontsize=10)
        ax.set_ylabel('Precision', fontsize=10)
        ax.set_title(f'{disease}', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Remove extra subplot
    if n_diseases < 6:
        fig.delaxes(axes[5])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n PR curves saved to: {save_path}")
    
    plt.show()

def plot_training_curve(history, title='Training History', save_path=None):
    """
    Plot training and validation loss curves
    
    Args:
        history (dict): Training history with 'train_loss' and 'val_loss' keys
        title (str): Plot title
        save_path (str, optional): Path to save the figure
    """
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot losses
    ax.plot(epochs, history['train_loss'], 'b-', linewidth=2, 
            label='Training Loss', marker='o', markersize=4)
    ax.plot(epochs, history['val_loss'], 'r-', linewidth=2, 
            label='Validation Loss', marker='s', markersize=4)
    
    # Mark best validation epoch
    best_epoch = np.argmin(history['val_loss']) + 1
    best_val_loss = min(history['val_loss'])
    ax.axvline(x=best_epoch, color='green', linestyle='--', linewidth=2, 
              alpha=0.7, label=f'Best Val Loss (Epoch {best_epoch})')
    ax.plot(best_epoch, best_val_loss, 'g*', markersize=20, 
           markeredgecolor='black', markeredgewidth=1.5)
    
    # Labels and formatting
    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add text box with best epoch info
    textstr = f'Best Epoch: {best_epoch}\nBest Val Loss: {best_val_loss:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Training curve saved: {save_path}")
    
    plt.show()
    
    # Print summary
    print(f"\nTraining Summary:")
    print(f"  Total epochs: {len(history['train_loss'])}")
    print(f"  Best validation loss: {best_val_loss:.4f} (epoch {best_epoch})")
    print(f"  Final training loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final validation loss: {history['val_loss'][-1]:.4f}")

def create_comparison_table(all_metrics, save_path=None):
    """
    Create comparison table for all models
    
    Args:
        all_metrics: Dict of {model_name: results_dict}
        save_path: Path to save CSV
    
    Returns:
        comparison_df: Pandas DataFrame with comparison
    """
    rows = []
    
    for model_name, metrics in all_metrics.items():
        row = {'Model': model_name}
        row['Macro AUROC'] = metrics['macro_auroc']
        row['Macro F1'] = metrics['macro_f1']
        row['Macro Precision'] = metrics['macro_precision']
        row['Macro Recall'] = metrics['macro_recall']
        rows.append(row)
    
    comparison_df = pd.DataFrame(rows)
    
    print(f"\n{'='*80}")
    print(f"MODEL COMPARISON TABLE")
    print(f"{'='*80}\n")
    print(comparison_df.to_string(index=False))
    
    if save_path:
        comparison_df.to_csv(save_path, index=False)
        print(f"\n Comparison table saved to: {save_path}")
    
    return comparison_df

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def get_predictions(model, dataloader, device):
    """
    Get model predictions and labels from a dataloader
    
    Args:
        model: PyTorch model
        dataloader: DataLoader to get predictions from
        device: Device (cuda/cpu)
    
    Returns:
        preds: Numpy array of predictions (N, num_classes)
        labels: Numpy array of true labels (N, num_classes)
    """
    
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print(f"Getting predictions from {len(dataloader)} batches...")
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Predicting"):
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)
            
            # Store predictions and labels
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    
    # Concatenate all batches
    preds = np.vstack(all_preds)
    labels = np.vstack(all_labels)
    
    print(f"✅ Got predictions: {preds.shape}")
    
    return preds, labels

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_model_from_preds(preds, labels, disease_names, model_name="Model", threshold=0.5):
    """
    Evaluate model performance from predictions and labels
    
    Args:
        preds: Predicted probabilities (N, num_classes)
        labels: True labels (N, num_classes)
        disease_names: List of disease names
        model_name: Name of model (for printing)
        threshold: Classification threshold (default: 0.5)
    
    Returns:
        results: Dictionary with all metrics
        per_disease_metrics: Dictionary with per-disease metrics
        macro_metrics: Dictionary with macro-averaged metrics
    """
    
    print(f"\n{'='*80}")
    print(f"EVALUATING: {model_name}")
    print(f"{'='*80}")
    
    num_classes = len(disease_names)
    
    # Convert predictions to binary using threshold
    preds_binary = (preds > threshold).astype(int)
    
    # Store results
    results = {}
    per_disease_metrics = {}
    
    # Calculate per-disease metrics
    print(f"\n{'Disease':<25} {'AUROC':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
    print("-" * 70)
    
    aurocs = []
    f1s = []
    precisions = []
    recalls = []
    
    for disease_idx, disease in enumerate(disease_names):
        # AUROC
        try:
            auroc = roc_auc_score(labels[:, disease_idx], preds[:, disease_idx])
        except:
            auroc = 0.0  # If only one class present
        
        # F1, Precision, Recall
        f1 = f1_score(labels[:, disease_idx], preds_binary[:, disease_idx], zero_division=0)
        precision = precision_score(labels[:, disease_idx], preds_binary[:, disease_idx], zero_division=0)
        recall = recall_score(labels[:, disease_idx], preds_binary[:, disease_idx], zero_division=0)
        
        # Store
        aurocs.append(auroc)
        f1s.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        
        # Store in results dict
        results[f'{disease}_auroc'] = auroc
        results[f'{disease}_f1'] = f1
        results[f'{disease}_precision'] = precision
        results[f'{disease}_recall'] = recall
        
        # Store in per-disease dict
        per_disease_metrics[disease] = {
            'auroc': auroc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
        
        # Print
        print(f"{disease:<25} {auroc:>10.4f} {f1:>10.4f} {precision:>10.4f} {recall:>10.4f}")
    
    # Calculate macro averages
    macro_auroc = np.mean(aurocs)
    macro_f1 = np.mean(f1s)
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    
    # Store macro metrics
    results['macro_auroc'] = macro_auroc
    results['macro_f1'] = macro_f1
    results['macro_precision'] = macro_precision
    results['macro_recall'] = macro_recall
    
    macro_metrics = {
        'auroc': macro_auroc,
        'f1': macro_f1,
        'precision': macro_precision,
        'recall': macro_recall
    }
    
    # Print macro averages
    print("-" * 70)
    print(f"{'MACRO AVERAGE':<25} {macro_auroc:>10.4f} {macro_f1:>10.4f} {macro_precision:>10.4f} {macro_recall:>10.4f}")
    print("=" * 70)
    
    return results, per_disease_metrics, macro_metrics


# ============================================================================
# COMPARISON TABLE FUNCTION
# ============================================================================

def create_comparison_table(all_metrics, save_path=None):
    """
    Create a comparison table of multiple models
    
    Args:
        all_metrics: Dict of {model_name: results_dict}
        save_path: Path to save CSV (optional)
    
    Returns:
        comparison_df: Pandas DataFrame with comparison
    """
    
    # Extract model names
    model_names = list(all_metrics.keys())
    
    # Create rows for comparison
    rows = []
    
    # Macro metrics
    for metric in ['macro_auroc', 'macro_f1', 'macro_precision', 'macro_recall']:
        row = {'Metric': metric}
        for model_name in model_names:
            row[model_name] = all_metrics[model_name][metric]
        rows.append(row)
    
    # Per-disease metrics (AUROC and F1)
    # Get disease names from first model
    first_model = all_metrics[model_names[0]]
    disease_names = [k.replace('_auroc', '') for k in first_model.keys() if k.endswith('_auroc') and not k.startswith('macro')]
    
    for disease in disease_names:
        # AUROC
        row = {'Metric': f'{disease}_auroc'}
        for model_name in model_names:
            row[model_name] = all_metrics[model_name][f'{disease}_auroc']
        rows.append(row)
        
        # F1
        row = {'Metric': f'{disease}_f1'}
        for model_name in model_names:
            row[model_name] = all_metrics[model_name][f'{disease}_f1']
        rows.append(row)
    
    # Create DataFrame
    comparison_df = pd.DataFrame(rows)
    
    # Print
    print("\n" + "="*80)
    print("MODEL COMPARISON TABLE")
    print("="*80)
    print(comparison_df.to_string(index=False))
    
    # Save if requested
    if save_path:
        comparison_df.to_csv(save_path, index=False)
        print(f"\n Comparison table saved: {save_path}")
    
    return comparison_df