# =============================================================================
# GRAD-CAM VISUALIZATION
# =============================================================================

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

class GradCAM:
    """
    Grad-CAM implementation for visualizing model attention
    """
    def __init__(self, model, target_layer):
        """
        Args:
            model: PyTorch model
            target_layer: The layer to compute gradients from (e.g., model.features[-1])
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, image, disease_idx):
        """
        Generate Grad-CAM heatmap for a specific disease
        
        Args:
            image: Input image tensor (1, C, H, W)
            disease_idx: Index of disease to visualize
        
        Returns:
            heatmap: Grad-CAM heatmap (H, W)
        """
        # Forward pass
        self.model.eval()
        output = self.model(image)
        
        # Backward pass for specific disease
        self.model.zero_grad()
        target = output[0, disease_idx]
        target.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Compute weights (global average pooling of gradients)
        weights = gradients.mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=0)  # (H, W)
        
        # Apply ReLU (only positive influences)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy()


def visualize_gradcam(model, image_tensor, image_rgb, disease_idx, disease_name, 
                      prediction, true_label, save_path=None):
    """
    Create Grad-CAM visualization for a single image
    
    Args:
        model: Trained model
        image_tensor: Preprocessed image tensor (1, C, H, W)
        image_rgb: Original RGB image (H, W, 3) for overlay
        disease_idx: Index of disease
        disease_name: Name of disease
        prediction: Model prediction (probability)
        true_label: Ground truth label (0 or 1)
        save_path: Path to save visualization
    """
    # Get target layer (last conv layer before classifier)
    if hasattr(model, 'features'):
        # DenseNet
        target_layer = model.features[-1]
    elif hasattr(model, 'layer4'):
        # ResNet
        target_layer = model.layer4[-1]
    else:
        # SimpleCNN
        target_layer = model.conv5
    
    # Generate Grad-CAM
    gradcam = GradCAM(model, target_layer)
    heatmap = gradcam.generate_cam(image_tensor, disease_idx)
    
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (image_rgb.shape[1], image_rgb.shape[0]))
    
    # Create heatmap overlay
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay on original image
    overlay = (heatmap_colored * 0.4 + image_rgb * 0.6).astype(np.uint8)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image_rgb, cmap='gray')
    axes[0].set_title('Original X-Ray')
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(heatmap_resized, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    # Add prediction info
    label_text = "Positive" if true_label == 1 else "Negative"
    fig.suptitle(f'{disease_name}\nPrediction: {prediction:.3f} | Ground Truth: {label_text}', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved: {save_path}")
    
    plt.show()


def generate_gradcam_examples(model, test_loader, disease_names, device, 
                              num_examples=5, save_dir='experiments/results/gradcam'):
    """
    Generate Grad-CAM visualizations for multiple examples
    
    Args:
        model: Trained model
        test_loader: Test dataloader
        disease_names: List of disease names
        device: Device (cuda/cpu)
        num_examples: Number of examples per disease
        save_dir: Directory to save visualizations
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    
    print("="*80)
    print("GENERATING GRAD-CAM VISUALIZATIONS")
    print("="*80)
    
    # Get some test images
    images_batch, labels_batch = next(iter(test_loader))
    
    # Process each disease
    for disease_idx, disease_name in enumerate(disease_names):
        print(f"\n📊 {disease_name}:")
        
        # Find examples where disease is present
        positive_indices = torch.where(labels_batch[:, disease_idx] == 1)[0]
        
        if len(positive_indices) == 0:
            print(f"   ⚠️  No positive examples in this batch, skipping...")
            continue
        
        # Take first few positive examples
        num_to_show = min(num_examples, len(positive_indices))
        
        for i in range(num_to_show):
            idx = positive_indices[i].item()
            
            # Get image
            image_tensor = images_batch[idx:idx+1].to(device)
            true_label = labels_batch[idx, disease_idx].item()
            
            # Get prediction
            with torch.no_grad():
                output = model(image_tensor)
                prediction = torch.sigmoid(output)[0, disease_idx].item()
            
            # Convert image to RGB for visualization
            # Denormalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image_denorm = image_tensor[0].cpu() * std + mean
            image_denorm = torch.clamp(image_denorm, 0, 1)
            
            # Convert to RGB
            image_rgb = (image_denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            
            # Generate Grad-CAM
            save_path = f'{save_dir}/{disease_name.replace(" ", "_")}_example_{i+1}.png'
            visualize_gradcam(
                model=model,
                image_tensor=image_tensor,
                image_rgb=image_rgb,
                disease_idx=disease_idx,
                disease_name=disease_name,
                prediction=prediction,
                true_label=true_label,
                save_path=save_path
            )
        
        print(f"  Generated {num_to_show} examples")
    
    print(f"\nAll Grad-CAM visualizations saved to: {save_dir}")