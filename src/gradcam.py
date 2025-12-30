import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from tqdm import tqdm

from models import initialize_model, ImagePreprocessor
from train import PropertyDataset
from config import OUTPUTS_DIR, DEVICE, PROCESSED_DATA_DIR


class GradCAM:

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, image, tabular_data, target_scaler=None):
        self.model.eval()

        image.requires_grad = True
        prediction = self.model(image, tabular_data)

        self.model.zero_grad()
        prediction.backward()

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = torch.mean(gradients, dim=(1, 2))

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()

        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (224, 224))

        pred_value = prediction.item()
        if target_scaler is not None:
            pred_value = target_scaler.inverse_transform([[pred_value]])[0][0]

        return cam, pred_value

    def visualize(self, original_image_path, cam, prediction, save_path=None):
        image = cv2.imread(str(original_image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        overlay = heatmap * 0.4 + image * 0.6
        overlay = np.uint8(overlay)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(image)
        axes[0].set_title('Original Satellite Image')
        axes[0].axis('off')

        axes[1].imshow(cam, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap\n(Red = High Influence)')
        axes[1].axis('off')

        axes[2].imshow(overlay)
        axes[2].set_title(f'Overlay\nPredicted Price: ${prediction:,.0f}')
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def generate_gradcam_visualizations(model, dataset, num_samples=20, device=DEVICE):
    print("\n" + "="*80)
    print("GENERATING GRAD-CAM VISUALIZATIONS")
    print("="*80)

    target_layer = model.cnn_branch.backbone[7][2].conv3
    grad_cam = GradCAM(model, target_layer)

    output_dir = OUTPUTS_DIR / 'visualizations' / 'gradcam'
    output_dir.mkdir(parents=True, exist_ok=True)

    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    for i, idx in enumerate(tqdm(indices, desc="Generating Grad-CAM")):
        image, tabular, target = dataset[idx]
        image_path = dataset.image_paths[idx]

        image = image.unsqueeze(0).to(device)
        tabular = tabular.unsqueeze(0).to(device)

        cam, prediction = grad_cam.generate_cam(image, tabular)

        save_path = output_dir / f'gradcam_sample_{i+1}.png'
        grad_cam.visualize(image_path, cam, prediction, save_path)

    print(f"\nSaved {len(indices)} Grad-CAM visualizations to {output_dir}")


def main():
    print("="*80)
    print("GRAD-CAM EXPLAINABILITY")
    print("="*80)

    print("\nLoading data and model...")
    data = np.load(PROCESSED_DATA_DIR / 'train_processed.npz', allow_pickle=True)
    val_dataset = PropertyDataset(data['X_val'], data['images_val'], data['y_val'])

    tabular_input_size = data['X_val'].shape[1]
    model, device = initialize_model(tabular_input_size, 'multimodal')

    checkpoint_path = OUTPUTS_DIR / 'models' / 'best_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    generate_gradcam_visualizations(model, val_dataset, num_samples=20, device=device)

    print("\n" + "="*80)
    print("GRAD-CAM GENERATION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()

