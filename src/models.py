import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
from config import IMG_HEIGHT, IMG_WIDTH, IMAGE_EMBEDDING_SIZE, PRETRAINED_MODEL, DEVICE


class ImagePreprocessor:

    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_image(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            return self.transform(image)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return torch.zeros(3, IMG_HEIGHT, IMG_WIDTH)

    def load_batch(self, image_paths):
        images = [self.load_image(path) for path in image_paths]
        return torch.stack(images)


class CNNBranch(nn.Module):

    def __init__(self, embedding_size=IMAGE_EMBEDDING_SIZE, pretrained=True):
        super(CNNBranch, self).__init__()

        if PRETRAINED_MODEL == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            backbone_output_size = 2048
        elif PRETRAINED_MODEL == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            backbone_output_size = 2048
        else:
            raise ValueError(f"Unsupported model: {PRETRAINED_MODEL}")

        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_output_size, embedding_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(embedding_size)
        )

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.projection(features)
        return embeddings


class TabularBranch(nn.Module):

    def __init__(self, input_size, embedding_size=IMAGE_EMBEDDING_SIZE):
        super(TabularBranch, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),

            nn.Linear(512, embedding_size),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_size),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        return self.network(x)


class MultimodalFusion(nn.Module):

    def __init__(self, embedding_size=IMAGE_EMBEDDING_SIZE):
        super(MultimodalFusion, self).__init__()

        combined_size = embedding_size * 2

        self.fusion = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),

            nn.Linear(128, 1)
        )

    def forward(self, image_embedding, tabular_embedding):
        combined = torch.cat([image_embedding, tabular_embedding], dim=1)
        prediction = self.fusion(combined)
        return prediction


class MultimodalPropertyValueModel(nn.Module):

    def __init__(self, tabular_input_size, embedding_size=IMAGE_EMBEDDING_SIZE):
        super(MultimodalPropertyValueModel, self).__init__()

        self.cnn_branch = CNNBranch(embedding_size=embedding_size)
        self.tabular_branch = TabularBranch(input_size=tabular_input_size, embedding_size=embedding_size)
        self.fusion = MultimodalFusion(embedding_size=embedding_size)

    def forward(self, images, tabular_data):
        image_features = self.cnn_branch(images)
        tabular_features = self.tabular_branch(tabular_data)
        predictions = self.fusion(image_features, tabular_features)
        return predictions

    def get_embeddings(self, images, tabular_data):
        image_features = self.cnn_branch(images)
        tabular_features = self.tabular_branch(tabular_data)
        return image_features, tabular_features


class TabularOnlyModel(nn.Module):

    def __init__(self, input_size):
        super(TabularOnlyModel, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),

            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.network(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_model(tabular_input_size, model_type='multimodal'):
    if model_type == 'multimodal':
        model = MultimodalPropertyValueModel(tabular_input_size)
    elif model_type == 'tabular_only':
        model = TabularOnlyModel(tabular_input_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print(f"\nInitialized {model_type} model")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Device: {device}")

    return model, device


if __name__ == "__main__":
    print("Testing model architectures...\n")

    batch_size = 4
    num_tabular_features = 25

    dummy_images = torch.randn(batch_size, 3, IMG_HEIGHT, IMG_WIDTH)
    dummy_tabular = torch.randn(batch_size, num_tabular_features)

    print("="*80)
    print("MULTIMODAL MODEL")
    print("="*80)
    model, device = initialize_model(num_tabular_features, 'multimodal')

    dummy_images = dummy_images.to(device)
    dummy_tabular = dummy_tabular.to(device)

    predictions = model(dummy_images, dummy_tabular)
    print(f"Input shapes: images={dummy_images.shape}, tabular={dummy_tabular.shape}")
    print(f"Output shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:3].detach().cpu().numpy().flatten()}")

    print("\n" + "="*80)
    print("BASELINE MODEL (Tabular Only)")
    print("="*80)
    baseline_model, device = initialize_model(num_tabular_features, 'tabular_only')
    predictions_baseline = baseline_model(dummy_tabular)
    print(f"Input shape: {dummy_tabular.shape}")
    print(f"Output shape: {predictions_baseline.shape}")

    print("\n✓ All models working correctly!")

