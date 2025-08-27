import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import random
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
import cv2
class imgAugmentation:
    def __init__(self):
        pass

    def __call__(self, img):
        augmentation_choice = random.choice([
            'contrast', 'brightness', 'slight_blur', 'gamma', 'none'
        ])

        if augmentation_choice == 'contrast':
            img = ImageEnhance.Contrast(img).enhance(random.uniform(1.1, 1.5))
        elif augmentation_choice == 'brightness':
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
        elif augmentation_choice == 'slight_blur':
            if random.random() < 0.3:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.4)))
        elif augmentation_choice == 'gamma':
            img_np = np.array(img).astype(np.float32) / 255.0
            gamma = random.uniform(0.7, 1.3)
            img_np = np.power(img_np, gamma)
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_np)
        return img
class CheetahFocusedNet(nn.Module):
    """Enhanced model optimized for cheetah spot patterns"""
    def __init__(self, embedding_dim=512):
        super().__init__()
        # Use EfficientNet-B3 for better feature extraction
        backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        
        # Multi-scale feature extraction for spot patterns
        self.scale_1 = nn.Sequential(
            nn.Conv2d(1536, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.scale_2 = nn.Sequential(
            nn.Conv2d(1536, 256, 5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.scale_3 = nn.Sequential(
            nn.Conv2d(1536, 256, 7, padding=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Enhanced spatial attention for spot patterns
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(768, 256, 1),  # 768 = 256*3 from multi-scale
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(768, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 768, 1),
            nn.Sigmoid()
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        
        # Enhanced embedding head
        self.embedding_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768 * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
    def forward(self, x):
        # Extract features
        features = self.features(x)
        
        # Multi-scale feature extraction
        scale1_feat = self.scale_1(features)  # [B, 256, H, W]
        scale2_feat = self.scale_2(features)  # [B, 256, H, W]
        scale3_feat = self.scale_3(features)  
        
        # Concatenate multi-scale features
        multi_scale = torch.cat([scale1_feat, scale2_feat, scale3_feat], dim=1)  # [B, 768, H, W]
        
        # Apply attention mechanisms
        spatial_att = self.spatial_attention(multi_scale)  # [B, 1, H, W]
        channel_att = self.channel_attention(multi_scale)  # [B, 768, 1, 1]
        
        # Apply attention
        attended = multi_scale * spatial_att * channel_att
        
        # Global pooling (both avg and max for richer representation)
        avg_pooled = self.avgpool(attended)
        max_pooled = self.maxpool(attended)
        pooled = torch.cat([avg_pooled, max_pooled], dim=1)  # [B, 768*2, 1, 1]
        
        pooled = torch.flatten(pooled, 1)
        emb = self.embedding_head(pooled)
        emb = F.normalize(emb, p=2, dim=1).float()  # L2 normalize
        return emb
class TripletDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.anchor_dir = os.path.join(root_dir, 'anchor')
        self.positive_dir = os.path.join(root_dir, 'positive')
        self.negative_dir = os.path.join(root_dir, 'negative')
        self.triplets = sorted([f for f in os.listdir(self.anchor_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        self.transform = transform

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        filename = self.triplets[idx]
        
        try:
            anchor = Image.open(os.path.join(self.anchor_dir, filename)).convert('RGB')
            positive = Image.open(os.path.join(self.positive_dir, filename)).convert('RGB')
            negative = Image.open(os.path.join(self.negative_dir, filename)).convert('RGB')
        except Exception as e:
            print(f"Error loading triplet {filename}: {e}")
            # Return a dummy triplet to avoid crashing
            dummy_img = Image.new('RGB', (224, 224), color=(128, 128, 128))
            anchor = positive = negative = dummy_img

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative
class AdvancedTripletLoss(nn.Module):
    def __init__(self, margin=0.3, hard_mining=True, adaptive_margin=True):
        super().__init__()
        self.margin = margin
        self.hard_mining = hard_mining
        self.adaptive_margin = adaptive_margin
        
    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)

        if self.adaptive_margin:
            # Adaptive margin based on current batch statistics
            batch_margin = self.margin + 0.1 * (pos_dist.mean() - neg_dist.mean()).clamp(min=0)
        else:
            batch_margin = self.margin

        if self.hard_mining:
            # Focus on hardest examples
            hardest_positive_dist = pos_dist.max()
            hardest_negative_dist = neg_dist.min()
            hard_triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + batch_margin)
            regular_triplet_loss = F.relu(pos_dist - neg_dist + batch_margin).mean()
            loss = 0.7 * regular_triplet_loss + 0.3 * hard_triplet_loss
        else:
            loss = F.relu(pos_dist - neg_dist + batch_margin).mean()

        return loss
def train_epoch(model, device, dataloader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    num_triplets = 0
    
    for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
        try:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            optimizer.zero_grad()
            
            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)
            
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_triplets += anchor.size(0)
            
            if batch_idx % 10 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(anchor)}/{len(dataloader.dataset)}] Loss: {loss.item():.6f}')
                
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    print(f'==> Epoch {epoch} Average Training Loss: {avg_loss:.6f}')
    return avg_loss
def validate_epoch(model, device, dataloader, criterion, epoch):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
            try:
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                anchor_emb = model(anchor)
                positive_emb = model(positive)
                negative_emb = model(negative)
                loss = criterion(anchor_emb, positive_emb, negative_emb)
                total_loss += loss.item()
                num_batches += 1
            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {e}")
                continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    print(f'==> Epoch {epoch} Average Validation Loss: {avg_loss:.6f}')
    return avg_loss
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/triplets')
    parser.add_argument('--model_path', default='models/cheetah_cropped_embedder.pt')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--margin', type=float, default=0.3)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Enhanced transforms for cropped regions
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        imgAugmentation(),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),  # Slightly more rotation for cropped regions
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.3),  # Add perspective changes
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')

    if os.path.exists(train_dir) and os.path.exists(val_dir):
        print("Using existing train/val split")
        train_dataset = TripletDataset(train_dir, transform=train_transform)
        val_dataset = TripletDataset(val_dir, transform=val_transform)
    else:
        print("No train/val split found, using all data for training")
        train_dataset = TripletDataset(args.data_dir, transform=train_transform)
        val_dataset = None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # Use enhanced model
    model = CheetahFocusedNet(embedding_dim=args.embedding_dim).to(device)
    criterion = AdvancedTripletLoss(margin=args.margin, hard_mining=True, adaptive_margin=True)
    
    # Use AdamW with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=args.patience, factor=0.5)

    # Ensure model directory exists
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, device, train_loader, optimizer, criterion, epoch)

        val_loss = None
        if val_loader:
            val_loss = validate_epoch(model, device, val_loader, criterion, epoch)
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'embedding_dim': args.embedding_dim,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, args.model_path)
                print(f"Saved best model at epoch {epoch}")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        else:
            scheduler.step(train_loss)
            if epoch % 10 == 0:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'embedding_dim': args.embedding_dim,
                    'train_loss': train_loss,
                }, args.model_path)
                print(f"Saved model at epoch {epoch} (no val set)")

    print("Training completed!")
if __name__ == "__main__":
    main()