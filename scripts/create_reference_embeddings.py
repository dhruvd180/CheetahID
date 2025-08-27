import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from collections import defaultdict
import pickle
import cv2
from ultralytics import YOLO

# Import the enhanced model
from train_embedder import CheetahFocusedNet

class CheetahAutoCropper:
    def __init__(self, model_path='yolov8n.pt'):
        """Initialize YOLO model for cheetah detection"""
        try:
            self.model = YOLO(model_path)
            print(f"Loaded YOLO model: {model_path}")
        except Exception as e:
            print(f"Warning: Could not load YOLO model ({e}). Falling back to center cropping.")
            self.model = None
    
    def detect_and_crop_cheetah_regions(self, image_path, target_regions=['tail', 'thigh']):
        """
        Detect cheetah and extract tail/thigh regions
        Returns list of cropped regions or original image if detection fails
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Cannot load image: {image_path}")
            
            h, w = img.shape[:2]
            
            if self.model is None:
                # Fallback: return center crop
                return [self._center_crop(img, 0.6)]
            
            # Run YOLO detection
            results = self.model(image_path, verbose=False)
            
            # Look for animal detections
            animal_classes = [15, 16, 17, 18, 19, 20, 21, 22, 23]  # Various animal classes in COCO
            
            best_detection = None
            best_confidence = 0
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        # Accept any animal class or high-confidence detection
                        if (cls in animal_classes or conf > 0.7) and conf > best_confidence:
                            best_confidence = conf
                            best_detection = box.xyxy[0].cpu().numpy()
            
            if best_detection is not None:
                x1, y1, x2, y2 = map(int, best_detection)
                
                # Extract different body regions based on cheetah anatomy
                crops = []
                
                # Ensure bounding box is valid
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                
                if bbox_w < 50 or bbox_h < 50:  # Too small detection
                    return [self._center_crop(img, 0.6)]
                
                if 'tail' in target_regions:
                    # Tail region: typically rear-right area of the detection
                    tail_x1 = x1 + int(bbox_w * 0.6)  # Right 40% of detection
                    tail_y1 = y1 + int(bbox_h * 0.2)  # Skip top 20%
                    tail_x2 = min(w, tail_x1 + int(bbox_w * 0.5))
                    tail_y2 = min(h, tail_y1 + int(bbox_h * 0.6))
                    
                    if tail_x2 > tail_x1 and tail_y2 > tail_y1:
                        tail_crop = img[tail_y1:tail_y2, tail_x1:tail_x2]
                        if tail_crop.size > 0:
                            crops.append(self._opencv_to_pil(tail_crop))
                
                if 'thigh' in target_regions:
                    # Thigh region: middle-left area of detection
                    thigh_x1 = x1 + int(bbox_w * 0.1)  # Start 10% in
                    thigh_y1 = y1 + int(bbox_h * 0.3)  # Start 30% down
                    thigh_x2 = min(w, thigh_x1 + int(bbox_w * 0.6))
                    thigh_y2 = min(h, thigh_y1 + int(bbox_h * 0.5))
                    
                    if thigh_x2 > thigh_x1 and thigh_y2 > thigh_y1:
                        thigh_crop = img[thigh_y1:thigh_y2, thigh_x1:thigh_x2]
                        if thigh_crop.size > 0:
                            crops.append(self._opencv_to_pil(thigh_crop))
                
                # If we got valid crops, return them
                if crops:
                    return crops
                else:
                    # Fallback to full detection
                    full_crop = img[y1:y2, x1:x2]
                    return [self._opencv_to_pil(full_crop)] if full_crop.size > 0 else [self._center_crop_pil(image_path, 0.8)]
            
            # No detection found, use center crop
            return [self._center_crop_pil(image_path, 0.6)]
            
        except Exception as e:
            print(f"Error in autocropping {image_path}: {e}")
            # Emergency fallback
            try:
                return [self._center_crop_pil(image_path, 0.6)]
            except:
                return None
    
    def _opencv_to_pil(self, cv_img):
        """Convert OpenCV image to PIL"""
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_img)
    
    def _center_crop(self, img, crop_ratio=0.6):
        """Fallback center crop for OpenCV image"""
        h, w = img.shape[:2]
        crop_h = int(h * crop_ratio)
        crop_w = int(w * crop_ratio)
        
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        
        return img[start_h:start_h + crop_h, start_w:start_w + crop_w]
    
    def _center_crop_pil(self, image_path, crop_ratio=0.6):
        """Fallback center crop returning PIL image"""
        img = Image.open(image_path).convert('RGB')
        w, h = img.size
        crop_w = int(w * crop_ratio)
        crop_h = int(h * crop_ratio)
        
        left = (w - crop_w) // 2
        top = (h - crop_h) // 2
        
        return img.crop((left, top, left + crop_w, top + crop_h))

def load_model(model_path, device):
    """Load the trained model with proper error handling"""
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Check if it's a full checkpoint or just state dict
        if 'model_state_dict' in checkpoint:
            embedding_dim = checkpoint.get('embedding_dim', 512)
            model = CheetahFocusedNet(embedding_dim=embedding_dim).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded enhanced model from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            # Old format - assume default embedding dim
            model = CheetahFocusedNet(embedding_dim=512).to(device)
            model.load_state_dict(checkpoint)
            print("Loaded model (legacy format)")
            
        model.eval()
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure the model path is correct and the model was trained with the enhanced architecture")
        raise

def create_enhanced_reference_embeddings(model_path, gallery_dir, device, cache_file=None, use_cropping=True):
    """Create reference embeddings with cropping and TTA"""
    
    # Check cache first
    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached embeddings from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    model = load_model(model_path, device)
    cropper = CheetahAutoCropper() if use_cropping else None
    
    # Enhanced transforms with multiple scales for better representation
    tta_transforms = [
        # Original
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        # Horizontal flip
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        # Different scale
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        # Slight brightness adjustment
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    ]

    embeddings = {}
    individual_stats = defaultdict(list)
    
    with torch.no_grad():
        for individual in sorted(os.listdir(gallery_dir)):
            individual_path = os.path.join(gallery_dir, individual)
            if not os.path.isdir(individual_path):
                continue
                
            print(f"Processing individual: {individual}")
            emb_list = []
            image_count = 0
            crop_count = 0
            
            for img_file in sorted(os.listdir(individual_path)):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                    
                img_path = os.path.join(individual_path, img_file)
                
                try:
                    # Get crops if cropping is enabled
                    if use_cropping and cropper:
                        crops = cropper.detect_and_crop_cheetah_regions(img_path)
                        if not crops:
                            crops = [Image.open(img_path).convert('RGB')]
                    else:
                        crops = [Image.open(img_path).convert('RGB')]
                    
                    # Process each crop
                    for crop_idx, crop in enumerate(crops):
                        # Apply multiple TTA transforms for robust embeddings
                        crop_embeddings = []
                        
                        for transform in tta_transforms:
                            try:
                                img_t = transform(crop).unsqueeze(0).to(device)
                                emb = model(img_t).cpu()
                                crop_embeddings.append(emb)
                            except Exception as e:
                                print(f"Error processing transform for {img_file} crop {crop_idx}: {e}")
                                continue
                        
                        if crop_embeddings:
                            # Average embeddings from different augmentations
                            avg_emb = torch.cat(crop_embeddings, dim=0).mean(dim=0, keepdim=True)
                            emb_list.append(avg_emb)
                            crop_count += 1
                    
                    image_count += 1
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
            
            if emb_list:
                # Stack all embeddings for this individual
                individual_embeddings = torch.cat(emb_list, dim=0)  # [num_crops, embedding_dim]
                
                # Compute statistics
                mean_emb = individual_embeddings.mean(dim=0)  # Average embedding
                std_emb = individual_embeddings.std(dim=0)    # Standard deviation
                
                # Store both mean and std for potential uncertainty estimation
                embeddings[individual] = {
                    'mean': mean_emb.squeeze(),  # Remove batch dimension
                    'std': std_emb.squeeze(),    # Remove batch dimension  
                    'num_images': image_count,
                    'num_crops': crop_count,
                    'all_embeddings': individual_embeddings  # Keep all for advanced matching
                }
                
                individual_stats[individual] = {'images': image_count, 'crops': crop_count}
                print(f"  -> Created embedding from {image_count} images ({crop_count} crops)")
            else:
                print(f"  -> No valid images/crops found for {individual}")

    print(f"\nSummary:")
    print(f"Created embeddings for {len(embeddings)} individuals")
    for ind, stats in individual_stats.items():
        print(f"  {ind}: {stats['images']} images, {stats['crops']} crops")
    
    # Cache the embeddings
    if cache_file:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"Cached embeddings to {cache_file}")
    
    return embeddings

def update_reference_embeddings(model_path, gallery_dir, new_individual_dir, 
                              cache_file, device, use_cropping=True):
    """Add a new individual to existing reference embeddings"""
    
    # Load existing embeddings
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            embeddings = pickle.load(f)
    else:
        embeddings = {}
    
    # Get new individual name
    new_individual = os.path.basename(new_individual_dir)
    
    print(f"Adding new individual: {new_individual}")
    
    # Create temporary gallery structure
    temp_dir = os.path.dirname(new_individual_dir)
    new_embeddings = create_enhanced_reference_embeddings(
        model_path, temp_dir, device, cache_file=None, use_cropping=use_cropping
    )
    
    # Add to existing embeddings
    if new_individual in new_embeddings:
        embeddings[new_individual] = new_embeddings[new_individual]
        
        # Save updated embeddings
        with open(cache_file, 'wb') as f:
            pickle.dump(embeddings, f)
        
        print(f"Added {new_individual} to reference embeddings")
    else:
        print(f"Failed to create embedding for {new_individual}")
    
    return embeddings

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='models/cheetah_cropped_embedder.pt')
    parser.add_argument('--gallery_dir', default='data/raw')
    parser.add_argument('--cache_file', default='models/reference_embeddings_cropped.pkl')
    parser.add_argument('--force_recreate', action='store_true', 
                       help='Force recreation of embeddings even if cache exists')
    parser.add_argument('--no_cropping', action='store_true', help='Disable auto-cropping')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.force_recreate and os.path.exists(args.cache_file):
        os.remove(args.cache_file)
        print("Removed existing cache file")
    
    use_cropping = not args.no_cropping
    print(f"Auto-cropping: {'Enabled' if use_cropping else 'Disabled'}")
    
    embeddings = create_enhanced_reference_embeddings(
        args.model_path, args.gallery_dir, device, args.cache_file, use_cropping
    )
    print(f"Reference embeddings ready for {len(embeddings)} individuals.")