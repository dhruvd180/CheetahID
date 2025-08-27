import os
import random
import shutil
from collections import defaultdict
import numpy as np
import cv2
from PIL import Image
import torch
from ultralytics import YOLO

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
            
            # Look for animal detections (assuming class 0 is person, we want animals)
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
                    
                    tail_x1 = x1 + int(bbox_w * 0.6) 
                    tail_y1 = y1 + int(bbox_h * 0.2)  
                    tail_x2 = min(w, tail_x1 + int(bbox_w * 0.5))
                    tail_y2 = min(h, tail_y1 + int(bbox_h * 0.6))
                    
                    if tail_x2 > tail_x1 and tail_y2 > tail_y1:
                        tail_crop = img[tail_y1:tail_y2, tail_x1:tail_x2]
                        if tail_crop.size > 0:
                            crops.append(tail_crop)
                
                if 'thigh' in target_regions:
                    thigh_x1 = x1 + int(bbox_w * 0.1) 
                    thigh_y1 = y1 + int(bbox_h * 0.3) 
                    thigh_x2 = min(w, thigh_x1 + int(bbox_w * 0.6))
                    thigh_y2 = min(h, thigh_y1 + int(bbox_h * 0.5))
                    
                    if thigh_x2 > thigh_x1 and thigh_y2 > thigh_y1:
                        thigh_crop = img[thigh_y1:thigh_y2, thigh_x1:thigh_x2]
                        if thigh_crop.size > 0:
                            crops.append(thigh_crop)
                
               
                if crops:
                    return crops
                else:
                    
                    full_crop = img[y1:y2, x1:x2]
                    return [full_crop] if full_crop.size > 0 else [self._center_crop(img, 0.8)]
            
            
            return [self._center_crop(img, 0.6)]
            
        except Exception as e:
            print(f"Error in autocropping {image_path}: {e}")
            
            try:
                img = cv2.imread(image_path)
                return [self._center_crop(img, 0.6)]
            except:
                return None
    
    def _center_crop(self, img, crop_ratio=0.6):
        """Fallback center crop"""
        h, w = img.shape[:2]
        crop_h = int(h * crop_ratio)
        crop_w = int(w * crop_ratio)
        
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        
        return img[start_h:start_h + crop_h, start_w:start_w + crop_w]

def get_all_individuals(data_dir):
    individuals = []
    for d in os.listdir(data_dir):
        path = os.path.join(data_dir, d)
        if os.path.isdir(path):
            img_count = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if img_count >= 2:
                individuals.append(d)
    return individuals

def save_crop_as_jpg(crop, output_path):
    """Save OpenCV image crop as JPG"""
    try:

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        

        w, h = pil_img.size
        if w < 224 or h < 224:

            scale = max(224/w, 224/h)
            new_w, new_h = int(w * scale), int(h * scale)
            pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        
        pil_img.save(output_path, 'JPEG', quality=90)
        return True
    except Exception as e:
        print(f"Error saving crop: {e}")
        return False

def generate_balanced_triplets_with_cropping(raw_dir, output_dir, num_triplets, use_cropping=True):

    cropper = CheetahAutoCropper() if use_cropping else None
    
    individuals = get_all_individuals(raw_dir)
    print(f"Found {len(individuals)} individuals with sufficient images")
    if len(individuals) < 2:
        raise ValueError("Need at least 2 individuals with 2+ images each")

    images_per_individual = defaultdict(list)
    

    for ind in individuals:
        ind_dir = os.path.join(raw_dir, ind)
        print(f"Processing {ind}...")
        
        for fname in os.listdir(ind_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(ind_dir, fname)
                
                if use_cropping and cropper:
                    # Get cropped regions
                    crops = cropper.detect_and_crop_cheetah_regions(img_path)
                    
                    if crops:
                        # Store multiple crops per image
                        for i, crop in enumerate(crops):
                            crop_key = f"{fname}_crop_{i}"
                            images_per_individual[ind].append((img_path, crop, crop_key))
                    else:
                        # Fallback to original image
                        images_per_individual[ind].append((img_path, None, fname))
                else:
                    # Use original images
                    images_per_individual[ind].append((img_path, None, fname))

    # Prepare output dirs
    for folder in ['anchor', 'positive', 'negative']:
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

    triplet_count = 0
    individuals_usage = defaultdict(int)
    target_usage_per_individual = num_triplets // len(individuals)
    attempts = 0
    max_attempts = num_triplets * 3

    while triplet_count < num_triplets and attempts < max_attempts:
        attempts += 1

        # Balanced sampling weights
        weights = []
        for ind in individuals:
            rem = max(1, target_usage_per_individual - individuals_usage[ind])
            weights.append(rem)
        total = sum(weights)
        weights = [w / total for w in weights] if total > 0 else [1 / len(individuals)] * len(individuals)

        anchor_ind = np.random.choice(individuals, p=weights)
        anchor_data = images_per_individual[anchor_ind]
        if len(anchor_data) < 2:
            continue

        # Sample anchor and positive from same individual
        anchor_item, positive_item = random.sample(anchor_data, 2)
        anchor_path, anchor_crop, anchor_key = anchor_item
        positive_path, positive_crop, positive_key = positive_item

        # Sample negative from different individual
        negative_candidates = [i for i in individuals if i != anchor_ind]
        if not negative_candidates:
            continue
        negative_ind = random.choice(negative_candidates)
        negative_data = images_per_individual[negative_ind]
        negative_path, negative_crop, negative_key = random.choice(negative_data)

        # Save triplet
        file_name = f"{triplet_count + 1:05d}.jpg"
        
        try:
            success = True
            
            # Save anchor
            if anchor_crop is not None:
                success &= save_crop_as_jpg(anchor_crop, os.path.join(output_dir, 'anchor', file_name))
            else:
                shutil.copy(anchor_path, os.path.join(output_dir, 'anchor', file_name))
            
            # Save positive
            if positive_crop is not None:
                success &= save_crop_as_jpg(positive_crop, os.path.join(output_dir, 'positive', file_name))
            else:
                shutil.copy(positive_path, os.path.join(output_dir, 'positive', file_name))
            
            # Save negative
            if negative_crop is not None:
                success &= save_crop_as_jpg(negative_crop, os.path.join(output_dir, 'negative', file_name))
            else:
                shutil.copy(negative_path, os.path.join(output_dir, 'negative', file_name))
            
            if success:
                individuals_usage[anchor_ind] += 1
                triplet_count += 1
                if triplet_count % 100 == 0:
                    print(f"Generated {triplet_count}/{num_triplets} triplets...")
            
        except Exception as e:
            print(f"Error saving triplet {triplet_count}: {e}")
            continue

    print(f"Generated {triplet_count} triplets in {output_dir}")
    print("Usage per individual:")
    for ind, usage in individuals_usage.items():
        print(f"  {ind}: {usage} times")

def generate_train_val_split(triplets_dir, train_ratio=0.8):
    anchor_files = sorted(os.listdir(os.path.join(triplets_dir, 'anchor')))
    num_triplets = len(anchor_files)
    num_train = int(num_triplets * train_ratio)
    indices = list(range(num_triplets))
    random.shuffle(indices)
    train_indices = set(indices[:num_train])
    val_indices = set(indices[num_train:])

    for split in ['train', 'val']:
        for folder in ['anchor', 'positive', 'negative']:
            os.makedirs(os.path.join(triplets_dir, split, folder), exist_ok=True)

    for i, filename in enumerate(anchor_files):
        split_dir = 'train' if i in train_indices else 'val'
        for folder in ['anchor', 'positive', 'negative']:
            src = os.path.join(triplets_dir, folder, filename)
            dst = os.path.join(triplets_dir, split, folder, filename)
            shutil.move(src, dst)

    for folder in ['anchor', 'positive', 'negative']:
        shutil.rmtree(os.path.join(triplets_dir, folder))

    print(f"Split into {num_train} training and {len(val_indices)} validation triplets")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, default='data/raw')
    parser.add_argument('--output_dir', type=str, default='data/triplets')
    parser.add_argument('--num_triplets', type=int, default=2000)
    parser.add_argument('--create_split', action='store_true')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--no_cropping', action='store_true', help='Disable auto-cropping')
    parser.add_argument('--yolo_model', type=str, default='yolov8n.pt', help='YOLO model path')
    args = parser.parse_args()

    use_cropping = not args.no_cropping
    print(f"Auto-cropping: {'Enabled' if use_cropping else 'Disabled'}")

    generate_balanced_triplets_with_cropping(args.raw_dir, args.output_dir, args.num_triplets, use_cropping)
    if args.create_split:
        generate_train_val_split(args.output_dir, args.train_ratio)