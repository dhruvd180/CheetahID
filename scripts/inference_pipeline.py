import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import pickle
from collections import defaultdict
import cv2
from ultralytics import YOLO

# Import the enhanced model and cropper
from train_embedder import CheetahFocusedNet
from create_reference_embeddings import load_model, CheetahAutoCropper

class EnhancedCheetahIdentifier:
    def __init__(self, model_path, reference_embeddings_path, device, use_cropping=True):
        self.device = device
        self.model = load_model(model_path, device)
        self.use_cropping = use_cropping
        self.cropper = CheetahAutoCropper() if use_cropping else None
        
        # Load reference embeddings
        with open(reference_embeddings_path, 'rb') as f:
            self.reference_embeddings = pickle.load(f)
        
        print(f"Loaded reference embeddings for {len(self.reference_embeddings)} individuals")
        if use_cropping:
            print("Auto-cropping enabled for tail/thigh regions")
        
        # Test time augmentation transforms
        self.tta_transforms = [
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        ]
        
        # Compute rejection thresholds based on reference embeddings
        self.rejection_thresholds = self._compute_rejection_thresholds()
        
    def _compute_rejection_thresholds(self):
        """Compute dynamic rejection thresholds based on reference data"""
        print("Computing rejection thresholds...")
        
        all_similarities = []
        individual_similarities = {}
        
        # Compute intra-individual similarities (same cheetah comparisons)
        for individual, ref_data in self.reference_embeddings.items():
            if 'all_embeddings' in ref_data and ref_data['all_embeddings'].shape[0] > 1:
                embeddings = ref_data['all_embeddings']
                similarities = []
                
                # Compare all pairs within the same individual
                for i in range(embeddings.shape[0]):
                    for j in range(i + 1, embeddings.shape[0]):
                        sim = F.cosine_similarity(embeddings[i:i+1], embeddings[j:j+1]).item()
                        similarities.append(sim)
                        all_similarities.append(sim)
                
                if similarities:
                    individual_similarities[individual] = {
                        'mean': np.mean(similarities),
                        'min': np.min(similarities),
                        'std': np.std(similarities)
                    }
        
        if all_similarities:
            overall_mean = np.mean(all_similarities)
            overall_std = np.std(all_similarities)
            
            # Conservative threshold: mean - 2*std
            conservative_threshold = overall_mean - 2 * overall_std
            # Moderate threshold: mean - 1*std  
            moderate_threshold = overall_mean - 1 * overall_std
            # Liberal threshold: mean - 0.5*std
            liberal_threshold = overall_mean - 0.5 * overall_std
            
            thresholds = {
                'conservative': max(0.3, conservative_threshold),  # Don't go below 0.3
                'moderate': max(0.4, moderate_threshold),
                'liberal': max(0.5, liberal_threshold),
                'stats': {
                    'mean': overall_mean,
                    'std': overall_std,
                    'n_comparisons': len(all_similarities)
                }
            }
            
            print(f"Rejection thresholds computed from {len(all_similarities)} intra-individual comparisons:")
            print(f"  Conservative: {thresholds['conservative']:.3f}")
            print(f"  Moderate: {thresholds['moderate']:.3f}")  
            print(f"  Liberal: {thresholds['liberal']:.3f}")
            print(f"  Reference mean similarity: {overall_mean:.3f} ± {overall_std:.3f}")
            
            return thresholds
        else:
            print("Warning: Could not compute rejection thresholds, using defaults")
            return {
                'conservative': 0.3,
                'moderate': 0.4,
                'liberal': 0.5,
                'stats': {'mean': 0.7, 'std': 0.2, 'n_comparisons': 0}
            }
    
    def extract_query_embedding(self, image_path):
        """Extract robust embedding from query image using cropping and TTA"""
        try:
            if self.use_cropping and self.cropper:
                # Get cropped regions
                crops = self.cropper.detect_and_crop_cheetah_regions(image_path)
                if not crops:
                    crops = [Image.open(image_path).convert('RGB')]
            else:
                crops = [Image.open(image_path).convert('RGB')]
        except Exception as e:
            raise ValueError(f"Cannot load/process image {image_path}: {e}")
        
        all_embeddings = []
        
        with torch.no_grad():
            # Process each crop
            for crop in crops:
                crop_embeddings = []
                
                # Apply TTA to each crop
                for transform in self.tta_transforms:
                    try:
                        img_t = transform(crop).unsqueeze(0).to(self.device)
                        emb = self.model(img_t).cpu()
                        crop_embeddings.append(emb)
                    except Exception as e:
                        print(f"Error in TTA transform: {e}")
                        continue
                
                if crop_embeddings:
                    # Average embeddings from different augmentations for this crop
                    avg_crop_emb = torch.cat(crop_embeddings, dim=0).mean(dim=0, keepdim=True)
                    all_embeddings.append(avg_crop_emb)
        
        if not all_embeddings:
            raise ValueError("Failed to extract any embeddings from image")
        
        # Average embeddings from all crops
        final_embedding = torch.cat(all_embeddings, dim=0).mean(dim=0, keepdim=True)
        return final_embedding
    
    def compute_similarity_scores(self, query_embedding):
        """Compute similarity scores against all reference individuals"""
        scores = {}
        
        for individual, ref_data in self.reference_embeddings.items():
            ref_embedding = ref_data['mean']
            
            # Ensure both embeddings have the same shape
            if query_embedding.dim() == 2:
                query_emb = query_embedding.squeeze(0)  # Remove batch dim
            else:
                query_emb = query_embedding
                
            if ref_embedding.dim() == 0:
                ref_embedding = ref_embedding.unsqueeze(0)
            
            # Use cosine similarity (better for normalized embeddings)
            similarity = F.cosine_similarity(query_emb.unsqueeze(0), ref_embedding.unsqueeze(0)).item()
            
            # Convert to distance for consistency (0 = identical, 2 = opposite)
            distance = 1 - similarity
            
            scores[individual] = {
                'similarity': similarity,
                'distance': distance,
                'num_ref_images': ref_data.get('num_images', 1),
                'num_ref_crops': ref_data.get('num_crops', 1)
            }
        
        return scores
    
    def advanced_matching(self, query_embedding, top_k=3):
        """Advanced matching with confidence estimation"""
        scores = self.compute_similarity_scores(query_embedding)
        
        # Sort by similarity (higher is better)
        sorted_matches = sorted(scores.items(), 
                              key=lambda x: x[1]['similarity'], 
                              reverse=True)
        
        results = []
        for individual, score_data in sorted_matches[:top_k]:
            similarity = score_data['similarity']
            distance = score_data['distance']
            num_images = score_data['num_ref_images']
            num_crops = score_data['num_ref_crops']
            
            # Enhanced confidence estimation
            confidence = self.estimate_confidence(similarity, distance, num_images, num_crops, sorted_matches)
            
            results.append({
                'individual': individual,
                'similarity': similarity,
                'distance': distance,
                'confidence': confidence,
                'num_reference_images': num_images,
                'num_reference_crops': num_crops
            })
        
        return results
    
    def estimate_confidence(self, similarity, distance, num_images, num_crops, all_matches):
        """Enhanced confidence estimation"""
        
        # Base confidence from similarity score
        base_confidence = max(0, similarity)  # Cosine similarity is [-1, 1]
        
        # Boost confidence if we have more reference data
        image_boost = min(0.1, num_images * 0.015)  # Up to 0.1 boost
        crop_boost = min(0.05, num_crops * 0.005)   # Up to 0.05 boost
        
        # Penalize if the gap between top matches is small (ambiguous case)
        gap_boost = 0
        if len(all_matches) > 1:
            top_sim = all_matches[0][1]['similarity']
            second_sim = all_matches[1][1]['similarity']
            gap = top_sim - second_sim
            gap_boost = min(0.2, gap * 3)  # Up to 0.2 boost for large gaps
        
        # Boost confidence if similarity is well above the rejection threshold
        threshold_boost = 0
        if similarity > self.rejection_thresholds['liberal']:
            excess = similarity - self.rejection_thresholds['liberal']
            threshold_boost = min(0.15, excess * 2)
        
        # Combine factors
        confidence = base_confidence + image_boost + crop_boost + gap_boost + threshold_boost
        
        # Normalize to [0, 1]
        confidence = max(0, min(1, confidence))
        
        return confidence
    
    def determine_rejection_level(self, best_similarity):
        """Determine if query should be rejected and at what level"""
        if best_similarity >= self.rejection_thresholds['liberal']:
            return 'accept', 'liberal'
        elif best_similarity >= self.rejection_thresholds['moderate']:
            return 'accept', 'moderate'
        elif best_similarity >= self.rejection_thresholds['conservative']:
            return 'accept', 'conservative'
        else:
            return 'reject', 'all'
    
    def get_confidence_level(self, confidence):
        """Convert numerical confidence to human-readable level"""
        if confidence >= 0.85:
            return "Very High"
        elif confidence >= 0.7:
            return "High"
        elif confidence >= 0.55:
            return "Medium"
        elif confidence >= 0.35:
            return "Low"
        else:
            return "Very Low"
    
    def identify(self, image_path, rejection_level='moderate', return_top_k=3):
        """
        Main identification function with unknown detection
        
        Args:
            image_path: Path to query image
            rejection_level: 'conservative', 'moderate', or 'liberal'
            return_top_k: Number of top matches to return
        """
        
        # Extract query embedding
        query_embedding = self.extract_query_embedding(image_path)
        
        # Get top matches
        matches = self.advanced_matching(query_embedding, top_k=return_top_k)
        
        if not matches:
            return {
                'status': 'error',
                'individual': None,
                'confidence': 0,
                'confidence_level': 'Error',
                'all_matches': []
            }
        
        best_match = matches[0]
        best_similarity = best_match['similarity']
        
        # Check rejection thresholds
        rejection_decision, threshold_level = self.determine_rejection_level(best_similarity)
        
        if rejection_decision == 'accept':
            result = {
                'status': 'identified',
                'individual': best_match['individual'],
                'confidence': best_match['confidence'],
                'confidence_level': self.get_confidence_level(best_match['confidence']),
                'similarity': best_match['similarity'],
                'distance': best_match['distance'],
                'threshold_used': rejection_level,
                'threshold_passed': threshold_level,
                'rejection_info': {
                    'similarity_vs_threshold': best_similarity - self.rejection_thresholds[rejection_level],
                    'threshold_value': self.rejection_thresholds[rejection_level]
                },
                'all_matches': matches
            }
        else:
            result = {
                'status': 'unknown',
                'individual': None,
                'confidence': best_match['confidence'],
                'confidence_level': 'Unknown Individual',
                'best_match_similarity': best_similarity,
                'threshold_used': rejection_level,
                'threshold_value': self.rejection_thresholds[rejection_level],
                'rejection_info': {
                    'similarity_vs_threshold': best_similarity - self.rejection_thresholds[rejection_level],
                    'reason': f'Best similarity ({best_similarity:.3f}) below {rejection_level} threshold ({self.rejection_thresholds[rejection_level]:.3f})'
                },
                'all_matches': matches
            }
        
        return result

def batch_inference(identifier, image_dir, output_file=None, rejection_level='moderate'):
    """Run inference on all images in a directory"""
    
    results = []
    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Processing {len(image_files)} images with {rejection_level} rejection threshold...")
    
    stats = {'identified': 0, 'unknown': 0, 'errors': 0}
    
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(image_dir, img_file)
        
        try:
            result = identifier.identify(img_path, rejection_level=rejection_level)
            result['image_file'] = img_file
            results.append(result)
            
            if result['status'] == 'identified':
                print(f"{i+1:3d}. {img_file}: {result['individual']} "
                      f"({result['confidence_level']}, sim: {result['similarity']:.3f})")
                stats['identified'] += 1
            elif result['status'] == 'unknown':
                print(f"{i+1:3d}. {img_file}: Unknown individual "
                      f"(best sim: {result['best_match_similarity']:.3f}, "
                      f"threshold: {result['threshold_value']:.3f})")
                stats['unknown'] += 1
            else:
                print(f"{i+1:3d}. {img_file}: Error - {result.get('error', 'Unknown error')}")
                stats['errors'] += 1
                
        except Exception as e:
            print(f"{i+1:3d}. {img_file}: Error - {e}")
            results.append({
                'image_file': img_file,
                'status': 'error',
                'error': str(e)
            })
            stats['errors'] += 1
    
    # Save results if requested
    if output_file:
        import json
        # Add summary stats to results
        summary = {
            'total_images': len(image_files),
            'identified': stats['identified'],
            'unknown': stats['unknown'], 
            'errors': stats['errors'],
            'rejection_level': rejection_level,
            'identification_rate': stats['identified'] / len(image_files) if image_files else 0,
            'results': results
        }
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Results saved to {output_file}")
    
    return results, stats

def test_rejection_thresholds(identifier, test_images, known_labels=None):
    """Test different rejection thresholds on a set of images"""
    
    levels = ['conservative', 'moderate', 'liberal']
    results = {}
    
    print("Testing rejection thresholds...")
    
    for level in levels:
        print(f"\nTesting {level} threshold ({identifier.rejection_thresholds[level]:.3f}):")
        
        level_stats = {'identified': 0, 'unknown': 0, 'correct': 0, 'incorrect': 0}
        level_results = []
        
        for img_path in test_images:
            try:
                result = identifier.identify(img_path, rejection_level=level)
                level_results.append(result)
                
                if result['status'] == 'identified':
                    level_stats['identified'] += 1
                    
                    # Check correctness if labels provided
                    if known_labels and img_path in known_labels:
                        true_label = known_labels[img_path]
                        if result['individual'] == true_label:
                            level_stats['correct'] += 1
                        else:
                            level_stats['incorrect'] += 1
                            
                elif result['status'] == 'unknown':
                    level_stats['unknown'] += 1
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        # Calculate metrics
        total = len(test_images)
        identification_rate = level_stats['identified'] / total
        rejection_rate = level_stats['unknown'] / total
        
        if known_labels:
            accuracy = level_stats['correct'] / level_stats['identified'] if level_stats['identified'] > 0 else 0
            print(f"  Identification rate: {identification_rate:.3f} ({level_stats['identified']}/{total})")
            print(f"  Rejection rate: {rejection_rate:.3f} ({level_stats['unknown']}/{total})")
            print(f"  Accuracy (of identified): {accuracy:.3f} ({level_stats['correct']}/{level_stats['identified']})")
        else:
            print(f"  Identification rate: {identification_rate:.3f}")
            print(f"  Rejection rate: {rejection_rate:.3f}")
        
        results[level] = {
            'stats': level_stats,
            'results': level_results,
            'identification_rate': identification_rate,
            'rejection_rate': rejection_rate
        }
    
    return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Cheetah Identification with Unknown Detection")
    parser.add_argument('--image_path', type=str, help='Single image to identify')
    parser.add_argument('--image_dir', type=str, help='Directory of images to process')
    parser.add_argument('--model_path', type=str, default='models/cheetah_cropped_embedder.pt')
    parser.add_argument('--reference_embeddings', type=str, default='models/reference_embeddings_cropped.pkl')
    parser.add_argument('--rejection_level', type=str, default='moderate', 
                       choices=['conservative', 'moderate', 'liberal'])
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--output_file', type=str, help='Save batch results to JSON file')
    parser.add_argument('--no_cropping', action='store_true', help='Disable auto-cropping')
    parser.add_argument('--test_thresholds', action='store_true', help='Test all rejection thresholds')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    use_cropping = not args.no_cropping
    
    # Initialize identifier
    try:
        identifier = EnhancedCheetahIdentifier(
            args.model_path, 
            args.reference_embeddings, 
            device, 
            use_cropping=use_cropping
        )
    except Exception as e:
        print(f"Error initializing identifier: {e}")
        return
    
    if args.image_path:
        # Single image inference
        try:
            result = identifier.identify(args.image_path, args.rejection_level, args.top_k)
            
            print(f"\nResults for {args.image_path}:")
            print(f"Status: {result['status'].upper()}")
            
            if result['status'] == 'identified':
                print(f"Individual: {result['individual']}")
                print(f"Confidence: {result['confidence']:.3f} ({result['confidence_level']})")
                print(f"Similarity: {result['similarity']:.3f}")
                print(f"Threshold used: {result['threshold_used']} ({result['rejection_info']['threshold_value']:.3f})")
                print(f"Similarity margin: +{result['rejection_info']['similarity_vs_threshold']:.3f}")
            else:
                print("Individual: Unknown")
                print(f"Best match similarity: {result['best_match_similarity']:.3f}")
                print(f"Threshold: {result['threshold_value']:.3f}")
                print(f"Reason: {result['rejection_info']['reason']}")
            
            print(f"\nTop {len(result['all_matches'])} matches:")
            for i, match in enumerate(result['all_matches']):
                print(f"  {i+1}. {match['individual']}: sim={match['similarity']:.3f}, "
                      f"conf={match['confidence']:.3f}")
                      
        except Exception as e:
            print(f"Error processing image: {e}")
    
    elif args.image_dir:
        # Batch inference
        if not os.path.exists(args.image_dir):
            print(f"Directory {args.image_dir} does not exist")
            return
        
        if args.test_thresholds:
            # Test all threshold levels
            image_files = [os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            threshold_results = test_rejection_thresholds(identifier, image_files)
        else:
            # Regular batch processing
            results, stats = batch_inference(identifier, args.image_dir, args.output_file, args.rejection_level)
            
            print(f"\nSummary (using {args.rejection_level} threshold):")
            print(f"  Identified: {stats['identified']}")
            print(f"  Unknown: {stats['unknown']}")
            print(f"  Errors: {stats['errors']}")
            print(f"  Total: {sum(stats.values())}")
            print(f"  Identification rate: {stats['identified']/sum(stats.values()):.3f}")
    
    else:
        print("Please provide either --image_path or --image_dir")

if __name__ == "__main__":
    main()