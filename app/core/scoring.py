"""
Advanced Image Scoring Module

This module provides more sophisticated image quality assessment algorithms.
It can be integrated with the core image analysis pipeline to provide better culling results.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, List, Optional
import logging
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ScoreWeights:
    """Configuration for score weights in quality assessment"""
    sharpness: float = 0.25
    exposure: float = 0.20
    contrast: float = 0.15
    color_balance: float = 0.15
    noise: float = 0.10
    composition: float = 0.15

class AdvancedImageScorer:
    """Advanced image quality scoring algorithms"""
    
    def __init__(self, weights: Optional[ScoreWeights] = None):
        """
        Initialize the advanced scorer with custom weights
        
        Parameters:
        -----------
        weights : ScoreWeights, optional
            Custom weights for different quality aspects
        """
        self.weights = weights or ScoreWeights()
        logger.info("Advanced image scorer initialized")
    
    def analyze_sharpness(self, img: np.ndarray) -> Tuple[float, Dict]:
        """
        Analyze image sharpness using multiple methods
        
        Parameters:
        -----------
        img : np.ndarray
            OpenCV image in BGR format
            
        Returns:
        --------
        tuple
            (score, metrics dictionary)
        """
        try:
            # Convert to grayscale
            if len(img.shape) > 2:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            # Method 1: Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            lap_var = np.var(laplacian)
            
            # Method 2: Sobel edge detection
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_var = np.var(sobelx) + np.var(sobely)
            
            # Method 3: High-pass filter
            kernel = np.array([[-1, -1, -1], 
                             [-1,  9, -1],
                             [-1, -1, -1]])
            highpass = cv2.filter2D(gray, -1, kernel)
            hp_mean = np.mean(highpass)
            
            # Calculate normalized score (0-100)
            # These thresholds can be fine-tuned based on your specific needs
            lap_score = min(100, lap_var / 500 * 100)
            sobel_score = min(100, sobel_var / 1000 * 100)
            hp_score = min(100, hp_mean / 50 * 100)
            
            # Combine methods with weights
            combined_score = lap_score * 0.5 + sobel_score * 0.3 + hp_score * 0.2
            
            # Create metrics dictionary
            metrics = {
                'laplacian_variance': lap_var,
                'sobel_variance': sobel_var,
                'highpass_mean': hp_mean,
                'laplacian_score': lap_score,
                'sobel_score': sobel_score,
                'highpass_score': hp_score
            }
            
            return combined_score, metrics
            
        except Exception as e:
            logger.error(f"Error in sharpness analysis: {str(e)}")
            return 0.0, {'error': str(e)}
    
    def analyze_exposure(self, img: np.ndarray) -> Tuple[float, Dict]:
        """
        Analyze image exposure (brightness levels)
        
        Parameters:
        -----------
        img : np.ndarray
            OpenCV image in BGR format
            
        Returns:
        --------
        tuple
            (score, metrics dictionary)
        """
        try:
            # Convert to grayscale
            if len(img.shape) > 2:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            # Calculate histogram
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten() / hist.sum()  # Normalize
            
            # Calculate mean and std of brightness
            mean_brightness = np.mean(gray)
            std_brightness = np.std(gray)
            
            # Calculate metrics
            # 1. Exposure deviation from optimal (around 128)
            exposure_deviation = abs(mean_brightness - 128) / 128
            
            # 2. Check for over/underexposure
            dark_pixels_ratio = np.sum(gray < 30) / gray.size
            bright_pixels_ratio = np.sum(gray > 220) / gray.size
            
            # 3. Check exposure distribution
            # Calculate entropy as a measure of information
            non_zero_hist = hist[hist > 0]
            hist_entropy = -np.sum(non_zero_hist * np.log2(non_zero_hist))
            
            # Calculate scores
            deviation_score = 100 * (1 - exposure_deviation)
            
            # Penalize for over/underexposure
            exposure_penalty = (dark_pixels_ratio + bright_pixels_ratio) * 100
            clipping_score = 100 - min(100, exposure_penalty * 200)
            
            # Entropy score (normalized to 0-100)
            max_entropy = np.log2(256)  # Maximum possible entropy
            entropy_score = (hist_entropy / max_entropy) * 100
            
            # Combined exposure score
            exposure_score = (
                deviation_score * 0.4 +
                clipping_score * 0.4 +
                entropy_score * 0.2
            )
            
            # Create metrics dictionary
            metrics = {
                'mean_brightness': mean_brightness,
                'std_brightness': std_brightness,
                'dark_ratio': dark_pixels_ratio,
                'bright_ratio': bright_pixels_ratio,
                'histogram_entropy': hist_entropy,
                'deviation_score': deviation_score,
                'clipping_score': clipping_score,
                'entropy_score': entropy_score
            }
            
            return exposure_score, metrics
            
        except Exception as e:
            logger.error(f"Error in exposure analysis: {str(e)}")
            return 0.0, {'error': str(e)}
    
    def analyze_contrast(self, img: np.ndarray) -> Tuple[float, Dict]:
        """
        Analyze image contrast
        
        Parameters:
        -----------
        img : np.ndarray
            OpenCV image in BGR format
            
        Returns:
        --------
        tuple
            (score, metrics dictionary)
        """
        try:
            # Convert to grayscale
            if len(img.shape) > 2:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            # Calculate standard deviation as a measure of contrast
            std_dev = np.std(gray)
            
            # Calculate histogram
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten() / hist.sum()  # Normalize
            
            # Calculate metrics
            # 1. Standard deviation of pixel values
            
            # 2. Dynamic range
            p1, p99 = np.percentile(gray, [1, 99])
            dynamic_range = p99 - p1
            
            # 3. RMS contrast
            rms_contrast = np.sqrt(np.mean(np.square(gray - np.mean(gray)))) / np.mean(gray)
            
            # Calculate scores
            std_score = min(100, std_dev / 80 * 100)  # Normalize to 0-100
            range_score = min(100, dynamic_range / 200 * 100)
            rms_score = min(100, rms_contrast * 250)
            
            # Combined contrast score
            contrast_score = std_score * 0.4 + range_score * 0.4 + rms_score * 0.2
            
            # Create metrics dictionary
            metrics = {
                'std_dev': std_dev,
                'dynamic_range': dynamic_range,
                'rms_contrast': rms_contrast,
                'std_score': std_score,
                'range_score': range_score,
                'rms_score': rms_score
            }
            
            return contrast_score, metrics
            
        except Exception as e:
            logger.error(f"Error in contrast analysis: {str(e)}")
            return 0.0, {'error': str(e)}
    
    def analyze_color_balance(self, img: np.ndarray) -> Tuple[float, Dict]:
        """
        Analyze color balance of the image
        
        Parameters:
        -----------
        img : np.ndarray
            OpenCV image in BGR format
            
        Returns:
        --------
        tuple
            (score, metrics dictionary)
        """
        try:
            # Ensure image is color
            if len(img.shape) < 3:
                # If grayscale, convert to color
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            # Split into channels
            b, g, r = cv2.split(img)
            
            # Calculate mean of each channel
            b_mean, g_mean, r_mean = np.mean(b), np.mean(g), np.mean(r)
            
            # Calculate metrics
            # 1. Channel imbalance
            max_mean = max(r_mean, g_mean, b_mean)
            min_mean = min(r_mean, g_mean, b_mean)
            channel_imbalance = (max_mean - min_mean) / max(1, max_mean)
            
            # 2. Color cast detection
            # Calculate differences between channels
            rg_diff = abs(r_mean - g_mean) / 255
            rb_diff = abs(r_mean - b_mean) / 255
            gb_diff = abs(g_mean - b_mean) / 255
            
            color_cast = max(rg_diff, rb_diff, gb_diff)
            
            # 3. Gray world assumption
            # In a balanced image, the average color should be gray
            overall_mean = (r_mean + g_mean + b_mean) / 3
            r_deviation = abs(r_mean - overall_mean) / 255
            g_deviation = abs(g_mean - overall_mean) / 255
            b_deviation = abs(b_mean - overall_mean) / 255
            
            gray_world_deviation = (r_deviation + g_deviation + b_deviation) / 3
            
            # Calculate scores
            imbalance_score = 100 * (1 - channel_imbalance)
            cast_score = 100 * (1 - color_cast)
            gray_score = 100 * (1 - gray_world_deviation)
            
            # Combined color balance score
            color_balance_score = imbalance_score * 0.4 + cast_score * 0.3 + gray_score * 0.3
            
            # Create metrics dictionary
            metrics = {
                'r_mean': r_mean,
                'g_mean': g_mean,
                'b_mean': b_mean,
                'channel_imbalance': channel_imbalance,
                'color_cast': color_cast,
                'gray_world_deviation': gray_world_deviation,
                'imbalance_score': imbalance_score,
                'cast_score': cast_score,
                'gray_score': gray_score
            }
            
            return color_balance_score, metrics
            
        except Exception as e:
            logger.error(f"Error in color balance analysis: {str(e)}")
            return 0.0, {'error': str(e)}
    
    def analyze_noise(self, img: np.ndarray) -> Tuple[float, Dict]:
        """
        Analyze image noise levels
        
        Parameters:
        -----------
        img : np.ndarray
            OpenCV image in BGR format
            
        Returns:
        --------
        tuple
            (score, metrics dictionary)
        """
        try:
            # Convert to grayscale
            if len(img.shape) > 2:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Calculate noise as the difference between original and blurred
            noise = cv2.absdiff(gray, blurred)
            
            # Calculate noise metrics
            noise_mean = np.mean(noise)
            noise_std = np.std(noise)
            
            # Calculate noise to signal ratio
            signal_mean = np.mean(gray)
            noise_ratio = noise_mean / max(1, signal_mean)
            
            # Calculate scores
            mean_score = 100 * (1 - min(1, noise_mean / 20))
            std_score = 100 * (1 - min(1, noise_std / 20))
            ratio_score = 100 * (1 - min(1, noise_ratio * 10))
            
            # Combined noise score (higher is better, meaning less noise)
            noise_score = mean_score * 0.3 + std_score * 0.3 + ratio_score * 0.4
            
            # Create metrics dictionary
            metrics = {
                'noise_mean': noise_mean,
                'noise_std': noise_std,
                'noise_ratio': noise_ratio,
                'mean_score': mean_score,
                'std_score': std_score,
                'ratio_score': ratio_score
            }
            
            return noise_score, metrics
            
        except Exception as e:
            logger.error(f"Error in noise analysis: {str(e)}")
            return 0.0, {'error': str(e)}
    
    def analyze_composition(self, img: np.ndarray) -> Tuple[float, Dict]:
        """
        Analyze image composition using rule of thirds and other metrics
        
        Parameters:
        -----------
        img : np.ndarray
            OpenCV image in BGR format
            
        Returns:
        --------
        tuple
            (score, metrics dictionary)
        """
        try:
            # Get image dimensions
            height, width = img.shape[:2]
            
            # Convert to grayscale
            if len(img.shape) > 2:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            # Edge detection
            edges = cv2.Canny(gray, 100, 200)
            
            # Rule of thirds grid points
            third_h, third_w = height // 3, width // 3
            roi_points = [
                (third_w, third_h),
                (third_w * 2, third_h),
                (third_w, third_h * 2),
                (third_w * 2, third_h * 2)
            ]
            
            # Calculate interest in rule of thirds points
            roi_interest = 0
            roi_radius = min(height, width) // 15  # Define region around points
            
            for point in roi_points:
                x, y = point
                roi = edges[max(0, y-roi_radius):min(height, y+roi_radius),
                           max(0, x-roi_radius):min(width, x+roi_radius)]
                roi_interest += np.count_nonzero(roi) / roi.size
            
            roi_interest /= len(roi_points)  # Normalize
            
            # Center weight - generally subjects in center may indicate worse composition
            # Create a gaussian mask with center weighted
            center_y, center_x = height // 2, width // 2
            Y, X = np.ogrid[:height, :width]
            center_dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            # Normalize distance
            max_dist = np.sqrt(center_x**2 + center_y**2)
            center_dist = center_dist / max_dist
            
            # Multiply edges by distance from center (further is better for rule of thirds)
            center_weight = np.sum(edges * center_dist) / np.sum(edges) if np.sum(edges) > 0 else 0
            
            # Calculate edge distribution (entropy)
            edge_hist = cv2.calcHist([edges], [0], None, [256], [0, 256])
            edge_hist = edge_hist / np.sum(edge_hist)
            non_zero_hist = edge_hist[edge_hist > 0]
            edge_entropy = -np.sum(non_zero_hist * np.log2(non_zero_hist)) if len(non_zero_hist) > 0 else 0
            
            # Calculate scores
            roi_score = roi_interest * 100  # Rule of thirds interest
            center_score = center_weight * 100  # Center weight score
            entropy_score = (edge_entropy / np.log2(256)) * 100  # Edge distribution score
            
            # Combined composition score
            composition_score = roi_score * 0.4 + center_score * 0.3 + entropy_score * 0.3
            
            # Create metrics dictionary
            metrics = {
                'roi_interest': roi_interest,
                'center_weight': center_weight,
                'edge_entropy': edge_entropy,
                'roi_score': roi_score,
                'center_score': center_score,
                'entropy_score': entropy_score
            }
            
            return composition_score, metrics
            
        except Exception as e:
            logger.error(f"Error in composition analysis: {str(e)}")
            return 0.0, {'error': str(e)}
    
    def score_image(self, img: np.ndarray) -> Tuple[float, Dict]:
        """
        Calculate comprehensive image quality score
        
        Parameters:
        -----------
        img : np.ndarray
            OpenCV image in BGR format
            
        Returns:
        --------
        tuple
            (final score, detailed metrics dictionary)
        """
        metrics = {}
        
        # Analyze individual aspects
        sharpness_score, sharpness_metrics = self.analyze_sharpness(img)
        metrics['sharpness'] = sharpness_metrics
        
        exposure_score, exposure_metrics = self.analyze_exposure(img)
        metrics['exposure'] = exposure_metrics
        
        contrast_score, contrast_metrics = self.analyze_contrast(img)
        metrics['contrast'] = contrast_metrics
        
        color_score, color_metrics = self.analyze_color_balance(img)
        metrics['color_balance'] = color_metrics
        
        noise_score, noise_metrics = self.analyze_noise(img)
        metrics['noise'] = noise_metrics
        
        composition_score, composition_metrics = self.analyze_composition(img)
        metrics['composition'] = composition_metrics
        
        # Store component scores
        component_scores = {
            'sharpness_score': sharpness_score,
            'exposure_score': exposure_score,
            'contrast_score': contrast_score,
            'color_balance_score': color_score,
            'noise_score': noise_score,
            'composition_score': composition_score
        }
        metrics['component_scores'] = component_scores
        
        # Calculate final weighted score
        final_score = (
            self.weights.sharpness * sharpness_score +
            self.weights.exposure * exposure_score +
            self.weights.contrast * contrast_score +
            self.weights.color_balance * color_score +
            self.weights.noise * noise_score +
            self.weights.composition * composition_score
        )
        
        # Round to 2 decimal places
        final_score = round(final_score, 2)
        
        return final_score, metrics


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python scoring.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image {image_path}")
            sys.exit(1)
        
        # Create scorer
        scorer = AdvancedImageScorer()
        
        # Score image
        score, metrics = scorer.score_image(img)
        
        # Print results
        print(f"Image Quality Score: {score}/100")
        print("\nComponent Scores:")
        for name, score in metrics['component_scores'].items():
            print(f"{name}: {score:.2f}/100")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
