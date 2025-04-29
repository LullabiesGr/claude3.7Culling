
"""
Photo Culling System - Core Analysis Pipeline
"""
import os
import cv2
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ImageAnalysisResult:
    """Represents the result of image analysis and culling"""
    filename: str
    approved: bool
    score: float
    metrics: Dict[str, float]
    
    def __str__(self) -> str:
        return f"Image: {self.filename}, Score: {self.score}/100, Approved: {self.approved}"

class ImageAnalyzer:
    """Core class for analyzing image quality"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize the image analyzer with configuration
        
        Parameters:
        -----------
        config : Dict
            Configuration parameters for analysis thresholds
        """
        self.config = config or {
            'min_resolution': (800, 600),  # Minimum acceptable resolution
            'min_brightness': 40,          # Minimum brightness (0-255)
            'max_brightness': 220,         # Maximum brightness (0-255)
            'min_contrast': 30,            # Minimum contrast
            'blur_threshold': 100,         # Laplacian variance threshold for blur detection
            'approval_threshold': 70       # Minimum score for approval
        }
        logger.info("Image analyzer initialized with configuration")
    
    def analyze_image(self, image_path: str) -> ImageAnalysisResult:
        """
        Analyze an image and return quality metrics with approval decision
        
        Parameters:
        -----------
        image_path : str
            Path to the image file
            
        Returns:
        --------
        ImageAnalysisResult
            Analysis results including approval status and score
        """
        try:
            # Load the image
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Failed to load image: {image_path}")
                return ImageAnalysisResult(
                    filename=os.path.basename(image_path),
                    approved=False,
                    score=0.0,
                    metrics={'error': 'Failed to load image'}
                )
            
            # Calculate metrics
            metrics = {}
            
            # Resolution check
            height, width = img.shape[:2]
            metrics['resolution'] = (width, height)
            resolution_score = min(100, (width * height) / (1920 * 1080) * 100)
            
            # Convert to grayscale for some analyses
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Brightness analysis
            brightness = np.mean(gray)
            metrics['brightness'] = brightness
            brightness_score = 100 - min(100, abs(brightness - 128) / 1.28)
            
            # Contrast analysis
            contrast = np.std(gray)
            metrics['contrast'] = contrast
            contrast_score = min(100, contrast / 80 * 100)
            
            # Blur detection using Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            blur_variance = np.var(laplacian)
            metrics['blur_variance'] = blur_variance
            blur_score = min(100, blur_variance / self.config['blur_threshold'] * 100)
            
            # Color balance
            b, g, r = cv2.split(img)
            b_mean, g_mean, r_mean = np.mean(b), np.mean(g), np.mean(r)
            max_diff = max(abs(r_mean - g_mean), abs(r_mean - b_mean), abs(g_mean - b_mean))
            color_balance = 1 - (max_diff / 255)
            metrics['color_balance'] = color_balance
            color_score = color_balance * 100
            
            # Calculate final score (weighted average)
            weights = {
                'resolution': 0.15,
                'brightness': 0.2,
                'contrast': 0.25,
                'blur': 0.3,
                'color': 0.1
            }
            
            final_score = (
                weights['resolution'] * resolution_score +
                weights['brightness'] * brightness_score +
                weights['contrast'] * contrast_score +
                weights['blur'] * blur_score +
                weights['color'] * color_score
            )
            
            # Round to 2 decimal places
            final_score = round(final_score, 2)
            
            # Determine if image is approved
            approved = final_score >= self.config['approval_threshold']
            
            # Store individual component scores for reference
            metrics.update({
                'resolution_score': resolution_score,
                'brightness_score': brightness_score,
                'contrast_score': contrast_score,
                'blur_score': blur_score,
                'color_score': color_score
            })
            
            return ImageAnalysisResult(
                filename=os.path.basename(image_path),
                approved=approved,
                score=final_score,
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {str(e)}")
            return ImageAnalysisResult(
                filename=os.path.basename(image_path),
                approved=False,
                score=0.0,
                metrics={'error': str(e)}
            )

class BatchProcessor:
    """Process multiple images in batch mode"""
    
    def __init__(self, analyzer: ImageAnalyzer = None):
        """
        Initialize the batch processor
        
        Parameters:
        -----------
        analyzer : ImageAnalyzer
            The analyzer to use for image processing
        """
        self.analyzer = analyzer or ImageAnalyzer()
        logger.info("Batch processor initialized")
    
    def process_directory(self, directory_path: str) -> List[ImageAnalysisResult]:
        """
        Process all images in a directory
        
        Parameters:
        -----------
        directory_path : str
            Path to directory containing images
            
        Returns:
        --------
        List[ImageAnalysisResult]
            Analysis results for all processed images
        """
        results = []
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        try:
            for filename in os.listdir(directory_path):
                file_path = os.path.join(directory_path, filename)
                if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in valid_extensions):
                    logger.info(f"Processing {filename}")
                    result = self.analyzer.analyze_image(file_path)
                    results.append(result)
            
            # Sort results by score (highest first)
            results.sort(key=lambda x: x.score, reverse=True)
            return results
            
        except Exception as e:
            logger.error(f"Error processing directory {directory_path}: {str(e)}")
            return results

def demo():
    """Demo function to test the image analysis pipeline"""
    # Create analyzer with default configuration
    analyzer = ImageAnalyzer()
    
    # Create batch processor
    processor = BatchProcessor(analyzer)
    
    # Sample directory - replace with actual path
    sample_dir = "./data/sample_images"
    
    # Check if directory exists
    if not os.path.exists(sample_dir):
        print(f"Sample directory {sample_dir} not found.")
        return
    
    # Process images
    results = processor.process_directory(sample_dir)
    
    # Display results
    print(f"Processed {len(results)} images")
    print("\nTop 5 images:")
    for i, result in enumerate(results[:5], 1):
        print(f"{i}. {result}")
        for key, value in result.metrics.items():
            if key.endswith('_score'):
                print(f"   - {key}: {value:.2f}")
    
    print("\nBottom 5 images:")
    for i, result in enumerate(results[-5:], len(results)-4):
        print(f"{i}. {result}")
        for key, value in result.metrics.items():
            if key.endswith('_score'):
                print(f"   - {key}: {value:.2f}")
    
    # Summary statistics
    approved_count = sum(1 for r in results if r.approved)
    avg_score = sum(r.score for r in results) / len(results) if results else 0
    
    print(f"\nSummary: {approved_count}/{len(results)} images approved. Average score: {avg_score:.2f}")

if __name__ == "__main__":
    demo()
