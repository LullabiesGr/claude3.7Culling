"""
Photo Culling System - Web Interface

A Flask web application that allows users to upload photos for automated culling.
"""
import os
import uuid
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import logging
from datetime import datetime

# Import our image analysis components
from app.core.analyzer import ImageAnalyzer, BatchProcessor, ImageAnalysisResult

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit uploads to 16MB
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the image analyzer
analyzer = ImageAnalyzer()
batch_processor = BatchProcessor(analyzer)

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """API endpoint for analyzing a single image"""
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    # If user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Create a unique filename to avoid overwriting
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save the file
        file.save(file_path)
        
        # Analyze the image
        result = analyzer.analyze_image(file_path)
        
        # Convert result to dict for JSON response
        response = {
            'filename': result.filename,
            'original_filename': filename,
            'approved': result.approved,
            'score': result.score,
            'metrics': result.metrics,
            'file_url': f"/uploads/{unique_filename}"
        }
        
        return jsonify(response)
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/api/batch', methods=['POST'])
def analyze_batch():
    """API endpoint for analyzing multiple images"""
    # Check if the post request has files
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files part'}), 400
    
    files = request.files.getlist('files[]')
    
    # Check if files were selected
    if not files or files[0].filename == '':
        return jsonify({'error': 'No selected files'}), 400
    
    results = []
    batch_id = str(uuid.uuid4())
    batch_dir = os.path.join(app.config['UPLOAD_FOLDER'], batch_id)
    os.makedirs(batch_dir, exist_ok=True)
    
    for file in files:
        if file and allowed_file(file.filename):
            # Create a unique filename
            filename = secure_filename(file.filename)
            file_path = os.path.join(batch_dir, filename)
            
            # Save the file
            file.save(file_path)
            
            # Add to the list of files to process
            results.append(file_path)
    
    # Check if we have valid files to process
    if not results:
        return jsonify({'error': 'No valid files uploaded'}), 400
    
    # Process all images in the batch
    analysis_results = []
    for file_path in results:
        result = analyzer.analyze_image(file_path)
        
        # Convert result to dict
        analysis_results.append({
            'filename': result.filename,
            'approved': result.approved,
            'score': result.score,
            'metrics': result.metrics,
            'file_url': f"/uploads/{batch_id}/{os.path.basename(file_path)}"
        })
    
    # Sort results by score
    analysis_results.sort(key=lambda x: x['score'], reverse=True)
    
    # Create summary statistics
    approved_count = sum(1 for r in analysis_results if r['approved'])
    avg_score = sum(r['score'] for r in analysis_results) / len(analysis_results) if analysis_results else 0
    
    response = {
        'batch_id': batch_id,
        'timestamp': datetime.now().isoformat(),
        'total_images': len(analysis_results),
        'approved_images': approved_count,
        'average_score': round(avg_score, 2),
        'results': analysis_results
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
