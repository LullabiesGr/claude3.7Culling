"""
Main Flask application for the photo culling service.
This file serves as the entry point for the web interface.
"""
from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
from werkzeug.utils import secure_filename
import sys

# Add the parent directory to sys.path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core components
from app.core.analyzer import analyze_image
from app.core.scoring import score_image
from app.core.decision import make_decision
from app.utils.image_processing import process_image
from app.utils.validators import validate_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle image upload and processing"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    files = request.files.getlist('file')
    
    if not files or files[0].filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    results = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the image
            try:
                # Validate the image
                if not validate_image(filepath):
                    results.append({
                        'filename': filename,
                        'status': 'error',
                        'message': 'Invalid image file'
                    })
                    continue
                
                # Process the image
                processed_image = process_image(filepath)
                
                # Analyze the image
                analysis_result = analyze_image(processed_image)
                
                # Score the image
                score = score_image(analysis_result)
                
                # Make a decision
                decision = make_decision(score)
                
                results.append({
                    'filename': filename,
                    'status': 'success',
                    'score': score,
                    'decision': decision,
                    'keep': decision['recommendation'] == 'keep'
                })
            except Exception as e:
                results.append({
                    'filename': filename,
                    'status': 'error',
                    'message': str(e)
                })
        else:
            results.append({
                'filename': file.filename if file else 'unknown',
                'status': 'error',
                'message': 'File type not allowed'
            })
    
    return jsonify(results)

@app.route('/results')
def results():
    """Display processing results"""
    return render_template('results.html')

@app.route('/api/status')
def api_status():
    """API endpoint to check service status"""
    return jsonify({
        'status': 'online',
        'version': '1.0.0',
        'message': 'Photo culling service is running'
    })

if __name__ == '__main__':
    # For local development
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
