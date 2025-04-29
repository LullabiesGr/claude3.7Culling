"""
Enhanced Flask application for the photo culling service.
This file serves as the entry point for the web interface,
integrating OpenCV, Pillow, and Hugging Face models for intelligent photo analysis.
"""
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import os
import uuid
import logging
from werkzeug.utils import secure_filename
import sys
import threading
from datetime import datetime

# Add the parent directory to sys.path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core components
from app.core.analyzer import analyze_image, analyze_batch
from app.core.scoring import score_image
from app.core.decision import make_decision
from app.utils.image_processing import process_image, extract_metadata
from app.utils.validators import validate_image
from app.core.models import load_models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs', 'app.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev_key_for_testing')
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'uploads')
app.config['PROCESSED_FOLDER'] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'processed')
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'heic', 'nef', 'cr2', 'arw'}
app.config['BATCH_SIZE'] = 4  # Number of images to process in parallel

# Create necessary folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs'), exist_ok=True)

# Store active processing jobs
active_jobs = {}
models = None

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.before_first_request
def load_ml_models():
    """Load ML models before handling requests"""
    global models
    try:
        logger.info("Loading ML models...")
        models = load_models()
        logger.info("ML models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading ML models: {str(e)}")
        models = None

def process_batch(job_id, filepaths, session_data):
    """Process a batch of images in a background thread"""
    global active_jobs, models
    
    try:
        logger.info(f"Starting batch processing for job {job_id} with {len(filepaths)} images")
        active_jobs[job_id]['status'] = 'processing'
        active_jobs[job_id]['processed'] = 0
        active_jobs[job_id]['total'] = len(filepaths)
        
        results = []
        
        # Process images in batches for better performance
        batch_size = app.config['BATCH_SIZE']
        for i in range(0, len(filepaths), batch_size):
            batch = filepaths[i:i+batch_size]
            processed_images = []
            
            # First, process all images in the batch
            for filepath in batch:
                filename = os.path.basename(filepath)
                try:
                    # Validate the image
                    if not validate_image(filepath):
                        results.append({
                            'filename': filename,
                            'status': 'error',
                            'message': 'Invalid image file'
                        })
                        continue
                    
                    # Process the image and extract metadata
                    processed_image = process_image(filepath)
                    metadata = extract_metadata(filepath)
                    processed_images.append((filename, filepath, processed_image, metadata))
                    
                except Exception as e:
                    logger.error(f"Error processing {filename}: {str(e)}")
                    results.append({
                        'filename': filename,
                        'status': 'error',
                        'message': str(e)
                    })
            
            # Then, analyze the batch with ML models
            if processed_images:
                batch_analyses = analyze_batch([img for _, _, img, _ in processed_images], models)
                
                # Process the results
                for idx, (filename, filepath, _, metadata) in enumerate(processed_images):
                    if idx < len(batch_analyses):
                        analysis = batch_analyses[idx]
                        score = score_image(analysis, metadata)
                        decision = make_decision(score, session_data.get('threshold', 7.0))
                        
                        results.append({
                            'filename': filename,
                            'status': 'success',
                            'score': score,
                            'decision': decision,
                            'keep': decision['recommendation'] == 'keep',
                            'metadata': metadata
                        })
                    
                    active_jobs[job_id]['processed'] += 1
                    active_jobs[job_id]['progress'] = (active_jobs[job_id]['processed'] / active_jobs[job_id]['total']) * 100
        
        active_jobs[job_id]['status'] = 'completed'
        active_jobs[job_id]['results'] = results
        active_jobs[job_id]['completed_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"Completed batch processing for job {job_id}")
        
    except Exception as e:
        logger.error(f"Batch processing error for job {job_id}: {str(e)}")
        active_jobs[job_id]['status'] = 'error'
        active_jobs[job_id]['error'] = str(e)

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle image upload and start processing"""
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    files = request.files.getlist('files[]')
    
    if not files or files[0].filename == '':
        return jsonify({'error': 'No selected files'}), 400
    
    # Generate a unique job ID
    job_id = str(uuid.uuid4())
    
    # Create a directory for this job
    job_dir = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    # Save uploaded files
    filepaths = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(job_dir, filename)
            file.save(filepath)
            filepaths.append(filepath)
    
    if not filepaths:
        return jsonify({'error': 'No valid files were uploaded'}), 400
    
    # Get session data for customized processing
    session_data = {
        'threshold': float(request.form.get('threshold', 7.0)),
        'prefer_faces': request.form.get('prefer_faces', 'true') == 'true',
        'prefer_sharpness': request.form.get('prefer_sharpness', 'true') == 'true',
        'prefer_exposure': request.form.get('prefer_exposure', 'true') == 'true'
    }
    
    # Initialize the job status
    active_jobs[job_id] = {
        'id': job_id,
        'status': 'starting',
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'progress': 0,
        'total': len(filepaths),
        'processed': 0,
        'session_data': session_data
    }
    
    # Start background processing
    thread = threading.Thread(target=process_batch, args=(job_id, filepaths, session_data))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'job_id': job_id,
        'status': 'started',
        'message': f'Processing {len(filepaths)} images',
        'redirect': url_for('job_status', job_id=job_id)
    })

@app.route('/job/<job_id>')
def job_status(job_id):
    """Render the job status page"""
    if job_id not in active_jobs:
        return render_template('error.html', message='Job not found'), 404
    
    return render_template('job_status.html', job_id=job_id)

@app.route('/api/job/<job_id>')
def api_job_status(job_id):
    """API endpoint to get job status"""
    if job_id not in active_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = active_jobs[job_id]
    
    # If job is completed, include the results
    if job['status'] == 'completed':
        return jsonify({
            'id': job_id,
            'status': job['status'],
            'progress': 100,
            'total': job['total'],
            'processed': job['processed'],
            'results': job['results'],
            'completed_at': job['completed_at']
        })
    
    # Otherwise just send the status info
    return jsonify({
        'id': job_id,
        'status': job['status'],
        'progress': job['progress'],
        'total': job['total'],
        'processed': job['processed']
    })

@app.route('/results/<job_id>')
def results(job_id):
    """Display processing results for a job"""
    if job_id not in active_jobs or active_jobs[job_id]['status'] != 'completed':
        return redirect(url_for('job_status', job_id=job_id))
    
    return render_template('results.html', job_id=job_id, results=active_jobs[job_id]['results'])

@app.route('/api/settings', methods=['GET', 'POST'])
def api_settings():
    """API endpoint to get or update culling settings"""
    if request.method == 'POST':
        data = request.json
        session['threshold'] = float(data.get('threshold', 7.0))
        session['prefer_faces'] = data.get('prefer_faces', True)
        session['prefer_sharpness'] = data.get('prefer_sharpness', True)
        session['prefer_exposure'] = data.get('prefer_exposure', True)
        return jsonify({'status': 'success', 'message': 'Settings updated'})
    
    # GET - return current settings
    return jsonify({
        'threshold': session.get('threshold', 7.0),
        'prefer_faces': session.get('prefer_faces', True),
        'prefer_sharpness': session.get('prefer_sharpness', True),
        'prefer_exposure': session.get('prefer_exposure', True)
    })

@app.route('/api/status')
def api_status():
    """API endpoint to check service status"""
    return jsonify({
        'status': 'online',
        'version': '1.1.0',
        'models_loaded': models is not None,
        'message': 'Photo culling service is running'
    })

@app.route('/batch_upload')
def batch_upload():
    """Render the batch upload page"""
    return render_template('batch_upload.html')

@app.route('/settings')
def settings():
    """Render the settings page"""
    return render_template('settings.html')

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({'error': 'File too large'}), 413

@app.errorhandler(500)
def server_error(error):
    """Handle server errors"""
    logger.error(f"Server error: {str(error)}")
    return jsonify({'error': 'Server error', 'message': str(error)}), 500

if __name__ == '__main__':
    # For local development
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'development') == 'development'
    app.run(debug=debug, host='0.0.0.0', port=port)
