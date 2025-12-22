"""
PanoStitch Backend API
REST API for panoramic image stitching
"""

from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import cv2
import tempfile
from pathlib import Path
from datetime import datetime
import traceback
import io
import sys
from pathlib import Path

# Add parent directories to path so we can import src
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from src.stitcher import PanoStitch

# Initialize Flask app with static folder pointing to test directory
static_folder = os.path.join(project_root, 'backend', 'test')
app = Flask(__name__, static_folder=static_folder, static_url_path='')
CORS(app)

# Configuration
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'panostitch_uploads')
RESULTS_FOLDER = os.path.join(tempfile.gettempdir(), 'panostitch_results')
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def root():
    """Redirect to API tester."""
    tester_path = os.path.join(project_root, 'backend', 'test', 'api_tester.html')
    if os.path.exists(tester_path):
        with open(tester_path, 'r') as f:
            return f.read()
    return jsonify({
        'message': 'PanoStitch Backend API',
        'endpoints': {
            'health': '/api/health',
            'stitch': '/api/stitch',
            'info': '/api/info',
            'tester': '/api_tester.html or /test'
        }
    })


@app.route('/api_tester.html')
@app.route('/test')
def serve_tester():
    """Serve the API tester HTML page."""
    tester_path = os.path.join(project_root, 'backend', 'test', 'api_tester.html')
    if os.path.exists(tester_path):
        with open(tester_path, 'r') as f:
            return f.read()
    return jsonify({'error': 'API tester not found'}), 404


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'PanoStitch Backend API',
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/api/stitch', methods=['POST'])
def stitch_images():
    """
    Main endpoint to stitch images.
    
    Expects:
    - files: Multiple image files
    - use_dnn: Boolean (optional, default False)
    - use_harris: Boolean (optional, default False)
    - use_cylindrical: Boolean (optional, default True)
    - focal_length: Float (optional, default 1200.0)
    - resize: Integer (optional, default 800)
    
    Returns:
    - Stitched panorama image or error message
    """
    try:
        # Check if files are present
        if 'files' not in request.files or len(request.files.getlist('files')) == 0:
            return jsonify({
                'error': 'No images provided',
                'message': 'Please provide at least 2 images for stitching'
            }), 400

        files = request.files.getlist('files')

        # Validate number of images
        if len(files) < 2:
            return jsonify({
                'error': 'Insufficient images',
                'message': f'Received {len(files)} image(s). At least 2 images are required for stitching'
            }), 400

        # Validate all files
        image_paths = []
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        session_folder = os.path.join(UPLOAD_FOLDER, session_id)
        os.makedirs(session_folder, exist_ok=True)

        for file in files:
            if not file or not allowed_file(file.filename):
                return jsonify({
                    'error': 'Invalid file type',
                    'message': f'File "{file.filename}" has invalid extension. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
                }), 400

            filename = secure_filename(file.filename)
            filepath = os.path.join(session_folder, filename)
            file.save(filepath)
            image_paths.append(filepath)

        # Get parameters
        use_dnn = request.form.get('use_dnn', 'false').lower() == 'true'
        use_harris = request.form.get('use_harris', 'false').lower() == 'true'
        use_cylindrical = request.form.get('use_cylindrical', 'true').lower() == 'true'
        focal_length = float(request.form.get('focal_length', '1200.0'))
        resize = int(request.form.get('resize', '800'))

        print(f"\n=== PanoStitch API Request ===")
        print(f"Session ID: {session_id}")
        print(f"Images: {len(image_paths)}")
        print(f"Settings: DNN={use_dnn}, Harris={use_harris}, Cylindrical={use_cylindrical}")
        print(f"Focal Length: {focal_length}, Resize: {resize}")

        # Create stitcher
        stitcher = PanoStitch(
            resize_size=resize,
            ratio=0.75,
            gain_sigma_n=10.0,
            gain_sigma_g=0.1,
            use_harris=use_harris and not use_dnn,
            use_dnn=use_dnn,
            verbose=True,
            focal_length=focal_length,
            use_cylindrical=use_cylindrical,
        )

        # Stitch images
        panoramas, source_dir = stitcher.stitch(image_paths)

        if not panoramas:
            return jsonify({
                'error': 'Stitching failed',
                'message': 'No panorama was generated. Check image compatibility.'
            }), 400

        # Save results
        results_output_dir = os.path.join(RESULTS_FOLDER, session_id)
        os.makedirs(results_output_dir, exist_ok=True)

        saved_files = []
        for i, panorama in enumerate(panoramas):
            suffix = "_dnn" if use_dnn else ""
            output_path = os.path.join(results_output_dir, f"panorama_{i}{suffix}.jpg")
            success = cv2.imwrite(output_path, panorama)
            if success:
                saved_files.append({
                    'id': i,
                    'filename': f'panorama_{i}{suffix}.jpg',
                    'path': output_path,
                    'shape': panorama.shape
                })
                print(f"✓ Saved: {output_path}")
            else:
                print(f"✗ Failed to save: {output_path}")

        return jsonify({
            'status': 'success',
            'session_id': session_id,
            'panoramas_count': len(panoramas),
            'panoramas': saved_files,
            'message': f'Successfully created {len(panoramas)} panorama(s)'
        }), 200

    except Exception as e:
        print(f"Error in /api/stitch: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/api/stitch/result/<session_id>/<filename>', methods=['GET'])
def get_result(session_id, filename):
    """
    Download a stitched panorama result.
    
    Args:
        session_id: The session ID from stitch response
        filename: The filename of the panorama
    
    Returns:
        The panorama image file
    """
    try:
        filepath = os.path.join(RESULTS_FOLDER, session_id, filename)
        
        # Security check
        if not os.path.exists(filepath) or not filepath.startswith(RESULTS_FOLDER):
            return jsonify({'error': 'File not found'}), 404

        return send_file(
            filepath,
            mimetype='image/jpeg',
            as_attachment=True,
            download_name=filename
        ), 200

    except Exception as e:
        print(f"Error in /api/stitch/result: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve result',
            'message': str(e)
        }), 500


@app.route('/api/stitch/preview/<session_id>/<filename>', methods=['GET'])
def get_preview(session_id, filename):
    """
    Get a preview of the stitched panorama (as response body).
    
    Args:
        session_id: The session ID from stitch response
        filename: The filename of the panorama
    
    Returns:
        The panorama image as binary data
    """
    try:
        filepath = os.path.join(RESULTS_FOLDER, session_id, filename)
        
        # Security check
        if not os.path.exists(filepath) or not filepath.startswith(RESULTS_FOLDER):
            return jsonify({'error': 'File not found'}), 404

        # Read and return image
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'error': 'Failed to read image'}), 500

        # Encode to JPEG
        _, buffer = cv2.imencode('.jpg', img)
        return send_file(
            io.BytesIO(buffer.tobytes()),
            mimetype='image/jpeg'
        ), 200

    except Exception as e:
        print(f"Error in /api/stitch/preview: {str(e)}")
        return jsonify({
            'error': 'Failed to get preview',
            'message': str(e)
        }), 500


@app.route('/api/info', methods=['GET'])
def get_info():
    """Get API information and supported parameters."""
    return jsonify({
        'name': 'PanoStitch Backend API',
        'version': '1.0.0',
        'endpoints': {
            'health': 'GET /api/health',
            'stitch': 'POST /api/stitch',
            'get_result': 'GET /api/stitch/result/<session_id>/<filename>',
            'get_preview': 'GET /api/stitch/preview/<session_id>/<filename>',
            'info': 'GET /api/info'
        },
        'stitch_parameters': {
            'files': 'Multiple image files (required, at least 2)',
            'use_dnn': 'Use DISK+LightGlue matcher (boolean, default: false)',
            'use_harris': 'Use Harris corner detection (boolean, default: false)',
            'use_cylindrical': 'Apply cylindrical warping (boolean, default: true)',
            'focal_length': 'Focal length for cylindrical warping (float, default: 1200.0)',
            'resize': 'Max image dimension (integer, default: 800)'
        },
        'allowed_formats': list(ALLOWED_EXTENSIONS),
        'max_file_size': f'{MAX_FILE_SIZE / (1024 * 1024):.0f} MB'
    }), 200


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    return jsonify({
        'error': 'File too large',
        'message': f'Maximum file size is {MAX_FILE_SIZE / (1024 * 1024):.0f} MB'
    }), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'error': 'Not found',
        'message': 'The requested endpoint does not exist',
        'available_endpoints': [
            'GET /api/health',
            'POST /api/stitch',
            'GET /api/stitch/result/<session_id>/<filename>',
            'GET /api/stitch/preview/<session_id>/<filename>',
            'GET /api/info'
        ]
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("PanoStitch Backend API")
    print("="*60)
    print("Starting server on http://0.0.0.0:5000")
    print("Documentation available at http://localhost:5000/api/info")
    print("="*60 + "\n")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        use_reloader=False,
        threaded=True
    )
