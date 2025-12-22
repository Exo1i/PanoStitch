"""
Client helper for testing the PanoStitch API
Usage: python api_client.py <image_folder> [options]
"""

import requests
import sys
import argparse
from pathlib import Path
import json

API_BASE_URL = 'http://localhost:5000/api'


def stitch_images(image_folder, use_dnn=False, use_harris=False, use_cylindrical=True, 
                  focal_length=1200.0, resize=800):
    """
    Call the stitch API with images from a folder.
    
    Args:
        image_folder: Path to folder containing images
        use_dnn: Use DISK+LightGlue matcher
        use_harris: Use Harris corner detection
        use_cylindrical: Apply cylindrical warping
        focal_length: Focal length for warping
        resize: Max image dimension
    
    Returns:
        Response from API
    """
    
    image_folder = Path(image_folder)
    if not image_folder.is_dir():
        print(f"Error: {image_folder} is not a directory")
        sys.exit(1)
    
    # Find images
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [f for f in image_folder.iterdir() if f.suffix.lower() in valid_extensions]
    
    if not image_files:
        print(f"Error: No valid images found in {image_folder}")
        sys.exit(1)
    
    print(f"Found {len(image_files)} images:")
    for img in sorted(image_files):
        print(f"  - {img.name}")
    
    # Prepare files
    files = []
    for img_path in sorted(image_files):
        files.append(('files', open(str(img_path), 'rb')))
    
    # Prepare data
    data = {
        'use_dnn': 'true' if use_dnn else 'false',
        'use_harris': 'true' if use_harris else 'false',
        'use_cylindrical': 'true' if use_cylindrical else 'false',
        'focal_length': str(focal_length),
        'resize': str(resize),
    }
    
    print("\nSending request to API...")
    print(f"Parameters: {json.dumps(data, indent=2)}")
    
    try:
        response = requests.post(
            f'{API_BASE_URL}/stitch',
            files=files,
            data=data,
            timeout=600  # 10 minutes timeout
        )
        
        # Close files
        for _, file_obj in files:
            file_obj.close()
        
        print(f"\nAPI Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n✓ Stitching successful!")
            print(f"Session ID: {result['session_id']}")
            print(f"Panoramas created: {result['panoramas_count']}")
            
            for pano in result['panoramas']:
                print(f"\nPanorama {pano['id']}:")
                print(f"  Filename: {pano['filename']}")
                print(f"  Dimensions: {pano['shape']}")
                print(f"  Download URL: {API_BASE_URL}/stitch/result/{result['session_id']}/{pano['filename']}")
            
            return result
        else:
            error = response.json()
            print(f"\n✗ Error: {error.get('error')}")
            print(f"Message: {error.get('message')}")
            return None
    
    except requests.exceptions.ConnectionError:
        print(f"\n✗ Error: Could not connect to API at {API_BASE_URL}")
        print("Make sure the backend server is running: python app.py")
    except requests.exceptions.Timeout:
        print("\n✗ Error: Request timeout (stitching took too long)")
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")


def download_result(session_id, filename, output_path=None):
    """
    Download a stitched panorama result.
    
    Args:
        session_id: Session ID from stitch response
        filename: Filename of panorama
        output_path: Where to save the file (default: current directory)
    """
    
    if not output_path:
        output_path = Path(filename)
    else:
        output_path = Path(output_path)
    
    print(f"Downloading {filename}...")
    
    try:
        response = requests.get(
            f'{API_BASE_URL}/stitch/result/{session_id}/{filename}',
            timeout=60
        )
        
        if response.status_code == 200:
            output_path.write_bytes(response.content)
            print(f"✓ Saved to {output_path}")
        else:
            print(f"✗ Error: {response.json()}")
    
    except Exception as e:
        print(f"✗ Error downloading: {str(e)}")


def get_api_info():
    """Get API information."""
    try:
        response = requests.get(f'{API_BASE_URL}/info')
        if response.status_code == 200:
            info = response.json()
            print(json.dumps(info, indent=2))
        else:
            print(f"Error: {response.json()}")
    except Exception as e:
        print(f"Error: {str(e)}")


def health_check():
    """Check API health."""
    try:
        response = requests.get(f'{API_BASE_URL}/health')
        if response.status_code == 200:
            info = response.json()
            print(f"✓ API is healthy")
            print(f"  Service: {info['service']}")
            print(f"  Status: {info['status']}")
        else:
            print(f"✗ API health check failed")
    except Exception as e:
        print(f"✗ Cannot connect to API: {str(e)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PanoStitch API Client')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Stitch command
    stitch_parser = subparsers.add_parser('stitch', help='Stitch images')
    stitch_parser.add_argument('folder', help='Folder containing images')
    stitch_parser.add_argument('--dnn', action='store_true', help='Use DISK+LightGlue matcher')
    stitch_parser.add_argument('--harris', action='store_true', help='Use Harris corner detection')
    stitch_parser.add_argument('--no-cylindrical', action='store_true', help='Disable cylindrical warping')
    stitch_parser.add_argument('--focal-length', type=float, default=1200.0, help='Focal length (default: 1200.0)')
    stitch_parser.add_argument('--resize', type=int, default=800, help='Max image dimension (default: 800)')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download result')
    download_parser.add_argument('session_id', help='Session ID')
    download_parser.add_argument('filename', help='Filename')
    download_parser.add_argument('--output', help='Output path (default: current directory)')
    
    # Info command
    subparsers.add_parser('info', help='Get API information')
    
    # Health command
    subparsers.add_parser('health', help='Check API health')
    
    args = parser.parse_args()
    
    if args.command == 'stitch':
        stitch_images(
            args.folder,
            use_dnn=args.dnn,
            use_harris=args.harris,
            use_cylindrical=not args.no_cylindrical,
            focal_length=args.focal_length,
            resize=args.resize
        )
    elif args.command == 'download':
        download_result(args.session_id, args.filename, args.output)
    elif args.command == 'info':
        get_api_info()
    elif args.command == 'health':
        health_check()
    else:
        parser.print_help()
