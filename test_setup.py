"""
Test script to verify the video summarizer setup
Run this to check if all components are working correctly
"""

import sys
import torch
from PIL import Image
import numpy as np

def test_imports():
    """Test if all required packages are installed"""
    print("Testing imports...")
    try:
        import streamlit
        print("✓ Streamlit installed")
    except ImportError:
        print("✗ Streamlit not found")
        return False
    
    try:
        import cv2
        print("✓ OpenCV installed")
    except ImportError:
        print("✗ OpenCV not found")
        return False
    
    try:
        import transformers
        print("✓ Transformers installed")
    except ImportError:
        print("✗ Transformers not found")
        return False
    
    return True

def test_gpu():
    """Test GPU availability"""
    print("\nTesting GPU availability...")
    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return True
    else:
        print("⚠ No GPU available - will use CPU (slower)")
        return False

def test_video_processor():
    """Test the video processor module"""
    print("\nTesting VideoProcessor...")
    try:
        from video_processor import VideoProcessor
        processor = VideoProcessor()
        print("✓ VideoProcessor initialized successfully")
        
        # Test timestamp formatting
        timestamp = processor.format_timestamp(125.5)
        print(f"  Timestamp formatting works: {timestamp}")
        return True
    except Exception as e:
        print(f"✗ VideoProcessor error: {e}")
        return False

def test_model_handler():
    """Test if model can be initialized (without actually loading it)"""
    print("\nTesting ModelHandler structure...")
    try:
        from model_handler import CosmosModelHandler
        print("✓ ModelHandler module loaded successfully")
        print("  Note: Actual model loading will happen when you run the app")
        return True
    except Exception as e:
        print(f"✗ ModelHandler error: {e}")
        return False

def test_summarizer():
    """Test the summarizer module"""
    print("\nTesting Summarizer...")
    try:
        from summarys.summarizer import VideoSummarizer
        summarizer = VideoSummarizer()
        print("✓ VideoSummarizer initialized successfully")
        
        # Test with dummy data
        dummy_descriptions = [
            {"frame_index": 0, "description": "A person walking in a park."},
            {"frame_index": 1, "description": "The person sits on a bench."},
        ]
        dummy_timestamps = [0.0, 5.0]
        
        summary = summarizer.generate_summary(dummy_descriptions, dummy_timestamps, "concise")
        print("✓ Summary generation works")
        print(f"  Sample output length: {len(summary)} characters")
        return True
    except Exception as e:
        print(f"✗ Summarizer error: {e}")
        return False

def test_create_dummy_image():
    """Test PIL image creation"""
    print("\nTesting image handling...")
    try:
        # Create a dummy image
        dummy_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        dummy_image = Image.fromarray(dummy_array)
        print(f"✓ Can create PIL images: {dummy_image.size}")
        return True
    except Exception as e:
        print(f"✗ Image handling error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Video Summarizer - Setup Verification")
    print("=" * 60)
    
    results = {
        "Imports": test_imports(),
        "GPU": test_gpu(),
        "VideoProcessor": test_video_processor(),
        "ModelHandler": test_model_handler(),
        "Summarizer": test_summarizer(),
        "Image Handling": test_create_dummy_image(),
    }
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed! ✓")
        print("You can now run: streamlit run app.py")
    else:
        print("Some tests failed. Please check the errors above.")
        print("Make sure all dependencies are installed:")
        print("  pip install -r requirements.txt")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
