import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple

class VideoProcessor:
    """Handles video processing and frame extraction"""
    
    def __init__(self):
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv']
    
    def extract_frames(
        self,
        video_path: str,
        interval_seconds: int = 2,
        max_frames: int = 20,
        resize_width: int = 512
    ) -> Tuple[List[Image.Image], List[float]]:
        """
        Extract frames from video at specified intervals
        
        Args:
            video_path: Path to video file
            interval_seconds: Extract one frame every N seconds
            max_frames: Maximum number of frames to extract
            resize_width: Resize frames to this width (maintains aspect ratio)
            
        Returns:
            Tuple of (list of PIL Images, list of timestamps in seconds)
        """
        frames = []
        timestamps = []
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"Video info: {fps} FPS, {total_frames} frames, {duration:.2f} seconds")
        
        # Calculate frame interval
        frame_interval = int(fps * interval_seconds)
        
        # Extract frames
        frame_count = 0
        extracted_count = 0
        
        while cap.isOpened() and extracted_count < max_frames:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Extract frame at intervals
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize frame
                height, width = frame_rgb.shape[:2]
                if width > resize_width:
                    aspect_ratio = height / width
                    new_height = int(resize_width * aspect_ratio)
                    frame_rgb = cv2.resize(frame_rgb, (resize_width, new_height))
                
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
                
                # Calculate timestamp
                timestamp = frame_count / fps
                timestamps.append(timestamp)
                
                extracted_count += 1
            
            frame_count += 1
        
        cap.release()
        
        print(f"Extracted {len(frames)} frames from video")
        return frames, timestamps
    
    def extract_keyframes(
        self,
        video_path: str,
        max_frames: int = 20,
        threshold: float = 30.0
    ) -> Tuple[List[Image.Image], List[float]]:
        """
        Extract keyframes based on scene changes (more intelligent sampling)
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
            threshold: Scene change detection threshold (higher = fewer frames)
            
        Returns:
            Tuple of (list of PIL Images, list of timestamps in seconds)
        """
        frames = []
        timestamps = []
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        prev_frame = None
        frame_count = 0
        
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Convert to grayscale for comparison
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Check if this is a keyframe (scene change)
            is_keyframe = False
            if prev_frame is None:
                is_keyframe = True  # Always include first frame
            else:
                # Calculate difference between frames
                diff = cv2.absdiff(prev_frame, gray)
                mean_diff = np.mean(diff)
                
                if mean_diff > threshold:
                    is_keyframe = True
            
            if is_keyframe:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize
                height, width = frame_rgb.shape[:2]
                if width > 512:
                    aspect_ratio = height / width
                    new_height = int(512 * aspect_ratio)
                    frame_rgb = cv2.resize(frame_rgb, (512, new_height))
                
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
                
                # Calculate timestamp
                timestamp = frame_count / fps
                timestamps.append(timestamp)
            
            prev_frame = gray
            frame_count += 1
        
        cap.release()
        
        print(f"Extracted {len(frames)} keyframes from video")
        return frames, timestamps
    
    def format_timestamp(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
