from typing import List, Dict
from video_processor import VideoProcessor

class VideoSummarizer:
    """Generates coherent video summaries from frame descriptions"""
    
    def __init__(self):
        self.processor = VideoProcessor()
    
    def generate_summary(
        self,
        frame_descriptions: List[Dict[str, str]],
        timestamps: List[float],
        style: str = "detailed"
    ) -> str:
        """
        Generate a comprehensive summary from frame descriptions
        
        Args:
            frame_descriptions: List of dicts with frame analysis
            timestamps: List of timestamps for each frame
            style: Summary style ("detailed", "concise", "bullet points")
            
        Returns:
            Formatted summary text
        """
        if not frame_descriptions:
            return "No content to summarize."
        
        if style == "bullet points":
            return self._generate_bullet_summary(frame_descriptions, timestamps)
        elif style == "concise":
            return self._generate_concise_summary(frame_descriptions, timestamps)
        else:
            return self._generate_detailed_summary(frame_descriptions, timestamps)
    
    def _generate_detailed_summary(
        self,
        frame_descriptions: List[Dict[str, str]],
        timestamps: List[float]
    ) -> str:
        """Generate a detailed narrative summary"""
        summary_parts = []
        
        # Introduction
        total_duration = self.processor.format_timestamp(timestamps[-1]) if timestamps else "unknown"
        summary_parts.append(
            f"**Video Summary** (Duration: {total_duration})\n"
        )
        
        # Group frames into scenes
        scenes = self._group_into_scenes(frame_descriptions, timestamps)
        
        # Generate narrative for each scene
        for scene_idx, scene in enumerate(scenes, 1):
            start_time = self.processor.format_timestamp(scene['start_timestamp'])
            end_time = self.processor.format_timestamp(scene['end_timestamp'])
            
            summary_parts.append(
                f"\n**Scene {scene_idx}** ({start_time} - {end_time}):\n"
            )
            
            # Combine descriptions in the scene
            scene_description = self._synthesize_scene_description(scene['frames'])
            summary_parts.append(scene_description)
        
        # Conclusion
        summary_parts.append(
            f"\n\n**Summary**: This video spans {total_duration} and contains "
            f"{len(frame_descriptions)} key moments across {len(scenes)} distinct scenes."
        )
        
        return "\n".join(summary_parts)
    
    def _generate_concise_summary(
        self,
        frame_descriptions: List[Dict[str, str]],
        timestamps: List[float]
    ) -> str:
        """Generate a brief, high-level summary"""
        summary_parts = []
        
        # Extract key themes
        all_descriptions = " ".join([fd['description'] for fd in frame_descriptions])
        
        # Simple overview
        summary_parts.append("**Quick Summary**\n")
        
        # Take descriptions from beginning, middle, and end
        key_indices = [
            0,  # Beginning
            len(frame_descriptions) // 2,  # Middle
            len(frame_descriptions) - 1  # End
        ]
        
        for idx in key_indices:
            if idx < len(frame_descriptions):
                timestamp = self.processor.format_timestamp(timestamps[idx])
                desc = frame_descriptions[idx]['description']
                # Truncate if too long
                if len(desc) > 200:
                    desc = desc[:200] + "..."
                summary_parts.append(f"At {timestamp}: {desc}")
        
        return "\n\n".join(summary_parts)
    
    def _generate_bullet_summary(
        self,
        frame_descriptions: List[Dict[str, str]],
        timestamps: List[float]
    ) -> str:
        """Generate a bullet-point summary"""
        summary_parts = []
        
        summary_parts.append("**Video Summary - Key Points**\n")
        
        for idx, (frame_desc, timestamp) in enumerate(zip(frame_descriptions, timestamps)):
            time_str = self.processor.format_timestamp(timestamp)
            desc = frame_desc['description']
            
            # Clean up description
            desc = desc.strip()
            if not desc.endswith('.'):
                desc += '.'
            
            summary_parts.append(f"• **{time_str}**: {desc}")
        
        return "\n".join(summary_parts)
    
    def _group_into_scenes(
        self,
        frame_descriptions: List[Dict[str, str]],
        timestamps: List[float],
        max_scene_gap: float = 30.0
    ) -> List[Dict]:
        """
        Group frames into scenes based on time gaps
        
        Args:
            frame_descriptions: Frame analysis data
            timestamps: Timestamps for each frame
            max_scene_gap: Maximum time gap (seconds) within a scene
            
        Returns:
            List of scene dictionaries
        """
        scenes = []
        current_scene = {
            'frames': [],
            'start_timestamp': timestamps[0] if timestamps else 0,
            'end_timestamp': timestamps[0] if timestamps else 0
        }
        
        for idx, (frame_desc, timestamp) in enumerate(zip(frame_descriptions, timestamps)):
            # Check if we should start a new scene
            if current_scene['frames'] and (timestamp - current_scene['end_timestamp']) > max_scene_gap:
                scenes.append(current_scene)
                current_scene = {
                    'frames': [],
                    'start_timestamp': timestamp,
                    'end_timestamp': timestamp
                }
            
            current_scene['frames'].append(frame_desc)
            current_scene['end_timestamp'] = timestamp
        
        # Add the last scene
        if current_scene['frames']:
            scenes.append(current_scene)
        
        return scenes
    
    def _synthesize_scene_description(self, frames: List[Dict[str, str]]) -> str:
        """
        Create a cohesive description from multiple frame descriptions
        
        Args:
            frames: List of frame description dictionaries
            
        Returns:
            Synthesized scene description
        """
        if not frames:
            return "No content in this scene."
        
        if len(frames) == 1:
            return frames[0]['description']
        
        # Combine descriptions intelligently
        descriptions = [f['description'] for f in frames]
        
        # For now, join with transitions
        # In a more advanced version, you could use an LLM to synthesize these
        synthesized = descriptions[0]
        
        for desc in descriptions[1:]:
            # Add transition words
            synthesized += f" Subsequently, {desc.lower() if desc else ''}"
        
        return synthesized
    
    def extract_key_topics(self, frame_descriptions: List[Dict[str, str]]) -> List[str]:
        """
        Extract main topics/themes from the video
        
        Args:
            frame_descriptions: Frame analysis data
            
        Returns:
            List of key topics
        """
        # This is a simplified version
        # You could use NLP techniques or another LLM call for better topic extraction
        topics = set()
        
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                       'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                       'this', 'that', 'these', 'those'}
        
        for frame_desc in frame_descriptions:
            words = frame_desc['description'].lower().split()
            # Extract potential topics (nouns, longer words)
            for word in words:
                cleaned = word.strip('.,!?;:"\'')
                if len(cleaned) > 5 and cleaned not in common_words:
                    topics.add(cleaned)
        
        return list(topics)[:10]  # Return top 10
