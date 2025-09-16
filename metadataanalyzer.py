import json
import subprocess
import os
from datetime import datetime
from pathlib import Path

class Metadata:
    """
    Video Metadata Analysis Class for AI Generation Detection
    
    Extracts and analyzes video metadata for signs of AI generation through:
    - Device information scrutiny
    - Encoding pattern analysis  
    - Timestamp anomaly detection
    
    Returns metadata score (0.0-1.0) ready to combine with other detection scores.
    """
    
    def __init__(self):
        self.anomalies = []
        self.metadata_score = 0.0
        self.extracted_metadata = {}
        self.analysis_results = {}
    
    def extract_metadata_ffmpeg(self, video_path):
        """Extract comprehensive metadata using ffmpeg/ffprobe"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', '-show_entries',
                'format:stream', video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                return None
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
            return None
    
    def extract_metadata_pymediainfo(self, video_path):
        """Extract metadata using pymediainfo as fallback"""
        try:
            from pymediainfo import MediaInfo
            media_info = MediaInfo.parse(video_path)
            
            metadata = {'format': {}, 'streams': []}
            
            for track in media_info.tracks:
                if track.track_type == 'General':
                    metadata['format'] = {
                        'filename': track.file_name,
                        'format_name': track.format,
                        'duration': track.duration,
                        'size': track.file_size,
                        'creation_time': track.recorded_date,
                        'tags': {
                            'software': getattr(track, 'writing_application', ''),
                            'creation_time': track.recorded_date
                        }
                    }
                elif track.track_type == 'Video':
                    stream = {
                        'codec_name': track.codec,
                        'width': track.width,
                        'height': track.height,
                        'r_frame_rate': f"{track.frame_rate}/1" if track.frame_rate else "0/1",
                        'bit_rate': track.bit_rate,
                        'tags': {}
                    }
                    metadata['streams'].append(stream)
            
            return metadata
        except ImportError:
            return None
        except Exception:
            return None
    
    def extract_metadata_opencv(self, video_path, file_size, filename):
        """Basic metadata extraction using OpenCV as last resort"""
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = frame_count / fps if fps > 0 else 0
                
                metadata = {
                    'format': {
                        'filename': filename,
                        'duration': duration,
                        'size': file_size,
                        'tags': {}
                    },
                    'streams': [{
                        'codec_name': 'unknown',
                        'width': width,
                        'height': height,
                        'r_frame_rate': f"{fps}/1",
                        'tags': {}
                    }]
                }
                
                cap.release()
                return metadata
            else:
                return None
        except Exception:
            return None
    
    def check_device_info_anomalies(self, metadata):
        """Analyze device information for AI generation indicators"""
        anomalies = []
        score = 0.0
        
        format_tags = metadata.get('format', {}).get('tags', {})
        
        # Critical device metadata fields
        device_fields = {
            'make': ['make', 'manufacturer', 'camera_make'],
            'model': ['model', 'camera_model', 'device'],
            'software': ['software', 'encoder', 'writing_application'],
            'creation_time': ['creation_time', 'date'],
            'location': ['location', 'gps', 'coordinates']
        }
        
        missing_fields = []
        for field_type, field_names in device_fields.items():
            found = any(name in format_tags for name in field_names)
            if not found:
                missing_fields.append(field_type)
        
        # Score based on missing device info
        if len(missing_fields) >= 4:
            anomalies.append({
                'type': 'Critical Missing Device Info',
                'severity': 'High',
                'description': f"Missing essential device metadata: {', '.join(missing_fields)}",
                'score_impact': 0.5
            })
            score += 0.5
        elif len(missing_fields) >= 2:
            anomalies.append({
                'type': 'Missing Device Info',
                'severity': 'Medium', 
                'description': f"Missing device metadata: {', '.join(missing_fields)}",
                'score_impact': 0.3
            })
            score += 0.3
        
        # Check for suspicious software signatures
        software_fields = ['software', 'encoder', 'writing_application', 'creation_tool']
        software_value = ''
        for field in software_fields:
            if field in format_tags:
                software_value = str(format_tags[field]).lower()
                break
        
        # AI/Synthetic generation indicators
        ai_indicators = [
            'synthetic', 'generated', 'artificial', 'deepfake', 'ai', 
            'gan', 'neural', 'diffusion', 'stable', 'midjourney'
        ]
        
        # Generic/suspicious software indicators  
        suspicious_software = [
            'ffmpeg', 'unknown', 'handbrake', 'generic', 'default',
            'python', 'opencv', 'moviepy', 'imageio'
        ]
        
        for indicator in ai_indicators:
            if indicator in software_value:
                anomalies.append({
                    'type': 'AI Generation Software',
                    'severity': 'High',
                    'description': f"Software signature indicates AI generation: '{indicator}'",
                    'score_impact': 0.6
                })
                score += 0.6
                break
        
        for indicator in suspicious_software:
            if indicator in software_value:
                anomalies.append({
                    'type': 'Generic/Suspicious Software',
                    'severity': 'Medium',
                    'description': f"Generic encoding software detected: '{indicator}'",
                    'score_impact': 0.2
                })
                score += 0.2
                break
        
        return anomalies, min(score, 1.0)
    
    def check_encoding_anomalies(self, metadata):
        """Analyze encoding patterns for AI generation indicators"""
        anomalies = []
        score = 0.0
        
        streams = metadata.get('streams', [])
        video_stream = next((s for s in streams if 'width' in s), None)
        
        if not video_stream:
            return anomalies, 0.0
        
        width = video_stream.get('width', 0)
        height = video_stream.get('height', 0)
        
        # Resolution analysis
        if width > 0 and height > 0:
            # Common camera/phone resolutions
            standard_resolutions = [
                (1920, 1080), (1280, 720), (854, 480), (640, 360),
                (3840, 2160), (2560, 1440), (1366, 768), (1024, 576),
                (720, 480), (640, 480), (320, 240), (1440, 1080),
                (2048, 1080), (1334, 750), (1136, 640)  # iPhone sizes
            ]
            
            aspect_ratio = width / height
            
            # Check for non-standard resolutions
            if (width, height) not in standard_resolutions:
                # Check aspect ratios common in AI generation
                ai_ratios = [1.0, 1.77, 0.75, 1.25]  # Square, 16:9, 3:4, 5:4
                is_ai_ratio = any(abs(aspect_ratio - ratio) < 0.05 for ratio in ai_ratios)
                
                if is_ai_ratio and width % 64 == 0 and height % 64 == 0:
                    # Dimensions divisible by 64 (common in AI models)
                    anomalies.append({
                        'type': 'AI-Typical Resolution',
                        'severity': 'Medium',
                        'description': f"Resolution {width}×{height} typical of AI generation (64-divisible)",
                        'score_impact': 0.3
                    })
                    score += 0.3
                elif not any(abs(aspect_ratio - r) < 0.1 for r in [16/9, 4/3, 1.85, 2.35]):
                    anomalies.append({
                        'type': 'Unusual Aspect Ratio',
                        'severity': 'Low',
                        'description': f"Non-standard aspect ratio: {aspect_ratio:.2f}",
                        'score_impact': 0.1
                    })
                    score += 0.1
        
        # Frame rate analysis
        fps_str = video_stream.get('r_frame_rate', '0/1')
        try:
            if '/' in fps_str:
                num, den = map(float, fps_str.split('/'))
                fps = num / den if den > 0 else 0
            else:
                fps = float(fps_str)
            
            # AI generation often uses specific frame rates
            ai_fps_ranges = [(24, 30), (59, 61)]  # Exact 24-30fps, 60fps
            standard_fps = [23.976, 24, 25, 29.97, 30, 50, 59.94, 60, 120]
            
            if fps > 0:
                is_standard = any(abs(fps - std_fps) < 0.1 for std_fps in standard_fps)
                
                if not is_standard:
                    if 20 <= fps <= 35 or 55 <= fps <= 65:
                        anomalies.append({
                            'type': 'AI-Typical Frame Rate',
                            'severity': 'Medium',
                            'description': f"Frame rate {fps:.1f} common in AI generation",
                            'score_impact': 0.2
                        })
                        score += 0.2
                    else:
                        anomalies.append({
                            'type': 'Unusual Frame Rate',
                            'severity': 'Low',
                            'description': f"Non-standard frame rate: {fps:.1f}",
                            'score_impact': 0.1
                        })
                        score += 0.1
        
        except (ValueError, ZeroDivisionError):
            anomalies.append({
                'type': 'Malformed Frame Rate',
                'severity': 'Medium',
                'description': f"Invalid frame rate data: {fps_str}",
                'score_impact': 0.2
            })
            score += 0.2
        
        # Codec analysis
        codec = video_stream.get('codec_name', '').lower()
        suspicious_codecs = ['raw', 'uncompressed', 'synthetic']
        ai_preferred_codecs = ['h264', 'h265', 'hevc']
        
        for sus_codec in suspicious_codecs:
            if sus_codec in codec:
                anomalies.append({
                    'type': 'Suspicious Codec',
                    'severity': 'High',
                    'description': f"Unusual codec for natural video: {codec}",
                    'score_impact': 0.3
                })
                score += 0.3
                break
        
        return anomalies, min(score, 1.0)
    
    def check_timestamp_anomalies(self, metadata):
        """Analyze timestamps for inconsistencies and anomalies"""
        anomalies = []
        score = 0.0
        
        format_data = metadata.get('format', {})
        tags = format_data.get('tags', {})
        
        # Look for creation time in various fields
        time_fields = ['creation_time', 'date', 'datetime', 'timestamp']
        creation_time = None
        
        for field in time_fields:
            if field in tags and tags[field]:
                creation_time = tags[field]
                break
        
        if creation_time:
            try:
                # Handle different timestamp formats
                creation_time_clean = creation_time.replace('Z', '+00:00')
                if 'T' not in creation_time_clean:
                    # Try parsing date-only format
                    created = datetime.strptime(creation_time_clean[:10], '%Y-%m-%d')
                else:
                    created = datetime.fromisoformat(creation_time_clean)
                
                now = datetime.now(created.tzinfo) if created.tzinfo else datetime.now()
                
                # Future timestamp check
                if created > now:
                    anomalies.append({
                        'type': 'Future Timestamp',
                        'severity': 'High',
                        'description': f"Creation time in future: {creation_time}",
                        'score_impact': 0.5
                    })
                    score += 0.5
                
                # Very old timestamp check (pre-digital video era)
                elif created.year < 1995:
                    anomalies.append({
                        'type': 'Anachronistic Timestamp',
                        'severity': 'High',
                        'description': f"Creation time predates digital video: {creation_time}",
                        'score_impact': 0.4
                    })
                    score += 0.4
                
                # Suspiciously precise timestamps (exactly on hour/minute)
                elif (created.minute == 0 and created.second == 0 and 
                      created.microsecond == 0):
                    anomalies.append({
                        'type': 'Artificially Precise Timestamp',
                        'severity': 'Low',
                        'description': f"Timestamp too precise (exact hour): {creation_time}",
                        'score_impact': 0.1
                    })
                    score += 0.1
                    
            except (ValueError, TypeError):
                anomalies.append({
                    'type': 'Invalid Timestamp Format',
                    'severity': 'Medium',
                    'description': f"Malformed timestamp: {creation_time}",
                    'score_impact': 0.2
                })
                score += 0.2
        else:
            # Missing timestamp is suspicious for recent videos
            anomalies.append({
                'type': 'Missing Creation Timestamp',
                'severity': 'Medium',
                'description': "No creation timestamp found",
                'score_impact': 0.2
            })
            score += 0.2
        
        return anomalies, min(score, 1.0)
    
    def analyze(self, video_path, file_size=None, filename=None):
        """
        Main analysis method
        
        Args:
            video_path (str): Path to video file
            file_size (int, optional): File size in bytes
            filename (str, optional): Original filename
            
        Returns:
            dict: Analysis results with score and details
        """
        self.anomalies = []
        self.metadata_score = 0.0
        self.extracted_metadata = {}
        
        # Extract metadata using multiple methods
        metadata = self.extract_metadata_ffmpeg(video_path)
        if not metadata:
            metadata = self.extract_metadata_pymediainfo(video_path)
        if not metadata and file_size and filename:
            metadata = self.extract_metadata_opencv(video_path, file_size, filename)
        
        if not metadata:
            self.anomalies.append({
                'type': 'Metadata Extraction Failed',
                'severity': 'High',
                'description': 'Unable to extract metadata from video file',
                'score_impact': 0.7
            })
            self.metadata_score = 0.7
            
            self.analysis_results = {
                'metadata_score': self.metadata_score,
                'anomalies': self.anomalies,
                'extracted_metadata': {},
                'analysis_summary': {
                    'total_anomalies': len(self.anomalies),
                    'high_severity': 1,
                    'medium_severity': 0,
                    'low_severity': 0,
                    'risk_level': 'HIGH'
                }
            }
            return self.analysis_results
        
        self.extracted_metadata = metadata
        
        # Run anomaly detection
        device_anomalies, device_score = self.check_device_info_anomalies(metadata)
        encoding_anomalies, encoding_score = self.check_encoding_anomalies(metadata)
        timestamp_anomalies, timestamp_score = self.check_timestamp_anomalies(metadata)
        
        # Combine anomalies
        self.anomalies.extend(device_anomalies)
        self.anomalies.extend(encoding_anomalies)
        self.anomalies.extend(timestamp_anomalies)
        
        # Calculate weighted metadata score
        weights = {'device': 0.4, 'encoding': 0.3, 'timestamp': 0.3}
        self.metadata_score = (
            device_score * weights['device'] +
            encoding_score * weights['encoding'] +
            timestamp_score * weights['timestamp']
        )
        
        self.metadata_score = min(self.metadata_score, 1.0)
        
        # Generate analysis summary
        severity_counts = {
            'High': len([a for a in self.anomalies if a['severity'] == 'High']),
            'Medium': len([a for a in self.anomalies if a['severity'] == 'Medium']),
            'Low': len([a for a in self.anomalies if a['severity'] == 'Low'])
        }
        
        # Determine risk level
        if self.metadata_score >= 0.6:
            risk_level = 'HIGH'
        elif self.metadata_score >= 0.3:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        self.analysis_results = {
            'metadata_score': self.metadata_score,
            'anomalies': self.anomalies,
            'extracted_metadata': self.extracted_metadata,
            'component_scores': {
                'device_score': device_score,
                'encoding_score': encoding_score,
                'timestamp_score': timestamp_score
            },
            'analysis_summary': {
                'total_anomalies': len(self.anomalies),
                'high_severity': severity_counts['High'],
                'medium_severity': severity_counts['Medium'], 
                'low_severity': severity_counts['Low'],
                'risk_level': risk_level
            }
        }
        
        return self.analysis_results
    
    def get_score(self):
        """Get the metadata score (0.0-1.0)"""
        return self.metadata_score
    
    def get_anomalies(self):
        """Get list of detected anomalies"""
        return self.anomalies
    
    def get_metadata(self):
        """Get extracted metadata"""
        return self.extracted_metadata
    
    def get_results(self):
        """Get complete analysis results"""
        return self.analysis_results