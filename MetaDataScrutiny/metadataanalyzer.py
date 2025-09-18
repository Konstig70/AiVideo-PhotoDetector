from pymediainfo import MediaInfo

class metadata:
    def __init__(self, filepath):
        self.filepath = filepath
        self.metadata = {}
        self.suspicion_score = 0
    
    def extract_metadata(self):
        media_info = MediaInfo.parse(self.filepath)
        for track in media_info.tracks:
            if track.track_type == "Video":
                self.metadata["resolution"] = f"{track.width}x{track.height}" if track.width and track.height else None
                self.metadata["codec"] = track.codec_id
            if track.track_type == "General":
                self.metadata["device_info"] = track.encoded_library or track.writing_application
                self.metadata["timestamps"] = track.tagged_date or track.file_last_modification_date
    
    def analyze(self):
        self.extract_metadata()
        # listataan tarkistettavat kentät
        required = ["resolution", "codec", "device_info", "timestamps"]
        for field in required:
            if not self.metadata.get(field):
                self.suspicion_score += 1

        print(f"Metadata: {metadata}")
        print(f"Suspicion score: {self.suspicion_score}")
        
        return {
            "metadata": self.metadata,
            "metadata_anomaly_score": self.suspicion_score
        }
