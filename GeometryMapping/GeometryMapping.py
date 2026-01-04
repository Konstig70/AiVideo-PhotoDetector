import cv2
import mediapipe as mp
import math
import os
import numpy as np
from queue import Queue
from threading import Thread

class GeometryMapper:

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_face_mesh = mp.solutions.face_mesh

    def __init__(self):
        # Motion-related
        self.motion_queue = Queue()
        self.motion_scores = []
        self.prev_gray_motion = None

        # Frame and anomaly stats
        self.total_frames = 0
        self.anomaly_frames = 0
        self.anomaly_score = 0
        self.anomaly_multiplier = 0.1
        self.previous_frame_anomalous = False
        self.frame_anomaly = False
        self.finger_angle_anomaly_frames = 0
        self.face_distance_anomaly_frames = 0
        self.arm_length_ratio_anomaly_frames = 0
        self.shoulder_to_shoulder_width_anomaly_frames = 0

    def analyze_video(self, video_path):
        """
        Analyze a video for anatomical anomalies.
        Returns a dict with anomaly stats.
        """
        # Open video
        capture = cv2.VideoCapture(video_path)

        # Initialize models
        hands = GeometryMapper.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        pose = GeometryMapper.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
        face_mesh = GeometryMapper.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

        # Per-frame tracking lists
        finger_angles = []
        pose_shoulder_widths = []
        face_distances = []

        total_frames_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        # Start motion thread
        motion_thread = Thread(target=self._motion_worker)
        motion_thread.start()

        try:
            while capture.isOpened():
                self.frame_anomaly = False
                ret, frame = capture.read()
                if not ret:
                    break

                self.total_frames += 1

                # Motion processing every 5 frames
                if self.total_frames % 5 == 0:
                    self.motion_queue.put((self.total_frames, frame.copy()))

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process hands, pose, face
                self.process_hands(rgb, hands, finger_angles, frame)
                self.anomaly_multiplier = 0.1
                self.process_pose(rgb, pose, pose_shoulder_widths, frame)
                self.anomaly_multiplier = 0.1
                self.process_face(rgb, face_mesh, face_distances, frame)

                if self.frame_anomaly:
                    self.anomaly_frames += 1

        finally:
            capture.release()
            # Stop motion thread
            self.motion_queue.put(None)
            motion_thread.join()
            # Clear remaining frames
            while not self.motion_queue.empty():
                self.motion_queue.get()

        # Calculate motion-based adjustment
        only_scores = [score for frame_id, score in self.motion_scores]
        avg_motion = np.mean(only_scores) if only_scores else 0
        anomaly_score = self.anomaly_score / max(self.total_frames, 1)
        anomaly_score = anomaly_score - (avg_motion / 100)

        # Generate justification text
        if anomaly_score < 0.025:
            justification = "Likely a real video"
        elif anomaly_score < 0.05:
            justification = "Probably a real video but some minor anomalies were detected"
        elif anomaly_score < 0.075:
            justification = "Low quality or edited video with some anomalies"
        elif anomaly_score < 0.1:
            justification = "Probably synthetic video with several anomalies"
        else:
            justification = "Highly suspicious, many anomalies detected"

        return {
            "total_frames": self.total_frames,
            "anomaly_frames": self.anomaly_frames,
            "anatomy_anomaly_rating": f"{anomaly_score:.3f} → {justification}",
            "finger_anomaly_frames": self.finger_angle_anomaly_frames,
            "arm_length_ratio_anomaly_frames": self.arm_length_ratio_anomaly_frames,
            "shoulder_to_shoulder_width_anomaly_frames": self.shoulder_to_shoulder_width_anomaly_frames,
            "face_distance_anomaly_frames": self.face_distance_anomaly_frames,
            "motion_score": float(avg_motion)
        }

    # ------------------ Motion Worker ------------------
    def _motion_worker(self):
        prev_gray = self.prev_gray_motion
        while True:
            item = self.motion_queue.get()
            if item is None:  # Sentinel to stop
                break
            frame_id, frame = item
            prev_gray, motion_score = self.calculate_motion(frame, prev_gray)
            self.motion_scores.append((frame_id, motion_score))
            self.motion_queue.task_done()
        self.prev_gray_motion = prev_gray

    def calculate_motion(self, frame, prev_gray):
        scale = 0.5
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small_gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale)
        if prev_gray is None:
            return small_gray, 0
        flow = cv2.calcOpticalFlowFarneback(prev_gray, small_gray, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_score = np.mean(magnitude)
        return small_gray, motion_score

    # ------------------ Face / Pose / Hands ------------------
    def process_face(self, rgb, face_mesh, face_distances, frame):
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            symmetric_pairs = [(33, 263), (61, 291), (159, 386)]
            distances = []
            for left_idx, right_idx in symmetric_pairs:
                left = landmarks[left_idx]
                right = landmarks[right_idx]
                distances.append(self.symmetry_distance(left, right))
            face_distances.append(distances)
            np_faces = np.array(face_distances)
            avg = np.mean(np_faces, axis=0)
            deviation = np.abs(np.array(distances) - avg)
            std_dev = np.std(np_faces, axis=0)
            if np.any(deviation > 2 * std_dev):
                if self.previous_frame_anomalous:
                    self.anomaly_multiplier = min(self.anomaly_multiplier + 0.3, 3.0)
                self.anomaly_score += 1 * self.anomaly_multiplier * 0.7
                self.previous_frame_anomalous = True
                self.frame_anomaly = True
                self.face_distance_anomaly_frames += 1
            else:
                self.previous_frame_anomalous = False
                self.anomaly_multiplier = 0.1

    def process_pose(self, rgb, pose, pose_shoulder_widths, frame):
        results = pose.process(rgb)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            ratios = self.get_limb_ratios(landmarks)
            assymetry = abs(ratios[0] - ratios[1])
            if assymetry > 0.45:
                if self.previous_frame_anomalous:
                    self.anomaly_multiplier = min(self.anomaly_multiplier + 0.3, 3.0)
                self.anomaly_score += 1 * self.anomaly_multiplier * 0.7
                self.previous_frame_anomalous = True
                self.frame_anomaly = True
                self.arm_length_ratio_anomaly_frames += 1
            else:
                self.previous_frame_anomalous = False
                self.anomaly_multiplier = 0.1

            # Shoulder width
            pose_shoulder_widths.append(ratios[2])
            np_shoulder = np.array(pose_shoulder_widths)
            avg_shoulder = np.mean(np_shoulder)
            deviation = abs(ratios[2] - avg_shoulder)
            if deviation > 3 * np.std(np_shoulder):
                if self.previous_frame_anomalous:
                    self.anomaly_multiplier = min(self.anomaly_multiplier + 0.3, 3.0)
                self.anomaly_score += 1 * self.anomaly_multiplier
                self.previous_frame_anomalous = True
                self.frame_anomaly = True
                self.shoulder_to_shoulder_width_anomaly_frames += 1
            else:
                self.previous_frame_anomalous = False
                self.anomaly_multiplier = 0.1

    def process_hands(self, rgb, hands, finger_angles, frame):
        results = hands.process(rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                angles, count = self.finger_angles(hand_landmarks)
                finger_angles.append(angles)
                np_angles = np.array(finger_angles)
                avg_angles = np.mean(np_angles, axis=0)
                deviation = np.abs(np.array(angles) - avg_angles)
                std_angles = np.std(np_angles, axis=0)
                if np.any(deviation > 2 * std_angles):
                    if self.previous_frame_anomalous:
                        self.anomaly_multiplier = min(self.anomaly_multiplier + 0.3, 3.0)
                    self.anomaly_score += 1 * self.anomaly_multiplier * 1.4
                    self.previous_frame_anomalous = True
                    self.frame_anomaly = True
                    self.finger_angle_anomaly_frames += 1
                else:
                    self.previous_frame_anomalous = False
                    self.anomaly_multiplier = 0.1

    # ------------------ Utilities ------------------
    def get_limb_ratios(self, landmarks):
        left_upper = self.segment_length(landmarks[11], landmarks[13])
        left_lower = self.segment_length(landmarks[13], landmarks[15])
        right_upper = self.segment_length(landmarks[12], landmarks[14])
        right_lower = self.segment_length(landmarks[14], landmarks[16])
        shoulder_width = self.segment_length(landmarks[11], landmarks[12])

        left_ratio = left_upper / left_lower if left_lower != 0 else 0
        right_ratio = right_upper / right_lower if right_lower != 0 else 0
        return [left_ratio, right_ratio, shoulder_width]

    def symmetry_distance(self, p_left, p_right):
        return abs(p_left.x - (1 - p_right.x))

    def segment_length(self, p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

    def finger_angles(self, hand_landmarks):
        fingers = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]]
        angles_frame = []
        count = 0
        for f in fingers:
            mcp, pip, dip, tip = [hand_landmarks.landmark[i] for i in f]
            length = self.distance(mcp, tip)
            angle1 = self.angle_between(mcp, pip, dip)
            angle2 = self.angle_between(pip, dip, tip)
            angles_frame.extend([angle1, angle2])
            if 40 <= angle1 <= 180 and 40 <= angle2 <= 180 and length >= 0.05:
                count += 1
        return angles_frame, count

    @staticmethod
    def distance(lm1, lm2):
        return math.sqrt((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2 + (lm1.z - lm2.z)**2)

    @staticmethod
    def angle_between(p1, p2, p3):
        a = (p1.x - p2.x, p1.y - p2.y)
        b = (p3.x - p2.x, p3.y - p2.y)
        dot = a[0]*b[0] + a[1]*b[1]
        mag_a = math.sqrt(a[0]**2 + a[1]**2)
        mag_b = math.sqrt(b[0]**2 + b[1]**2)
        if mag_a * mag_b == 0:
            return 0
        cos_angle = dot / (mag_a * mag_b)
        return math.degrees(math.acos(max(min(cos_angle,1),-1)))
