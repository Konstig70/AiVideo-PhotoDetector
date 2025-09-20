import cv2
import mediapipe as mp
import math
import os
import numpy as np
import streamlit as st
from queue import Queue
from threading import Thread


motion_queue = Queue()
motion_scores = []
prev_gray_motion = None

total_frames = 0
# Set up anomaly related variables
anomaly_frames = 0
anomaly_score = 0
anomaly_multiplier = 0.1
previous_frame_anomalous = False
frame_anomaly = False
finger_angle_anomaly_frames = 0
is_finger_anomaly_frame_saved = False    
face_distance_anomaly_frames = 0
arm_length_ratio_anomaly_frames = 0
shoulder_to_shoulder_width_anomaly_frames = 0
        
""""
GeometryMapper class, responsible for geometry mapping and basic human/object-detection
"""
class GeometryMapper:


    #Get all needed models as shorter variables
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_face_mesh = mp.solutions.face_mesh
        

    @staticmethod
    def analyze_video(video_path, display = False, progress_bar = None, progress = 0):
        """
        Analyze a video for hand geometry anomalies (extra fingers, melting shapes).
        Returns a dict with anomaly stats.
        """
        global total_frames
        global anomaly_frames
        global anomaly_score
        global anomaly_multiplier
        global finger_angle_anomaly_frames
        global face_distance_anomaly_frames
        global arm_length_ratio_anomaly_frames
        global shoulder_to_shoulder_width_anomaly_frames
        
        # Open video file from path
        capture = cv2.VideoCapture(video_path)
        # initialize hand tracking model
        hands = GeometryMapper.mp_hands.Hands(
            static_image_mode = False,
            max_num_hands = 2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5 
        )
        #Initialize pose and face tracking
        pose = GeometryMapper.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
        face_mesh = GeometryMapper.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
        
        
        # Set up necessary variables for finger analysis
        finger_angles = []

        #Initialized variables for pose
        pose_shoulder_widths = []

        #Initialize variables for face
        face_distances = []
        

        total_frames_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        #Reset all values        
        total_frames = 0
        anomaly_frames = 0
        anomaly_score = 0
        anomaly_multiplier = 0.1
        finger_angle_anomaly_frames = 0    
        face_distance_anomaly_frames = 0
        arm_length_ratio_anomaly_frames = 0
        shoulder_to_shoulder_width_anomaly_frames = 0


        if progress_bar is not None:
            progress_bar.progress(progress)  # Initial progress
        
        motion_thread = Thread(target=GeometryMapper.motion_worker)
        motion_thread.start()

        
        #loop through each frame 
        while capture.isOpened():
            global frame_anomaly
            frame_anomaly = False
            ret, frame = capture.read()
            if not ret:
                break
            total_frames += 1
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #Get hands and analyze
            is_finger_anomaly_frame_saved = GeometryMapper.process_hands(rgb, hands, finger_angles, frame, is_finger_anomaly_frame_saved)        
            # Process the pose of the frame
            anomaly_multiplier = 0.1
            GeometryMapper.process_pose(rgb,frame,pose, pose_shoulder_widths)
            anomaly_multiplier = 0.1    
            #Process face
            GeometryMapper.process_face(rgb, face_mesh, face_distances)
            print("Anomaly score so far: ", anomaly_score)

            if frame_anomaly:
                # print(f"Frame {total_frames} was anomalous")
                anomaly_frames += 1
            #Calculate motion for frames
            if total_frames % 5 == 0:  # Process every 5th frame for motion to reduce load
                print("Calculating motion for frame:", total_frames)
                motion_queue.put((total_frames, frame.copy()))
           
            #Progress bar continuation
            progress = 25 + int((total_frames / total_frames_count) * 65)  # 25 -> 90
            progress = min(100, max(0, progress))
            progress_bar.progress(progress)
            # Display and save frames with anomalies
            if display and frame_anomaly and total_frames % 20 == 0:
                cv2.imshow("Anomalous Frame", frame)
                # Press 'q' to skip/exit display early
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break
            
        #Clean up
        capture.release()
        cv2.destroyAllWindows()
        only_scores = [score for frame_id, score in motion_scores]
        avg_motion = np.mean(only_scores)
        anomaly_score = anomaly_score / total_frames 
        anomaly_score = anomaly_score - (avg_motion / 100)
        anomaly_justification = ""
        if anomaly_score < 0.025:
            anomaly_justification = "Likely a real video"
        elif 0.025 <= anomaly_score  < 0.050:
            anomaly_justification = "Probably a real video but some minor anomalies were detected"
        elif 0.05 <= anomaly_score < 0.075:
            anomaly_justification = "Most possibly a low quality or highly edited video with some synthetic tampering, some anomalies were detected"
        elif 0.075 <= anomaly_score < 0.1:
            anomaly_justification = "Probably synthetic video, quite many anomalies"
        elif anomaly_score >= 0.1:
            anomaly_justification = "Highly suspicious most likely a synthethic video, many anomalies detected"

        return {
            "total_frames": total_frames,
            "anomaly_frames": anomaly_frames,
            "anatomy_anomaly_rating": f"{anomaly_score:.3f}   which equates to : {anomaly_justification}",
            "finger_anomaly_frames": finger_angle_anomaly_frames,
            "arm_length_ratio_anomaly_frames ": arm_length_ratio_anomaly_frames,
            "shoulder_to_shoulder_width_anomaly_frames": shoulder_to_shoulder_width_anomaly_frames,
            "face_distance_anomaly_frames": face_distance_anomaly_frames,
            "motion_score": avg_motion
        }
    
    def motion_worker():
        global prev_gray_motion
        while True:
            item = motion_queue.get()
            if item is None:  # sentinel to stop
                break
            frame_id, frame = item
            prev_gray_motion, motion_score = GeometryMapper.calculate_motion(frame, prev_gray_motion)
            motion_scores.append((frame_id, motion_score))
            motion_queue.task_done()

    def calculate_motion(frame, prev_gray):
        scale = 0.5
        # Convert frame to gray and resize
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small_gray = cv2.resize(gray, (0,0), fx=scale, fy=scale)
        if prev_gray is None:
            prev_gray = small_gray
            return prev_gray, 0
        else:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, small_gray, None,
                                                pyr_scale=0.5, levels=3, winsize=15,
                                                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            magnitude, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
            motion_score = np.mean(magnitude)
            prev_gray = small_gray
            return prev_gray, motion_score

    def process_face(rgb, face_mesh, face_distances):
        global previous_frame_anomalous
        global anomaly_multiplier
        global anomaly_score
        global frame_anomaly
        global face_distance_anomaly_frames
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
                # get all landmarks
                face_landmarks = results.multi_face_landmarks[0].landmark
                # currently we check 33 and 263 = eyes, 61 and 291 = mouth corners 159 and 386 = cheeks  
                symmetric_pairs = [(33, 263), (61, 291), (159, 386)]
                distances = []
                for left_idx, right_idx in symmetric_pairs:
                    left = face_landmarks[left_idx]
                    right = face_landmarks[right_idx]
                    # Get the symmetry distance low score = close, high score = far = likely synthetical 
                    symmetry_distance = GeometryMapper.symmetry_distance(left, right) 
                    # Append distance
                    distances.append(symmetry_distance)
                #Append this frames distances to all frames
                face_distances.append(distances)
                np_faces_distances = np.array(face_distances)
                face_distance_averages = np.mean(np_faces_distances, axis=0)
                #Get deviation
                deviation = np.abs(np.array(distances) - face_distance_averages)
                distances_std = np.std(np_faces_distances, axis=0)
                if np.any(deviation > 2 * distances_std): #once again same heuristic
                    global previous_frame_anomalous
                    #Check previous frame for anomality
                    if (previous_frame_anomalous):
                        anomaly_multiplier += min(anomaly_multiplier + 0.3, 3.0)
                    #Raise score times multiplier, also face anomalies are weighted less than finger anomalies so we increment the score less
                    
                    anomaly_score += 1 * anomaly_multiplier * 0.7
                    previous_frame_anomalous = True
                    frame_anomaly = True
                    face_distance_anomaly_frames += 1
                    print(f"Face anomaly detected at frame: {total_frames}!")
                else:
                    previous_frame_anomalous = False
                    anomaly_multiplier = 0.1


    def process_pose(rgb, frame,pose, pose_shoulder_widths):
        global previous_frame_anomalous
        global anomaly_multiplier
        global anomaly_score
        global frame_anomaly
        global arm_length_ratio_anomaly_frames
        global shoulder_to_shoulder_width_anomaly_frames
        results = pose.process(rgb)
        if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Limb ratios from current frame
                ratios_current_frame = GeometryMapper.getLimbRatios(landmarks)
                assymetry = abs(ratios_current_frame[0] - ratios_current_frame[1])
                
                # normally arms are symmetrical 1:1 so if assymetry is higher than 0.35 that is an anomaly
                if assymetry > 0.45:    
                    print(f"Arm assymetry anomaly detected at frame : {total_frames}!")   
                    #Check previous frame for anomality
                    if (previous_frame_anomalous):
                        #Grow multiplier
                        anomaly_multiplier += min(anomaly_multiplier + 0.3, 3.0)
                    #Raise score times multiplier
                    anomaly_score += 1 * anomaly_multiplier * 0.7
                    previous_frame_anomalous = True
                    global frame_anomaly
                    frame_anomaly = True
                    arm_length_ratio_anomaly_frames += 1
                else:
                    #Frame was not anomalous so we reset multiplier 
                    anomaly_multiplier = 0.1
                    previous_frame_anomalous = False
                # check shoulder to shoulder width
                pose_shoulder_widths.append(ratios_current_frame[2])
                np_shoulder_widths = np.array(pose_shoulder_widths)
                np_shoulder_average = np.mean(np_shoulder_widths, axis=0)
                deviation = np.abs(ratios_current_frame[2] - np_shoulder_average)
                std_lengths = np.std(np_shoulder_widths, axis=0)
                if deviation > 3 * std_lengths:
                    print(f"shoulder to shoulder width anomaly detected at frame : {total_frames}!")   
                    #Check previous frame for anomality
                    if (previous_frame_anomalous):
                        #Grow multiplier
                        anomaly_multiplier += min(anomaly_multiplier + 0.3, 3.0)
                    #Raise score times multiplier
                    anomaly_score += 1 * anomaly_multiplier
                    previous_frame_anomalous = True
                    frame_anomaly = True
                    shoulder_to_shoulder_width_anomaly_frames += 1
                else:
                    #Frame was not anomalous so we reset multiplier 
                    anomaly_multiplier = 0.1
                    previous_frame_anomalous = False
                GeometryMapper.mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)


    def process_hands(rgb, hands, finger_angles, frame, is_finger_anomaly_frame_saved):
        global previous_frame_anomalous
        global anomaly_multiplier
        global anomaly_score
        global frame_anomaly
        global finger_angle_anomaly_frames
        results = hands.process(rgb)
            #process hands
        if results.multi_hand_landmarks:
                #iterate over each spotted hand
                for hand_landmarks in results.multi_hand_landmarks:
                    #Get finger angles from current frame
                    frame_angles, finger_count = GeometryMapper.finger_angles(hand_landmarks)
                    finger_angles.append(frame_angles)
                    np_angles = np.array(finger_angles)
                    # Get the averages for each fingers angle
                    finger_angle_averages = np.mean(np_angles, axis=0)
                    # Get deviaton
                    deviation = np.abs(np.array(frame_angles) - finger_angle_averages)
                    # If deviaton too large we call an anomaly
                    std_angles = np.std(np_angles, axis=0)
                    if (np.any(deviation > 2 * std_angles)): # 2σ rule since it adapts to baseline
                        frame_anomaly = True
                        # Check if previous frame was anomalous and grow multiplier if it was
                        if (previous_frame_anomalous):
                            anomaly_multiplier = min(anomaly_multiplier + 0.3, 3.0)
                        previous_frame_anomalous = True
                        #Grow anomaly_score
                        anomaly_score += 1 * anomaly_multiplier * 1.4
                        #Log the anomaly and save frame
                        print(f"Finger curl anomaly detected at frame {total_frames}!")
                        finger_angle_anomaly_frames += 1
                        if not is_finger_anomaly_frame_saved:
                            save_path = os.path.join(os.getcwd(), f"finger_anomaly_frame_.png")
                            cv2.imwrite(save_path, frame)
                        print("frame saved at: ", save_path) 
                        return True
                    else :
                        #Frame was not anomalous so multiplier zeroed
                        anomaly_multiplier = 0
                        previous_frame_anomalous = False
                        return False
                

    def getLimbRatios(landmarks):
        left_upper_arm = GeometryMapper.segment_length(landmarks[11], landmarks[13])  # shoulder->elbow
        left_lower_arm = GeometryMapper.segment_length(landmarks[13], landmarks[15])  # elbow->wrist
        right_upper_arm = GeometryMapper.segment_length(landmarks[12], landmarks[14])  # shoulder->elbow
        right_lower_arm = GeometryMapper.segment_length(landmarks[14], landmarks[16])  # elbow->wrist
        shoulder_width = GeometryMapper.segment_length(landmarks[11], landmarks[12]) # right shoulder <-> left shoulder

        # normally human limbs have a 1:1 ratio (some deviation)
        left_arm_ratio = left_upper_arm / left_lower_arm if left_lower_arm != 0 else 0
        right_arm_ratio = right_upper_arm / right_lower_arm if right_lower_arm != 0 else 0

        
        return [left_arm_ratio, right_arm_ratio, shoulder_width]
                

    def symmetry_distance(p_left, p_right):
        """Returns how far the points are from perfect horizontal symmetry"""
        return abs(p_left.x - (1 - p_right.x))  # normalized coordinates

    def segment_length(p1, p2):
        """Distance between two normalized landmarks"""
        return math.sqrt(
            (p1.x - p2.x)**2 +
            (p1.y - p2.y)**2 +
            (p1.z - p2.z)**2
        )


    def finger_angles(hand_landmarks):
        #Gets fingers angles and also count of fingers
        # these indecies are the landmark indecies for different fingers 
        fingers = [
        [1, 2, 3, 4],   # Thumb
        [5, 6, 7, 8],   # Index
        [9, 10, 11, 12],# Middle
        [13, 14, 15, 16],# Ring
        [17, 18, 19, 20]# Pinky
    ]
        angles_frame = []
        count = 0
        for f in fingers:
            #Get finger's landamarks
            mcp = hand_landmarks.landmark[f[0]]
            pip = hand_landmarks.landmark[f[1]]
            dip = hand_landmarks.landmark[f[2]]
            tip = hand_landmarks.landmark[f[3]]
            #Get fingers length
            length = distance(mcp, tip)
            angle1 = angle_between(mcp, pip, dip)
            angle2 = angle_between(pip, dip, tip)
            angles_frame.extend([angle1, angle2])

            if 40 <= angle1 <= 180 and 40 <= angle2 <= 180 and length >= 0.05:
                count += 1
            
        return angles_frame, count

def distance(lm1, lm2):
    """
    Compute the Euclidean distance between two landmarks.
    """
    return math.sqrt(
        (lm1.x - lm2.x)**2 +
        (lm1.y - lm2.y)**2 +
        (lm1.z - lm2.z)**2
    )

        
def angle_between(p1, p2, p3):
    """Angle at p2 formed by p1-p2-p3"""
    import math
    a = (p1.x - p2.x, p1.y - p2.y)
    b = (p3.x - p2.x, p3.y - p2.y)
    dot = a[0]*b[0] + a[1]*b[1]
    mag_a = math.sqrt(a[0]**2 + a[1]**2)
    mag_b = math.sqrt(b[0]**2 + b[1]**2)
    if mag_a * mag_b == 0:
        return 0
    cos_angle = dot / (mag_a * mag_b)
    return math.degrees(math.acos(max(min(cos_angle,1),-1)))


def main():
    current_dir = os.getcwd()
    # Go one level up with ".." and then into data/shorterTest.mp4
    file_path = os.path.join(current_dir, "..", "data", "testingReal.mp4")
    # Normalize the path to get an absolute path
    file_path = os.path.abspath(file_path)

    result = GeometryMapper.analyze_video(file_path, False, None)
    print(result)

if __name__ == "__main__":
    main()
