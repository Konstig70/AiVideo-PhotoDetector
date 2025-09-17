import cv2
import mediapipe as mp
import math
import os
import numpy as np

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
    def analyze_video(video_path, display = False):
        """
        Analyze a video for hand geometry anomalies (extra fingers, melting shapes).
        Returns a dict with anomaly stats.
        """
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
        
        total_frames = 0
        #Set up anomaly related variables
        anomaly_frames = 0
        anomaly_score = 0
        anomaly_multiplier = 0.1
        previous_frame_anomalous = False
        
        # Set up necessary variables for finger analysis
        finger_angles = []
        finger_angle_averages = 0
        finger_anomaly_frames = 0

        #Initialized variables for pose
        pose_ratios = []
        pose_ratios_averages = 0
        pose_ratio_anomaly_frames = 0

        #Initialize variables for face
        face_distances = []
        face_distance_averages = 0
        face_distance_anomaly_frames = 0

        #loop through each frame 
        while capture.isOpened():
            frame_anomaly = False
            finger_anomaly = False
            pose_anomaly = False
            face_anomaly= False
            ret, frame = capture.read()
            if not ret:
                break
            total_frames += 1

            #Get hands and analyze
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = hands.process(rgb)
            
            if results.multi_hand_landmarks:

                #iterate over each spotted hand
                for hand_landmarks in results.multi_hand_landmarks:
                    #Get finger angles from current frame
                    frame_angles = GeometryMapper.finger_angles(hand_landmarks)
                    finger_angles.append(frame_angles)
                    np_angles = np.array(finger_angles)
                    # Get the averages for each fingers angle
                    finger_angle_averages = np.mean(np_angles, axis=0)
                    # Get deviaton
                    deviation = np.abs(np.array(frame_angles) - finger_angle_averages)
                    # If deviaton too large we call an anomaly
                    std_angles = np.std(np_angles, axis=0)
                    if (np.any(deviation > 2 * std_angles)): # 2σ rule since it adapts to baseline
                        finger_anomaly = True
                        frame_anomaly = True
                        # Check if previous frame was anomalous and grow multiplier if it was
                        if (previous_frame_anomalous):
                            anomaly_multiplier = min(anomaly_multiplier + 0.3, 3.0)
                        previous_frame_anomalous = True
                        #Grow anomaly_score
                        anomaly_score += 1 * anomaly_multiplier
                        #Log the anomaly and save frame
                        print(f"Finger anomaly detected at frame {total_frames}!")
                        GeometryMapper.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, GeometryMapper.mp_hands.HAND_CONNECTIONS
                        )
                        finger_anomaly_frames += 1
                        """save_path = os.path.join(os.getcwd(), f"anomaly_frame_{total_frames}.png")
                        cv2.imwrite(save_path, frame)
                        print("frame saved at: ", save_path)"""
                    else :
                        #Frame was not anomalous so multiplier zeroed
                        anomaly_multiplier = 0
                        previous_frame_anomalous = False
                    
                        
            # Process the pose of the frame
            anomaly_multiplier = 0.1
            results = pose.process(rgb)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Limb ratios from current frame
                ratios_current_frame = GeometryMapper.getLimbRatios(landmarks)
                pose_ratios.append(ratios_current_frame)
                np_ratios = np.array(pose_ratios)
                pose_ratios_averages = np.mean(np_ratios, axis=0)
                
                deviation = np.abs(np.array(ratios_current_frame) - pose_ratios_averages)
                pose_ratios_std = np.std(np_ratios, axis=0)
                if np.any(deviation > 2 * pose_ratios_std): #2σ rule since it adapts to baseline
                    print(f"Limb ratio anomaly detected at frame : {total_frames}!")   
                    #Check previous frame for anomality
                    if (previous_frame_anomalous):
                        #Grow multiplier
                        anomaly_multiplier += min(anomaly_multiplier + 0.3, 3.0)
                    #Raise score times multiplier
                    anomaly_score += 1 * anomaly_multiplier
                    previous_frame_anomalous = True
                    frame_anomaly = True
                    pose_ratio_anomaly_frames += 1
                else:
                    #Frame was not anomalous so we reset multiplier 
                    anomaly_multiplier = 0.1
                    previous_frame_anomalous = False
                # Draw landmarks if desired
                GeometryMapper.mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

                
            #Process face
            results = face_mesh.process(rgb)
            anomaly_multiplier = 0.1
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
                #Append this frames distances to all
                face_distances.append(distances)
                np_faces_distances = np.array(face_distances)
                face_distance_averages = np.mean(np_faces_distances, axis=0)
                #Get deviation
                deviation = np.abs(np.array(distances) - face_distance_averages)
                distances_std = np.std(np_faces_distances, axis=0)
                if np.any(deviation > 2 * distances_std): #once again same heuristic
                    if (previous_frame_anomalous):
                        anomaly_multiplier += min(anomaly_multiplier + 0.3, 3.0)
                    #Raise score times multiplier
                    anomaly_score += 1 * anomaly_multiplier
                    previous_frame_anomalous = True
                    frame_anomaly = True
                    face_distance_anomaly_frames += 1
                    print(f"Face anomaly detected at frame: {total_frames}!")
                else:
                    previous_frame_anomalous = False
                    anomaly_multiplier = 0.1
                if frame_anomaly:
                    anomaly_frames += 1
            # Display and save frames with anomalies
            if display and frame_anomaly and total_frames % 20 == 0:
                cv2.imshow("Anomalous Frame", frame)
                # Press 'q' to skip/exit display early
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break
            
            #Clean up
        capture.release()
        cv2.destroyAllWindows()

        return {
            "total_frames": total_frames,
            "anomaly_frames": anomaly_frames,
            "anomaly_rating": anomaly_score / total_frames,
            "finger_anomaly_frames": finger_anomaly_frames,
            "pose_ratio_anomaly_frames": pose_ratio_anomaly_frames,
            "face_distance_anomaly_frames": face_distance_anomaly_frames
        }

    def getLimbRatios(landmarks):
        left_upper_arm = GeometryMapper.segment_length(landmarks[11], landmarks[13])  # shoulder->elbow
        left_lower_arm = GeometryMapper.segment_length(landmarks[13], landmarks[15])  # elbow->wrist
        right_upper_arm = GeometryMapper.segment_length(landmarks[12], landmarks[14])  # shoulder->elbow
        right_lower_arm = GeometryMapper.segment_length(landmarks[14], landmarks[16])  # elbow->wrist
                
        # normally human limbs have a 1:1 ratio (some deviation)
        left_arm_ratio = left_upper_arm / left_lower_arm if left_lower_arm != 0 else 0
        right_arm_ratio = right_upper_arm / right_lower_arm if right_lower_arm != 0 else 0
        return [left_arm_ratio, right_arm_ratio]
                

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
        # these indecies are the landmark indecies for different fingers 
        fingers = [
        [1, 2, 3, 4],   # Thumb
        [5, 6, 7, 8],   # Index
        [9, 10, 11, 12],# Middle
        [13, 14, 15, 16],# Ring
        [17, 18, 19, 20]# Pinky
    ]
        angles_frame = []
        for f in fingers:
            mcp = hand_landmarks.landmark[f[0]]
            pip = hand_landmarks.landmark[f[1]]
            dip = hand_landmarks.landmark[f[2]]
            tip = hand_landmarks.landmark[f[3]]

            angle1 = angle_between(mcp, pip, dip)
            angle2 = angle_between(pip, dip, tip)
            angles_frame.extend([angle1, angle2])
            
        return angles_frame

        
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
    file_path = os.path.join(current_dir, "..", "data", "Test2.mp4")
    # Normalize the path to get an absolute path
    file_path = os.path.abspath(file_path)

    result = GeometryMapper.analyze_video(file_path, False)
    print(result)

if __name__ == "__main__":
    main()
