import cv2
import mediapipe as mp
import math
import os

""""
GeometryMapper class, responsible for geometry mapping and basic human/object-detection
"""
class GeometryMapper:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

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

        
        total_frames = 0
        #i.e frames with anomalies
        anomaly_frames = 0
        #loop through each frame 
        while capture.isOpened():
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
                    #Get fingers
                    finger_count = GeometryMapper.count_fingers(hand_landmarks)
                    print(f"fingers: {finger_count} in frame {total_frames}")
                    if finger_count != 5:  # anomaly we want only 5 fingers per hand
                        anomaly_frames += 1
                        frame_anomaly = True
                    GeometryMapper.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, GeometryMapper.mp_hands.HAND_CONNECTIONS
                    )
                if display and frame_anomaly and total_frames % 20 == 0:
                    cv2.imshow("Anomalous Frame", frame)
                    # Press 'q' to skip/exit display early
                    if cv2.waitKey(0) & 0xFF == ord('q'):
                        break
                if total_frames % 20 == 0:
                    save_path = os.path.join(os.getcwd(), f"anomaly_frame_{total_frames}.png")
                    cv2.imwrite(save_path, frame)
                    print("frame saved at: ", save_path)
            #Clean up
        capture.release()
        cv2.destroyAllWindows()

        return {
            "total_frames": total_frames,
            "anomaly_frames": anomaly_frames,
            "anomaly_rating": anomaly_frames / total_frames if total_frames > 0 else 0
        }

    @staticmethod
    def count_fingers(hand_landmarks):
        # these indecies are the landmark indecies for different fingers 
        fingers = [
        [1, 2, 3, 4],   # Thumb
        [5, 6, 7, 8],   # Index
        [9, 10, 11, 12],# Middle
        [13, 14, 15, 16],# Ring
        [17, 18, 19, 20]# Pinky
    ]
        finger_count = 0
        for f in fingers:
            mcp = hand_landmarks.landmark[f[0]] # mcp is the lowest finger landmark
            tip = hand_landmarks.landmark[f[3]] # tip is the highest i.e the tip of the finger
            dist = distance(mcp, tip) 
            if dist > 0.1:  # if the distance between these landmarks is really small, then we can be certain that the finger doesnt exist 
                # we then calculate the angle to check that the finger is not distorted
                # first we need the oher landmarks
                pip = hand_landmarks.landmark[f[1]] 
                dip = hand_landmarks.landmark[f[2]]
                #now calculate the angles
                angle1 = angle_between(mcp, pip, dip)
                angle2 = angle_between(pip, dip, tip)
                # Fingers are typically beteen angles 40-180 so if both angles are not between these values we can be certain that there is no finger, or finger is distorted 
                if (40 <= angle1 <= 180 and 40 <= angle2 <= 180):
                    finger_count += 1


        return finger_count

        
def distance(lm1, lm2):
    """Distance between two landmarks """
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
    file_path = os.path.join(current_dir, "..", "data", "shorterTest.mp4")
    # Normalize the path to get an absolute path
    file_path = os.path.abspath(file_path)

    result = GeometryMapper.analyze_video(file_path, False)
    print(result)

if __name__ == "__main__":
    main()
