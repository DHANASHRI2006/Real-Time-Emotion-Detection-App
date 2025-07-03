# import streamlit as st
# import cv2
# import numpy as np
# from fer import FER
# import mediapipe as mp
# from PIL import Image
# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disables oneDNN messages
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduces TensorFlow logging
# # Initialize MediaPipe Face Mesh
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(
#     static_image_mode=False,
#     max_num_faces=1,
#     refine_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5,
# )

# # Initialize FER (Facial Emotion Recognition)
# emotion_detector = FER(mtcnn=False)  # Disable MTCNN for faster CPU processing

# def detect_emotion(frame):
#     # Convert BGR to RGB (MediaPipe requires RGB)
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
#     # Detect faces using MediaPipe
#     results = face_mesh.process(rgb_frame)
    
#     if results.multi_face_landmarks:
#         # Get face bounding box (approximate)
#         landmarks = results.multi_face_landmarks[0].landmark
#         h, w, _ = frame.shape
        
#         # Extract approximate face region
#         x_min = int(min(l.x * w for l in landmarks))
#         x_max = int(max(l.x * w for l in landmarks))
#         y_min = int(min(l.y * h for l in landmarks))
#         y_max = int(max(l.y * h for l in landmarks))
        
#         # Expand bounding box slightly
#         margin = 20
#         x_min = max(0, x_min - margin)
#         x_max = min(w, x_max + margin)
#         y_min = max(0, y_min - margin)
#         y_max = min(h, y_max + margin)
        
#         # Crop face region
#         face_roi = frame[y_min:y_max, x_min:x_max]
        
#         if face_roi.size > 0:
#             # Detect emotion using FER
#             emotions = emotion_detector.detect_emotions(face_roi)
            
#             if emotions:
#                 dominant_emotion = max(
#                     emotions[0]["emotions"].items(),
#                     key=lambda x: x[1],
#                 )
#                 return dominant_emotion[0], emotions[0]["emotions"], (x_min, y_min, x_max, y_max)
    
#     return None, None, None

# def main():
#     st.title("Real-Time Emotion Detection (MediaPipe + FER)")
#     st.markdown("🔹 Detects: `angry`, `disgust`, `fear`, `happy`, `sad`, `surprise`, `neutral`")

#     run = st.checkbox("Start Webcam")
#     FRAME_WINDOW = st.image([])
#     emotion_placeholder = st.empty()
#     prob_placeholder = st.empty()

#     cap = cv2.VideoCapture(0)

#     while run:
#         ret, frame = cap.read()
#         if not ret:
#             st.error("Failed to capture frame")
#             break

#         # Detect emotion
#         dominant_emotion, emotion_probs, bbox = detect_emotion(frame)

#         # Draw bounding box (if face detected)
#         if bbox:
#             x_min, y_min, x_max, y_max = bbox
#             cv2.rectangle(
#                 frame,
#                 (x_min, y_min),
#                 (x_max, y_max),
#                 (0, 255, 0),
#                 2,
#             )

#         # Convert to RGB for Streamlit
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         FRAME_WINDOW.image(frame_rgb, channels="RGB")

#         # Display results
#         if dominant_emotion:
#             emotion_placeholder.success(f"Dominant Emotion: **{dominant_emotion.upper()}**")
            
#             if emotion_probs:
#                 prob_text = " | ".join(
#                     f"{k}: {v:.2f}" for k, v in emotion_probs.items()
#                 )
#                 prob_placeholder.code(prob_text)
#         else:
#             emotion_placeholder.warning("No face detected")

#     cap.release()

# if __name__ == "__main__":
#     main()
import streamlit as st
import cv2
import numpy as np
from fer import FER
import mediapipe as mp
import os

# Configure environment for optimal performance
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Initialize FER with caching
@st.cache_resource
def load_emotion_detector():
    return FER(mtcnn=False)

emotion_detector = load_emotion_detector()

def detect_emotion(frame):
    try:
        # Convert BGR to RGB (MediaPipe requires RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces using MediaPipe
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]
            
            # Get face bounding box with buffer
            x_coords = [int(l.x * w) for l in landmarks]
            y_coords = [int(l.y * h) for l in landmarks]
            x_min, x_max = max(0, min(x_coords)-20), min(w, max(x_coords)+20)
            y_min, y_max = max(0, min(y_coords)-20), min(h, max(y_coords)+20)
            
            # Extract face ROI
            face_roi = frame[y_min:y_max, x_min:x_max]
            
            if face_roi.size > 0:
                # Detect emotions
                emotions = emotion_detector.detect_emotions(face_roi)
                if emotions:
                    # Replace 'neutral' with 'normal' in emotions dictionary
                    if 'neutral' in emotions[0]["emotions"]:
                        emotions[0]["emotions"]['normal'] = emotions[0]["emotions"].pop('neutral')
                    
                    dominant = max(emotions[0]["emotions"].items(), key=lambda x: x[1])
                    # Replace 'neutral' with 'normal' in dominant emotion if needed
                    dominant_emotion = 'normal' if dominant[0] == 'neutral' else dominant[0]
                    return dominant_emotion, (x_min, y_min, x_max, y_max)
    
    except Exception as e:
        st.error(f"Detection error: {str(e)}")
    
    return None, None

def main():
    st.title("Real-Time Emotion Detection")
    
    # Initialize session state
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    
    # Create columns for buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Start Camera", disabled=st.session_state.camera_active):
            st.session_state.camera_active = True
    
    with col2:
        if st.button("Stop Camera", disabled=not st.session_state.camera_active):
            st.session_state.camera_active = False
    
    # Display area
    frame_placeholder = st.empty()
    result_placeholder = st.empty()
    
    # Camera processing
    if st.session_state.camera_active:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        try:
            while st.session_state.camera_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame")
                    break

                # Process frame
                dominant_emotion, bbox = detect_emotion(frame)

                # Visualization
                if bbox:
                    x_min, y_min, x_max, y_max = bbox
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Display camera feed
                frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                                      channels="RGB")
                
                # Clear previous results
                result_placeholder.empty()
                
                # Show current results
                if dominant_emotion:
                    result_placeholder.success(f"Detected Emotion: {dominant_emotion.upper()}")
                else:
                    result_placeholder.warning("No face detected")
                
                # Small delay to prevent high CPU usage
                cv2.waitKey(100)
                
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Clear displays when camera stops
            frame_placeholder.empty()
            result_placeholder.empty()
            
            if st.session_state.camera_active:
                st.session_state.camera_active = False

if __name__ == "__main__":
    main()