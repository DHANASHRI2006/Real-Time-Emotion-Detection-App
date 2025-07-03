
import streamlit as st
import cv2
import numpy as np
from fer import FER
import mediapipe as mp
import os


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


@st.cache_resource
def load_emotion_detector():
    return FER(mtcnn=False)

emotion_detector = load_emotion_detector()

def detect_emotion(frame):
    try:

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
     
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]
            
  
            x_coords = [int(l.x * w) for l in landmarks]
            y_coords = [int(l.y * h) for l in landmarks]
            x_min, x_max = max(0, min(x_coords)-20), min(w, max(x_coords)+20)
            y_min, y_max = max(0, min(y_coords)-20), min(h, max(y_coords)+20)
     
            face_roi = frame[y_min:y_max, x_min:x_max]
            
            if face_roi.size > 0:
   
                emotions = emotion_detector.detect_emotions(face_roi)
                if emotions:
                    # Replace 'neutral' with 'normal' in emotions dictionary
                    if 'neutral' in emotions[0]["emotions"]:
                        emotions[0]["emotions"]['normal'] = emotions[0]["emotions"].pop('neutral')
                    
                    dominant = max(emotions[0]["emotions"].items(), key=lambda x: x[1])
        
                    dominant_emotion = 'normal' if dominant[0] == 'neutral' else dominant[0]
                    return dominant_emotion, (x_min, y_min, x_max, y_max)
    
    except Exception as e:
        st.error(f"Detection error: {str(e)}")
    
    return None, None

def main():
    st.title("Real-Time Emotion Detection")
    

    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    

    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Start Camera", disabled=st.session_state.camera_active):
            st.session_state.camera_active = True
    
    with col2:
        if st.button("Stop Camera", disabled=not st.session_state.camera_active):
            st.session_state.camera_active = False
    

    frame_placeholder = st.empty()
    result_placeholder = st.empty()
    

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

     
                dominant_emotion, bbox = detect_emotion(frame)

        
                if bbox:
                    x_min, y_min, x_max, y_max = bbox
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

     
                frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                                      channels="RGB")
                
           
                result_placeholder.empty()
                
    
                if dominant_emotion:
                    result_placeholder.success(f"Detected Emotion: {dominant_emotion.upper()}")
                else:
                    result_placeholder.warning("No face detected")
                
       
                cv2.waitKey(100)
                
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
      
            frame_placeholder.empty()
            result_placeholder.empty()
            
            if st.session_state.camera_active:
                st.session_state.camera_active = False

if __name__ == "__main__":
    main()
