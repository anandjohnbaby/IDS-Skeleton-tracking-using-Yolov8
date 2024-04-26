from ultralytics import YOLO
import streamlit as st
import tensorflow as tf
import cv2
import pandas as pd
from twilio.rest import Client
import numpy as np
import tempfile
from ultralytics.utils.plotting import Annotator

# Load the model
model = tf.keras.models.load_model("Intrusion_detection_model.h5")

# Twilio credentials
account_sid = 'AC921ab6089d395c7954ea46c25850741a'
auth_token = '4f49cdad7a8392115ffd1fa1bd4c747c'
twilio_phone_number = '+12512209339'
recipient_phone_number = '+917592972157'

# Initialize Twilio client
client = Client(account_sid, auth_token)

@st.cache_data
def send_sms(message):
    try:
        client.messages.create(to=recipient_phone_number, 
                               from_=twilio_phone_number, 
                               body=message)
        return True  # Indicate successful message sending
    except Exception as e:
        st.write(f"Error sending SMS: {e}")
        return False  # Indicate failure
    

# Load YOLO model
yolo_model = YOLO('yolov8n-pose.pt')


def detect_intruder(keypoints_list, model, feature_names):

    # Convert list of dictionaries to DataFrame
    keypoints_df = pd.DataFrame(keypoints_list)
    
    # Reorder columns to match those used during training
    keypoints_df = keypoints_df[feature_names]
    
    # Predict actions using the model
    predictions = model.predict(keypoints_df)

    predictions_list = list(predictions)
    
    proportion_of_ones = predictions_list.count(1) / len(predictions_list)

    # Check if intruder is detected
    if proportion_of_ones >= 0.3:
        st.write("🚨 Intruder detected!")
        sms_sent = send_sms("Intruder detected at your location!")
        if sms_sent:
            st.write("📩 SMS sent successfully!")
        else:
            st.write("❌ Failed to send SMS.")
    else:
        st.write("✅ No intruder detected")
    
    return predictions


def extract_keypoints(video_bytes):
    temp_file_path = None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(video_bytes)
        temp_file_path = temp_file.name

    cap = cv2.VideoCapture(temp_file_path)

    keypoints_list = []

    # Create a placeholder for displaying the processed frames
    placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Predict with the model
        results = yolo_model(frame, save=False)
        annotated_frame = results[0].plot()

        # Check if results is empty
        if len(results) == 0:
            continue

         # Extract keypoints 
        keypoints = results[0].keypoints.xyn[0].cpu().numpy()

        # Check the number of keypoints detected
        if len(keypoints) < 8:  # Adjust the threshold as per your requirement
            continue

        # Create dictionary to store coordinates
        coordinates_dict = {}
        
        # Define keypoints of interest with corresponding indices
        keypoints_indices = {
            5: ('LShoulder', 'LShoulder'),
            6: ('RShoulder', 'RShoulder'),
            7: ('LElbow', 'LElbow'),
            8: ('RElbow', 'RElbow'),
            3: ('LWrist', 'LWrist'),
            4: ('RWrist', 'RWrist'),
            11: ('LHip', 'LHip'),
            12: ('RHip', 'RHip'),
            13: ('LKnee', 'LKnee'),
            14: ('Rknee', 'Rknee'),
            9: ('LAnkle', 'LAnkle'),
            10: ('RAnkle', 'RAnkle')
        }

        # Extract coordinates for keypoints of interest
        for keypoint_index, (key_x, key_y) in keypoints_indices.items():
            if keypoint_index < keypoints.shape[0]:
                x, y = keypoints[keypoint_index]
                coordinates_dict[key_x + '_X'] = x
                coordinates_dict[key_y + '_Y'] = y
            else:
                coordinates_dict[key_x + '_X'] = None
                coordinates_dict[key_y + '_Y'] = None

        # Append coordinates dictionary to list
        keypoints_list.append(coordinates_dict)
        

        # Display frame with keypoints and bounding boxes
        for r in results:
            annotator = Annotator(frame)
        
            # Draw bounding boxes
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                c = box.cls
                annotator.box_label(b, yolo_model.names[int(c)])

        # Display the video in streamlit app
        placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return keypoints_list


# Streamlit UI
st.title("IntrusionGuard: A Deep Learning-Powered Automated Intruder Detection System")
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "webm"])

if uploaded_file is not None:

    # Open the uploaded video file
    video_bytes = uploaded_file.read()
    
    # Analyze the video for intruders
    keypoints_list = extract_keypoints(video_bytes)
    if keypoints_list is not None:
        feature_names = ['LShoulder_X', 'LShoulder_Y', 'RShoulder_X', 'RShoulder_Y', 
                     'LElbow_X', 'LElbow_Y', 'RElbow_X', 'RElbow_Y', 
                     'LWrist_X', 'LWrist_Y', 'RWrist_X', 'RWrist_Y',
                     'LHip_X', 'LHip_Y', 'RHip_X', 'RHip_Y', 
                     'LKnee_X', 'LKnee_Y', 'Rknee_X', 'Rknee_Y', 
                     'LAnkle_X', 'LAnkle_Y', 'RAnkle_X', 'RAnkle_Y']
        result = detect_intruder(keypoints_list, model, feature_names)

