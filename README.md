# 🎥 Intrusion-Detection-Using-Yolov8

![W1](https://github.com/anandjohnbaby/IDS-Skeleton-tracking-using-Yolov8/assets/108878068/eb5bad53-4ef8-4410-a0d3-b7f948998b75)
# 📝 Description
- Utilizes **YOLOv8** for object detection and skeleton tracking, enabling accurate extraction of keypoints from video frames.
- Employs a **Convolutional Neural Network** architecture for action classification, trained on extracted keypoints from videos.
- During inference, the model analyzes uploaded videos to detect intruders, leveraging the extracted keypoints for precise action classification.
- Real-time annotated frames, showcasing detected keypoints, are displayed through **Streamlit's** frontend
- In the event of an intruder detection, the system promptly triggers SMS notifications via "**Twilio**, ensuring timely alerts to designated recipients.

# 🎯 Inference demo
| No Intrusion | No Intrusion |
|---------|---------|
| ![Image 1](https://github.com/anandjohnbaby/IDS-Skeleton-tracking-using-Yolov8/assets/108878068/d7ff29ea-0151-49b8-b30e-bbd1efd4a322) | ![Image 2](https://github.com/anandjohnbaby/IDS-Skeleton-tracking-using-Yolov8/assets/108878068/f55e7da2-1f6f-4e14-853b-fe557772eb0b) |

| Intrusion | Intrusion |
|---------|---------|
| ![Image 3](https://github.com/anandjohnbaby/IDS-Skeleton-tracking-using-Yolov8/assets/108878068/b5c180fb-345f-48ea-8795-f9a13886893b) | ![Image 4](https://github.com/anandjohnbaby/IDS-Skeleton-tracking-using-Yolov8/assets/108878068/552e3d75-3614-4d5a-b0fc-18cced94ba91) |


# 🖥️ Installation
### 🛠️ Requirements
- Python 3.9
- Tensorflow 2.16+
- Ultralytics 8.2.2+
- Streamlit 1.33.0
- Windows

# 📊 Dataset
I have created a custom dataset by collecting videos from different sources. This dataset comprises 48 intrusion videos and 32 normal videos. Using YOLOv8 and OpenCv, we extracted keypoints from each frame of the videos and stored them in a CSV file. Additionally, preprocessing steps were applied, including the removal of NaN values and frames with insufficient coordinates. Feature selection/extraction techniques were also employed to enhance the dataset's quality and relevance for further analysis.

# 📧 Contact
Email : anandjohnbabyv4@gmail.com
