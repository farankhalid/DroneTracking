
# Object Detection
## Overview
This code performs real-time object detection on a video stream using the YOLOv4 model. It utilizes OpenCV for video input/output and inference, along with the imutils library for FPS (Frames Per Second) calculation. The YOLOv4 model is pretrained on the COCO dataset for detecting a variety of objects.
## Prerequisites

Ensure you have the following prerequisites before running the code:

- OpenCV (cv2) and imutils libraries installed in your Python environment.
- YOLOv4 model weights (yolov4.weights), configuration file (yolov4.cfg), and class names file (coco.names) available in the input directory.
- CUDA-enabled GPU with CUDA Toolkit and cuDNN installed for faster inference (optional but recommended).
- RTSP URL of the video stream for object detection.
## Setup Instructions
- Clone this repository to your local machine.
- Place the YOLOv4 model files (yolov4.weights, yolov4.cfg, coco.names) in the input directory.
- Update the RTSP URL in the code to point to your video stream.
- Ensure CUDA Toolkit and cuDNN are correctly installed if using GPU acceleration.
- Run the script and observe real-time object detection on the specified video stream.
## Usage
### Confidence Threshold
Adjust CONFIDENCE_THRESHOLD to filter out detections with low confidence scores.
### NMS Threshold
Modify NMS_THRESHOLD to control the non-maximum suppression threshold for eliminating overlapping detections.
### Colors
Update COLORS if you wish to change the bounding box colors for different classes.

## Notes
- The script continuously reads frames from the video stream and performs object detection on each frame.
- Detected objects are displayed with bounding boxes and labels in real-time.
- The FPS (Frames Per Second) of the detection process is calculated and displayed on the output frame.
## Performance
- The performance of the object detection depends on various factors such as hardware resources, input video resolution, model complexity, and GPU acceleration.
- Utilizing a CUDA-enabled GPU can significantly improve the inference speed, especially for real-time applications.
## Authors

- [@farankhalid](https://www.github.com/farankhalid)
- [@aromabear](https://github.com/aromabear)
- [@Tabed23](https://github.com/Tabed23)



