# 🚗 Autonomous Lane Detection System 

A real-time lane detection system designed to enhance road safety for autonomous vehicles using traditional computer vision techniques like edge detection, lane departure warnings, road condition assessment, and optical flow estimation. This repository provides the full implementation, testing, and evaluation of the system.

## 🌟 Features 

Lane Detection: Detects lane markings using Canny edge detection, Hough transform, and optical flow.
Lane Departure Warning: Alerts the driver when the vehicle deviates from its lane.
Road Condition Assessment: Evaluates road surface quality to adjust the vehicle's behavior.
Motion Estimation: Detects the motion of surrounding objects using optical flow.
Real-time Processing: Designed for real-time autonomous driving applications.
Unit Testing: Comprehensive unit tests to ensure the robustness of each component of the system.

## 🛠️ Technologies Used

Python: Main programming language for implementation.
OpenCV: For computer vision techniques such as edge detection and optical flow estimation.
NumPy: For numerical computing and image processing.
Matplotlib: For visualizing results and testing data.
Unittest: For comprehensive unit testing of all system components.

## 📝 Installation

Clone this repository to your local machine:

`git clone https://github.com/your-username/lanedetection.git`

Navigate to the project directory:

`cd lanedetection`

Install dependencies using pip:

`pip install -r requirements.txt`

## 📊 Results

Feature	Result
Lane Detection	Successfully detected lanes in video data.
Lane Departure Warning	Alerts triggered on lane departure.
Road Condition Detection	Identified road surface conditions accurately.
Motion Estimation	Detected vehicle motion using optical flow.

## 📑 Unit Testing

Unit tests are included to verify the functionality of the core components of the lane detection system. Here are the tests performed:
Preprocessing Test: Validates grayscale conversion, CLAHE, and Gaussian blur steps.
Edge Detection Test: Verifies edge detection using Canny edge detector.
Lane Detection Test: Checks lane identification using Hough transform.
Lane Departure Warning Test: Ensures the warning system triggers correctly.
Road Condition Test: Tests road condition classification.
Motion Estimation Test: Verifies optical flow calculation.
