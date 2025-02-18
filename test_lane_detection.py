import unittest
import cv2
import numpy as np
import os
from lane_detection import LaneDetection

class TestLaneDetection(unittest.TestCase):
    def setUp(self):
        """ Set up the initial test environment """
        self.lane_detector = LaneDetection(threshold=100, hough_threshold=50)
        self.test_image = cv2.imread('images/test_image.jpg')  # Replace with a valid image path
        self.test_video_path = 'video.mp4'  # Replace with your video file path
        self.output_video_path = 'output_video.mp4'  # Path for saving the output video

    def test_preprocess_image(self):
        """ Test image preprocessing (grayscale and Gaussian blur) """
        preprocessed_image = self.lane_detector.preprocess_image(self.test_image)
        self.assertEqual(preprocessed_image.shape, self.test_image.shape[:2])  # Grayscale image should have the same height/width as input

    def test_detect_edges(self):
        """ Test Canny edge detection """
        edges = self.lane_detector.detect_edges(self.test_image)
        self.assertEqual(edges.shape, self.test_image.shape[:2])  # Output edges image should be same size as input

    def test_video_edge_detection(self):
        """ Test edge detection on a video feed and saving output """
        # Open the video file
        cap = cv2.VideoCapture(self.test_video_path)
        self.assertTrue(cap.isOpened(), "Failed to open video file")

        ret, frame = cap.read()  # Read the first frame to test the video
        self.assertTrue(ret, "Failed to read frame from video")

        # Process the first frame using edge detection
        processed_frame = self.lane_detector.preprocess_image(frame)
        edges = self.lane_detector.detect_edges(processed_frame)
        
        # Check if edges are detected in the frame (edges should not be empty)
        self.assertTrue(np.any(edges), "No edges detected in the video frame")

        # Now save the processed video with edge detection applied
        self.lane_detector.detect_edges_in_camera_feed(self.test_video_path, self.output_video_path)

        # Verify that the output video was saved successfully
        self.assertTrue(os.path.exists(self.output_video_path), "Output video was not saved")

        cap.release()  # Close the video file

    def test_lane_departure_warning(self):
        """ Test lane departure warning printed outputs """
        with self.assertLogs(level="INFO") as log:
            lane_points = [
                (100, 500, 200, 300),  # Example left lane
                (300, 500, 400, 300)   # Example right lane
            ]
            self.lane_detector.lane_departure_warning(lane_points)
            # Check if the lane departure warning was logged correctly
            self.assertIn("Lane Departure Warning! Vehicle is moving out of lane.", log.output)

    def test_road_condition_detection(self):
        """ Test road condition detection output """
        with self.assertLogs(level="INFO") as log:
            frame = cv2.imread('images/test_image.jpg')  # Replace with a valid image path
            self.lane_detector.road_condition_detection(frame)
            # Check if the road condition message was logged correctly
            self.assertIn("Road Condition: Good Road Condition", log.output)
    
    def test_lane_departure_warning(self):
        """ Test lane departure warning printed outputs """
        with self.assertLogs(level="INFO") as log:
            # Simulate lane points for testing
            lane_points = [
                (100, 500, 200, 300),  # Left lane points
                (300, 500, 400, 300)   # Right lane points
            ]
            # Invoke the lane departure warning function
            self.lane_detector.lane_departure_warning(lane_points)
            
            # Test if the warning is correctly displayed
            self.assertIn("Lane Departure Warning! Vehicle is moving out of lane.", log.output)
            
            # Simulate neutral lane points (should trigger 'correctly centered' warning)
            lane_points = [
                (150, 500, 250, 300),  # Left lane points
                (350, 500, 450, 300)   # Right lane points
            ]
            self.lane_detector.lane_departure_warning(lane_points)
            self.assertIn("Lane Keeping is Correctly. Vehicle is staying centered within the lanes.", log.output)
            
            # Simulate diverging lanes (should trigger significant divergence warning)
            lane_points = [
                (100, 500, 150, 300),  # Left lane points
                (600, 500, 650, 300)   # Right lane points
            ]
            self.lane_detector.lane_departure_warning(lane_points)
            self.assertIn("Lane Departure Warning! Lanes are diverging significantly.", log.output)


if __name__ == '__main__':
    unittest.main()
