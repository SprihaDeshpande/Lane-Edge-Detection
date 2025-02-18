import cv2
import numpy as np
import os
import json
import time
from sklearn.metrics import precision_score, recall_score, f1_score

class LaneDetection:
    def __init__(self, threshold=50, hough_threshold=30, min_line_length=50, max_line_gap=150):
        self.threshold = threshold  # Canny edge detection threshold
        self.hough_threshold = hough_threshold  # Hough transform threshold
        self.min_line_length = min_line_length  # Min line length for Hough transform
        self.max_line_gap = max_line_gap  # Max gap between line segments
        
        # Kalman Filter for Temporal Smoothing
        self.kf = cv2.KalmanFilter(4, 2)  # 4 state variables (x, y, dx, dy), 2 measured variables (x, y)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], dtype=np.float32)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2  # Process noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1  # Measurement noise

        self.prev_gray = None  # To store the previous frame for optical flow calculation

    def preprocess_image(self, image):
        """ Preprocess the image by converting to grayscale, applying CLAHE, and Gaussian blur """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Apply stronger Gaussian blur to reduce noise and improve edge detection
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)  # Increased blur kernel size

        return blurred

    def detect_edges(self, image):
        """ Detect edges using Canny edge detection with adjusted thresholds """
        edges = cv2.Canny(image, 50, 150)  # Lowering the threshold to make edge detection more sensitive
        return edges

    def region_of_interest(self, image):
        """ Focus on the region of interest (bottom half of the image) for lane detection """
        height, width = image.shape
        mask = np.zeros_like(image)
        polygon = np.array([[
            (0, height),
            (width / 2, height / 2),
            (width, height)
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image

    def detect_lanes(self, image):
        """ Detect lanes using Hough Transform on the edges """
        processed_image = self.preprocess_image(image)  # Preprocess image
        edges = self.detect_edges(processed_image)  # Detect edges first
        region_of_interest_image = self.region_of_interest(edges)  # Apply ROI
        lines = cv2.HoughLinesP(region_of_interest_image, 1, np.pi / 180, self.hough_threshold, np.array([]), self.min_line_length, self.max_line_gap)

        lane_points = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                lane_points.append((x1, y1, x2, y2))
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Draw the lanes in green
        else:
            print("No lanes detected")
        
        return image, lane_points

    def lane_curve_detection(self, image, lane_points):
        """ Fit a curve to the detected lane points (Polynomial curve fitting for curved lanes) """
        curvature = None
        if len(lane_points) > 0:
            # Fit polynomial curve (2nd degree) to the detected lane points
            x_coords = np.array([pt[0] for pt in lane_points])
            y_coords = np.array([pt[1] for pt in lane_points])
            poly_coeff = np.polyfit(x_coords, y_coords, 2)  # 2nd degree polynomial fitting
            poly = np.poly1d(poly_coeff)

            # Calculate the curvature from the polynomial coefficients
            curvature = 2 * poly_coeff[0]  # Curvature = 2 * a (from y = ax^2 + bx + c)

            # Draw the fitted curve on the image
            x_line = np.linspace(min(x_coords), max(x_coords), num=100)
            y_line = poly(x_line)
            for i in range(len(x_line) - 1):
                cv2.line(image, (int(x_line[i]), int(y_line[i])), (int(x_line[i + 1]), int(y_line[i + 1])), (0, 255, 255), 2)
        
        return image, curvature

    def road_condition_detection(self, image):
        """ Detect road conditions (e.g., worn-out markings) using texture analysis and adaptive contrast """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()  # Variance of Laplacian to check texture detail
        road_condition = "Good Road Condition" if laplacian_var > 100 else "Poor Road Condition"
        print(f"Road Condition: {road_condition}")
        
        return image, road_condition

    def lane_departure_warning(self, lane_points):
        """ Lane departure warning: Check if the vehicle is deviating from its lane """
        if len(lane_points) > 0:
            left_lane = lane_points[0]
            right_lane = lane_points[-1]
            
            # Calculate the center of the detected lanes
            left_lane_center = (left_lane[0] + left_lane[2]) / 2
            right_lane_center = (right_lane[0] + right_lane[2]) / 2
            
            # Calculate the center of the frame (assuming the car is supposed to be in the center of the road)
            frame_center = 640  # Adjust based on your frame resolution (e.g., 1280 for 1280x720 video)

            # Calculate the width of the lane
            lane_width = abs(right_lane_center - left_lane_center)
            
            # Allow some margin before triggering lane departure
            margin_of_error = lane_width * 0.1  # 10% of lane width as margin for error
            
            # Check if the lanes are diverging significantly
            if abs(left_lane_center - right_lane_center) > lane_width * 0.4:  # 50% of lane width as threshold for divergence
                print("Lane Departure Warning! Lanes are diverging significantly.")
            
            # Check if the vehicle is moving out of lane (vehicle is too far from the center)
            elif abs(left_lane_center - frame_center) > margin_of_error or abs(right_lane_center - frame_center) > margin_of_error:
                print("Lane Departure Warning! Vehicle is moving out of lane.")
            
            # If neither condition is true, the vehicle is keeping in the lane
            else:
                print("Lane Keeping is Correctly. Vehicle is staying centered within the lanes.")




    def calculate_optical_flow(self, prev_gray, curr_gray):
        """ Calculate optical flow to detect motion (e.g., vehicles moving) """
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return np.mean(magnitude)  # Return the average magnitude as an indicator of motion

    def detect_edges_in_camera_feed(self, video_path, output_path):
        """ Capture edges in a live camera feed (or video file) and save the output """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("Error: Could not open video.")
            return
        
        # Get frame dimensions for VideoWriter
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Set up the VideoWriter to save the processed video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))

        prev_gray = None  # Initialize prev_gray to None at the start

        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or unable to read frame.")
                break

            # Record the start time for performance measurement
            start_time = time.time()

            # Process the frame (detect edges only)
            processed_frame = self.preprocess_image(frame)
            edges = self.detect_edges(processed_frame)

            # Apply lane detection
            processed_frame, lane_points = self.detect_lanes(frame)

            # Apply lane curvature detection
            processed_frame, curvature = self.lane_curve_detection(processed_frame, lane_points)

            # Apply road condition detection
            processed_frame, road_condition = self.road_condition_detection(frame)

            # Print lane curvature and road condition to the terminal
            if curvature is not None:
                print(f"Lane Curvature: {curvature:.4f}")
            
            # Lane Departure Warning
            self.lane_departure_warning(lane_points)

            # Optical Flow Calculation for Vehicle Detection / Speed Estimation
            if prev_gray is not None:
                # Ensure prev_gray and processed_frame are both grayscale
                prev_gray_resized = cv2.resize(prev_gray, (processed_frame.shape[1], processed_frame.shape[0]))  # Resize to match
                prev_gray_resized = cv2.cvtColor(prev_gray_resized, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                processed_frame_gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)  # Ensure processed_frame is grayscale
                
                motion_magnitude = self.calculate_optical_flow(prev_gray_resized, processed_frame_gray)
                print(f"Optical Flow Magnitude (Motion): {motion_magnitude:.2f}")

            # Record the time taken for processing the frame
            end_time = time.time()
            processing_time = end_time - start_time
            print(f"Processing time for current frame: {processing_time:.4f} seconds")

            # Write the processed edge-detected frame to the output video
            out.write(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))  # Convert edges to color and write

            # Display the edge-detected video
            cv2.imshow("Edge-detected Video", edges)

            # Break the loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Update prev_gray with the current processed frame for optical flow calculation
            prev_gray = processed_frame.copy()

        cap.release()
        out.release()
        cv2.destroyAllWindows()
