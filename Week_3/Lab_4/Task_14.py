# Importing required modules for GUI, image processing, and threading
import tkinter as tk  # GUI library
from tkinter import filedialog, messagebox  # File dialog and message box
from PIL import Image, ImageTk  # Image handling
import cv2  # OpenCV library for image processing
import numpy as np  # NumPy library for numerical operations
import threading  # Threading library for parallel processing

class YOLOFaceDetectionApp:
    def __init__(self, root):
        """
        Initialize the YOLO Face Detection App
        """
        self.root = root  # Tkinter root window
        self.root.title("YOLO Object Detection and Filters")  # Set window title

        # File paths for YOLO model files
        self.weights_path = ""
        self.cfg_path = ""
        self.names_path = ""

        # GUI elements
        self.panel = tk.Label(root)  # Label to display images/videos
        self.panel.pack(padx=10, pady=10)  # Add padding around the label

        btn_frame = tk.Frame(root)  # Frame to hold buttons
        btn_frame.pack(fill=tk.X, pady=10)  # Add padding and fill horizontally

        # Buttons for various functionalities
        btn_select_weights = tk.Button(btn_frame, text="Select Weights", command=self.select_weights)
        btn_select_weights.pack(side=tk.LEFT, padx=10)  # Button to select YOLO weights file

        btn_select_cfg = tk.Button(btn_frame, text="Select CFG", command=self.select_cfg)
        btn_select_cfg.pack(side=tk.LEFT, padx=10)  # Button to select YOLO cfg file

        btn_select_names = tk.Button(btn_frame, text="Select Names", command=self.select_names)
        btn_select_names.pack(side=tk.LEFT, padx=10)  # Button to select YOLO names file

        btn_select_image = tk.Button(btn_frame, text="Select Image", command=self.select_image)
        btn_select_image.pack(side=tk.LEFT, padx=10)  # Button to select an image file

        btn_select_video = tk.Button(btn_frame, text="Select Video", command=self.select_video)
        btn_select_video.pack(side=tk.LEFT, padx=10)  # Button to select a video file

        btn_live_video = tk.Button(btn_frame, text="Live Video", command=self.toggle_live_video)
        btn_live_video.pack(side=tk.LEFT, padx=10)  # Button to toggle live video from webcam

        btn_detect_objects = tk.Button(btn_frame, text="Detect Objects", command=self.toggle_detect_objects)
        btn_detect_objects.pack(side=tk.LEFT, padx=10)  # Button to toggle object detection

        btn_edge_detection = tk.Button(btn_frame, text="Edge Detection", command=self.toggle_edge_detection)
        btn_edge_detection.pack(side=tk.LEFT, padx=10)  # Button to toggle edge detection filter

        btn_sharpen = tk.Button(btn_frame, text="Sharpen", command=self.toggle_sharpen)
        btn_sharpen.pack(side=tk.LEFT, padx=10)  # Button to toggle sharpen filter

        btn_sharpen = tk.Button(btn_frame, text="Gaussian Blur", command=self.toggle_sharpen)
        btn_sharpen.pack(side=tk.LEFT, padx=10)  # Button to toggle sharpen filter


        # Other variables
        self.image_path = None  # Path to the selected image
        self.video_path = None  # Path to the selected video
        self.image = None  # Loaded image
        self.video_capture = None  # Video capture object
        self.net = None  # YOLO network
        self.classes = None  # YOLO class labels
        self.output_layers = None  # YOLO output layers
        self.filter_mode = None  # Current filter mode
        self.detect_objects_flag = False  # To track object detection state
        self.running = False  # To track live video state
        self.thread = None  # Thread for parallel processing

    # File selection methods
    def select_weights(self):
        """
        Select YOLO weights file
        """
        self.weights_path = filedialog.askopenfilename()  # Open file dialog to select weights file
        self.load_yolo()  # Load YOLO model if all files are provided

    def select_cfg(self):
        """
        Select YOLO cfg file
        """
        self.cfg_path = filedialog.askopenfilename()  # Open file dialog to select cfg file
        self.load_yolo()  # Load YOLO model if all files are provided

    def select_names(self):
        """
        Select YOLO names file
        """
        self.names_path = filedialog.askopenfilename()  # Open file dialog to select names file
        self.load_yolo()  # Load YOLO model if all files are provided

    def load_yolo(self):
        """
        Load YOLO model if all required files are provided
        """
        if self.weights_path and self.cfg_path and self.names_path:  # Check if all files are provided
            try:
                self.net = cv2.dnn.readNet(self.weights_path, self.cfg_path)  # Load YOLO network
                self.layer_names = self.net.getLayerNames()  # Get YOLO layer names
                self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]  # Get YOLO output layers
                with open(self.names_path, "r") as f:
                    self.classes = [line.strip() for line in f.readlines()]  # Read YOLO class labels
                messagebox.showinfo("YOLO", "YOLO model loaded successfully.")  # Show success message
            except Exception as e:
                messagebox.showerror("YOLO Error", f"Error loading YOLO: {e}")  # Show error message

    # Image and video selection methods
    def select_image(self):
        """
        Select an image file
        """
        self.image_path = filedialog.askopenfilename()  # Open file dialog to select image file
        if len(self.image_path) > 0:  # Check if an image file is selected
            self.load_image()  # Load the selected image

    def select_video(self):
        """
        Select a video file
        """
        self.video_path = filedialog.askopenfilename()  # Open file dialog to select video file
        if len(self.video_path) > 0:  # Check if a video file is selected
            if self.running:  # Check if video is already running
                self.stop_running()  # Stop running video
            else:
                self.running = True  # Set running state to True
                self.thread = threading.Thread(target=self.detect_objects_video)  # Create a new thread for video processing
                self.thread.start()  # Start the thread

    def toggle_live_video(self):
        """
        Start or stop the live video feed
        """
        if self.running:  # Check if live video is already running
            self.stop_running()  # Stop running live video
        else:
            self.video_capture = cv2.VideoCapture(0)  # Open webcam for live video
            self.running = True  # Set running state to True
            self.thread = threading.Thread(target=self.show_live_video)  # Create a new thread for live video
            self.thread.start()  # Start the thread

    def load_image(self):
        """
        Load the selected image and display it
        """
        image = cv2.imread(self.image_path)  # Read the selected image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image from BGR to RGB
        image = Image.fromarray(image)  # Convert image to PIL format
        image = ImageTk.PhotoImage(image)  # Convert image to ImageTk format

        self.panel.config(image=image)  # Update the panel with the image
        self.panel.image = image  # Keep a reference to avoid garbage collection

    # Toggle methods for object detection and filters
    def toggle_detect_objects(self):
        """
        Toggle object detection on or off
        """
        self.detect_objects_flag = not self.detect_objects_flag  # Toggle the detect_objects_flag

    def toggle_edge_detection(self):
        """
        Toggle edge detection filter on or off
        """
        if self.filter_mode == "edge":  # Check if edge detection is already active
            self.filter_mode = None  # Deactivate edge detection
        else:
            self.filter_mode = "edge"  # Activate edge detection
            if not self.running:  # Check if live video is not running
                self.toggle_live_video()  # Start live video

    def toggle_sharpen(self):
        """
        Toggle sharpening filter on or off
        """
        if self.filter_mode == "sharpen":  # Check if sharpening is already active
            self.filter_mode = None  # Deactivate sharpening
        else:
            self.filter_mode = "sharpen"  # Activate sharpening
            if not self.running:  # Check if live video is not running
                self.toggle_live_video()  # Start live video

    def toggle_gaussian_blur(self):
        """
        Toggle Gaussian Blur filter on or off
        """
        if self.filter_mode == "gaussian_blur":  # Check if Gaussian Blur is already active
            self.filter_mode = None  # Deactivate Gaussian Blur
        else:
            self.filter_mode = "gaussian_blur"  # Activate Gaussian Blur
            if not self.running:  # Check if live video is not running
                self.toggle_live_video()  # Start live video

    # Object detection methods
    def _detect_objects(self, frame):
        """
        Apply YOLO object detection to the frame
        """
        return self.apply_yolo(frame)  # Apply YOLO object detection

    def detect_objects_video(self):
        """
        Detect objects in the selected video file
        """
        cap = cv2.VideoCapture(self.video_path)  # Open the selected video file
        while self.running:  # Check if the video is running
            ret, frame = cap.read()  # Read a frame from the video
            if not ret:  # Check if frame is not read successfully
                break  # Exit the loop

            if self.detect_objects_flag:  # Check if object detection is active
                frame = self.apply_yolo(frame)  # Apply YOLO object detection

            if self.filter_mode == "edge":
                frame = cv2.Canny(frame, 100, 200)  # Apply edge detection
            elif self.filter_mode == "sharpen":
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                frame = cv2.filter2D(frame, -1, kernel)  # Apply sharpening
            elif self.filter_mode == "gaussian_blur":
                frame = cv2.GaussianBlur(frame, (15, 15), 0)  # Apply Gaussian Blur

            cv2.imshow("Video", frame)  # Display the frame
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Check if 'q' key is pressed
                break  # Exit the loop

        cap.release()  # Release the video capture object
        cv2.destroyAllWindows()  # Close all OpenCV windows
        self.running = False  # Set running state to False

    def show_live_video(self):
        """
        Display live video feed from the webcam
        """
        while self.running:  # Check if live video is running
            ret, frame = self.video_capture.read()  # Read a frame from the webcam
            if not ret:  # Check if frame is not read successfully
                break  # Exit the loop

            if self.detect_objects_flag:  # Check if object detection is active
                frame = self.apply_yolo(frame)  # Apply YOLO object detection

            if self.filter_mode == "edge":  # Check if edge detection is active
                frame = cv2.Canny(frame, 100, 200)  # Apply edge detection
            elif self.filter_mode == "sharpen":  # Check if sharpening is active
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Sharpening kernel
                frame = cv2.filter2D(frame, -1, kernel)  # Apply sharpening
            elif self.filter_mode == "gaussian_blur":
                frame = cv2.GaussianBlur(frame, (15, 15), 0)  # Apply Gaussian Blur

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
            image = Image.fromarray(frame)  # Convert frame to PIL format
            image = ImageTk.PhotoImage(image)  # Convert frame to ImageTk format

            self.panel.config(image=image)  # Update the panel with the frame
            self.panel.image = image  # Keep a reference to avoid garbage collection

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Check if 'q' key is pressed
                break  # Exit the loop

        self.video_capture.release()  # Release the webcam
        self.running = False  # Set running state to False
    def stop_running(self):
        """
        Stop all running processes, release the video capture, and close all windows
        """
        self.running = False  # Set running state to False
        self.detect_objects_flag = False  # Deactivate object detection
        self.filter_mode = None  # Deactivate filter mode
        if self.video_capture is not None:  # Check if video capture object is not None
            self.video_capture.release()  # Release the video capture object
            self.video_capture = None  # Set video capture object to None
        cv2.destroyAllWindows()  # Close all OpenCV windows

    def apply_yolo(self, image):
        """
        Apply YOLO object detection to the provided image
        """
        if not self.net:  # Check if YOLO model is not loaded
            messagebox.showerror("YOLO Error", "YOLO model is not loaded. Please load the model first.")  # Show error message
            return image  # Return the original image

        height, width, channels = image.shape  # Get the dimensions of the image

        # Create a 4D blob from a frame
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)  # Create a blob from the image
        self.net.setInput(blob)  # Set the blob as input to the network
        outs = self.net.forward(self.output_layers)  # Get the output from the network

        class_ids = []  # List to hold class IDs
        confidences = []  # List to hold confidences
        boxes = []  # List to hold bounding boxes

        # Iterate over each detection
        for out in outs:
            for detection in out:
                scores = detection[5:]  # Get the scores for all classes
                class_id = np.argmax(scores)  # Get the class ID with the highest score
                confidence = scores[class_id]  # Get the confidence of the highest score
                if confidence > 0.5:  # Check if confidence is above the threshold
                    center_x = int(detection[0] * width)  # Calculate center x-coordinate
                    center_y = int(detection[1] * height)  # Calculate center y-coordinate
                    w = int(detection[2] * width)  # Calculate width of the bounding box
                    h = int(detection[3] * height)  # Calculate height of the bounding box

                    x = int(center_x - w / 2)  # Calculate top-left x-coordinate
                    y = int(center_y - h / 2)  # Calculate top-left y-coordinate

                    if (x, y, w, h) and isinstance(x, int) and isinstance(y, int) and isinstance(w, int) and isinstance(h, int):  # Check if coordinates and dimensions are valid
                        boxes.append([x, y, w, h])  # Add bounding box to the list
                        confidences.append(float(confidence))  # Add confidence to the list
                        class_ids.append(class_id)  # Add class ID to the list

        # Apply non-maxima suppression to suppress weak overlapping boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)  # Perform non-maxima suppression

        # Draw bounding boxes and labels on the image
        for i in range(len(boxes)):  # Iterate over bounding boxes
            if i in indexes:  # Check if index is in the list of remaining indexes
                x, y, w, h = boxes[i]  # Get the coordinates and dimensions of the bounding box
                label = str(self.classes[class_ids[i]])  # Get the class label
                confidence = confidences[i]  # Get the confidence
                color = (0, 255, 0)  # Set the color for the bounding box
                print(f"Drawing rectangle at {(x, y)} to {(x + w, y + h)}")  # Print coordinates for debugging
                if isinstance(x, int) and isinstance(y, int) and isinstance(w, int) and isinstance(h, int):  # Check if coordinates and dimensions are valid
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)  # Draw the bounding box
                    cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # Draw the label and confidence

        return image  # Return the image with bounding boxes

    def apply_filter_to_image(self, filter_type):
        """
        Apply the selected filter to the loaded image
        """
        image = cv2.imread(self.image_path)  # Read the selected image
        if filter_type == "edge":  # Check if edge detection filter is selected
            image = cv2.Canny(image, 100, 200)  # Apply edge detection filter
        elif filter_type == "sharpen":  # Check if sharpening filter is selected
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Sharpening kernel
            image = cv2.filter2D(image, -1, kernel)  # Apply sharpening filter
        elif self.filter_mode == "gaussian_blur":
            frame = cv2.GaussianBlur(frame, (15, 15), 0)  # Apply Gaussian Blur

        image = Image.fromarray(image)  # Convert image to PIL format
        image = ImageTk.PhotoImage(image)  # Convert image to ImageTk format

        self.panel.config(image=image)  # Update the panel with the image
        self.panel.image = image  # Keep a reference to avoid garbage collection

# Main execution block
if __name__ == "__main__":
    root = tk.Tk()  # Create the Tkinter root window
    app = YOLOFaceDetectionApp(root)  # Create the YOLOFaceDetectionApp
    root.mainloop()  # Start the Tkinter main loop
