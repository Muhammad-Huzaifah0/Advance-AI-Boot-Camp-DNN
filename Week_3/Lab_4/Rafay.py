import numpy as np
import matplotlib.pyplot as plt
import cv2
import ipywidgets as widgets
from IPython.display import display, Video
from skimage import io
from io import BytesIO
from PIL import Image, ImageTk
import tkinter as tk
import threading

# Function to load image
def load_image(uploaded_file):
    image = io.imread(uploaded_file)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, img_gray

# Function to display images
def display_images(img_color, img_gray, title):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'{title} - Color')
    axes[0].axis('off')
    
    axes[1].imshow(img_gray, cmap='gray')
    axes[1].set_title(f'{title} - Grayscale')
    axes[1].axis('off')
    
    plt.show()

# Function to process video
def process_video(video_file, operation, **kwargs):
    cap = cv2.VideoCapture(video_file)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_file = 'output.avi'
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if operation == 'Apply Filter':
            frame = apply_filter(frame, kwargs['filter_type'], kwargs['intensity'])
        
        out.write(frame)
    
    cap.release()
    out.release()
    return output_file

# Function for edge detection
def edge_detection(image, sobel_strength=1):
    edge_detected = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_strength)
    return np.uint8(np.absolute(edge_detected))

# Function for padding
def pad_image(image, padding=1):
    return cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)

# Function for strided convolution
def strided_convolution(image, kernel, stride):
    output_shape = ((image.shape[0] - kernel.shape[0]) // stride + 1,
                    (image.shape[1] - kernel.shape[1]) // stride + 1)
    output = np.zeros(output_shape)
    
    for i in range(0, image.shape[0] - kernel.shape[0] + 1, stride):
        for j in range(0, image.shape[1] - kernel.shape[1] + 1, stride):
            output[i // stride, j // stride] = np.sum(image[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel)
    
    return output

# Function for max pooling
def max_pooling(image, pool_size=2, stride=2):
    output_shape = ((image.shape[0] - pool_size) // stride + 1,
                    (image.shape[1] - pool_size) // stride + 1)
    output = np.zeros(output_shape)
    
    for i in range(0, image.shape[0] - pool_size + 1, stride):
        for j in range(0, image.shape[1] - pool_size + 1, stride):
            output[i // stride, j // stride] = np.max(image[i:i+pool_size, j:j+pool_size])
    
    return output

# Function to apply filters
def apply_filter(image, filter_type='Identity', intensity=1.0):
    filters = {
        'Identity': np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
        'Laplacian': np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
        'Sharpening': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
        'Box Blur': np.ones((3, 3), np.float32) / 9.0,
        'Gaussian Blur': cv2.getGaussianKernel(3, 0.5) @ cv2.getGaussianKernel(3, 0.5).T,
        'Emboss': np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]),
        'High Pass': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
        'Low Pass': np.ones((3, 3), np.float32) / 9.0
    }
    
    kernel = filters.get(filter_type, filters['Identity'])
    filtered_image = cv2.filter2D(image, -1, kernel * intensity)
    return filtered_image

# Function to apply median filter
def apply_median_filter(image, kernel_size=3):
    return cv2.medianBlur(image, kernel_size)

# Function to convert to grayscale
def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Function to adjust brightness
def adjust_brightness(image, brightness=30):
    return cv2.convertScaleAbs(image, alpha=1, beta=brightness)

# Function to adjust contrast
def adjust_contrast(image, contrast=1.5):
    return cv2.convertScaleAbs(image, alpha=contrast, beta=0)

# Interactive UI for image processing
def interactive_ui():
    file_upload = widgets.FileUpload(accept='image/,video/', multiple=False)
    display(file_upload)
    img_color = img_gray = None

    def on_upload_change(change):
        nonlocal img_color, img_gray
        file_name = list(file_upload.value.keys())[0]
        file_extension = file_name.split('.')[-1]
        if file_extension in ['jpg', 'jpeg', 'png', 'bmp']:
            img_color, img_gray = load_image(file_upload.value[file_name])
            display_images(img_color, img_gray, 'Original Images')
            create_interactive_widgets()
        elif file_extension in ['mp4', 'avi', 'mov']:
            video_path = process_video(BytesIO(file_upload.value[file_name]), 'Apply Filter', filter_type='Sharpening', intensity=1.5)
            display(Video(video_path))

    def update_operation(operation, padding=1, stride=2, pool_size=2, sobel_strength=1, filter_type='Identity', intensity=1.0, brightness=30, contrast=1.5):
        if img_gray is None or img_color is None:
            return

        if operation == 'Edge Detection':
            result_gray = edge_detection(img_gray, sobel_strength)
            result_color = edge_detection(img_color, sobel_strength)
            display_images(result_color, result_gray, f'Edge Detection (Sobel Strength={sobel_strength})')
        elif operation == 'Padding':
            result_gray = pad_image(img_gray, padding)
            result_color = pad_image(img_color, padding)
            display_images(result_color, result_gray, f'Padded Image (Padding={padding})')
        elif operation == 'Strided Convolution':
            kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])  # Example kernel
            result_gray = strided_convolution(img_gray, kernel, stride)
            result_color = strided_convolution(img_color, kernel, stride)
            display_images(result_color, result_gray, f'Strided Convolution (Stride={stride})')
        elif operation == 'Max Pooling':
            result_gray = max_pooling(img_gray, pool_size, stride)
            result_color = max_pooling(img_color, pool_size, stride)
            display_images(result_color, result_gray, f'Max Pooling (Pool Size={pool_size}, Stride={stride})')
        elif operation == 'Apply Filter':
            result_gray = apply_filter(img_gray, filter_type, intensity)
            result_color = apply_filter(img_color, filter_type, intensity)
            display_images(result_color, result_gray, f'{filter_type} Filter (Intensity={intensity})')
        elif operation == 'Median Filter':
            result_gray = apply_median_filter(img_gray, 3)
            result_color = apply_median_filter(img_color, 3)
            display_images(result_color, result_gray, 'Median Filter')
        elif operation == 'Grayscale':
            result_gray = convert_to_grayscale(img_gray)
            result_color = convert_to_grayscale(img_color)
            display_images(result_color, result_gray, 'Grayscale')
        elif operation == 'Brightness Adjustment':
            result_gray = adjust_brightness(img_gray, brightness)
            result_color = adjust_brightness(img_color, brightness)
            display_images(result_color, result_gray, f'Brightness Adjustment (Value={brightness})')
        elif operation == 'Contrast Adjustment':
            result_gray = adjust_contrast(img_gray, contrast)
            result_color = adjust_contrast(img_color, contrast)
            display_images(result_color, result_gray, f'Contrast Adjustment (Alpha={contrast})')

    def create_interactive_widgets():
        operations = ['Edge Detection', 'Padding', 'Strided Convolution', 'Max Pooling', 'Apply Filter', 'Median Filter', 'Grayscale', 'Brightness Adjustment', 'Contrast Adjustment']
        dropdown = widgets.Dropdown(options=operations, description='Operation:')
        padding_slider = widgets.IntSlider(value=1, min=1, max=5, step=1, description='Padding:')
        stride_slider = widgets.IntSlider(value=2, min=1, max=5, step=1, description='Stride:')
        pool_size_slider = widgets.IntSlider(value=2, min=2, max=5, step=1, description='Pool Size:')
        sobel_strength_slider = widgets.IntSlider(value=1, min=1, max=5, step=1, description='Sobel Strength:')
        filter_dropdown = widgets.Dropdown(
            options=['Identity', 'Laplacian', 'Sharpening', 'Box Blur', 'Gaussian Blur', 'Emboss', 'High Pass', 'Low Pass'],
            description='Filter Type:'
        )
        intensity_slider = widgets.FloatSlider(value=1.0, min=0.1, max=10.0, step=0.1, description='Intensity:')
        brightness_slider = widgets.IntSlider(value=30, min=-100, max=100, step=1, description='Brightness:')
        contrast_slider = widgets.FloatSlider(value=1.5, min=0.5, max=3.0, step=0.1, description='Contrast:')

        sliders_box = widgets.VBox()

        def update_widgets(operation):
            if operation == 'Edge Detection':
                sliders_box.children = [sobel_strength_slider]
            elif operation == 'Padding':
                sliders_box.children = [padding_slider]
            elif operation == 'Strided Convolution':
                sliders_box.children = [stride_slider]
            elif operation == 'Max Pooling':
                sliders_box.children = [pool_size_slider, stride_slider]
            elif operation == 'Apply Filter':
                sliders_box.children = [filter_dropdown, intensity_slider]
            elif operation == 'Median Filter':
                sliders_box.children = []
            elif operation == 'Grayscale':
                sliders_box.children = []
            elif operation == 'Brightness Adjustment':
                sliders_box.children = [brightness_slider]
            elif operation == 'Contrast Adjustment':
                sliders_box.children = [contrast_slider]

        dropdown.observe(lambda change: update_widgets(change['new']), names='value')
        update_widgets('Edge Detection')

        ui = widgets.VBox([dropdown, sliders_box])
        out = widgets.interactive_output(update_operation, {
            'operation': dropdown,
            'padding': padding_slider,
            'stride': stride_slider,
            'pool_size': pool_size_slider,
            'sobel_strength': sobel_strength_slider,
            'filter_type': filter_dropdown,
            'intensity': intensity_slider,
            'brightness': brightness_slider,
            'contrast': contrast_slider
        })

        display(ui, out)

    file_upload.observe(on_upload_change, names='value')

interactive_ui()