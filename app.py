import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def load_image(image_file):
    """Load an image file and convert it to PIL format"""
    try:
        img = Image.open(image_file)
        return img
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None

def convert_to_cv2(pil_img):
    """Convert PIL image to OpenCV format"""
    try:
        numpy_img = np.array(pil_img)
        if len(numpy_img.shape) == 3 and numpy_img.shape[2] == 4:
            numpy_img = cv2.cvtColor(numpy_img, cv2.COLOR_RGBA2RGB)
        opencv_img = cv2.cvtColor(numpy_img, cv2.COLOR_RGB2BGR)
        return opencv_img
    except Exception as e:
        st.error(f"Error converting image: {str(e)}")
        return None

def process_image(cv2_image, transform_option, resize_percent, edge_low, edge_high, compression_quality, jpeg_quality):
    """Process image based on selected transformation with compression"""
    try:
        if transform_option == "Resized":
            if resize_percent != 100:
                width = int(cv2_image.shape[1] * resize_percent / 100)
                height = int(cv2_image.shape[0] * resize_percent / 100)
                processed = cv2.resize(cv2_image, (width, height), interpolation=cv2.INTER_AREA)
                
                # Apply JPEG compression
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
                _, encoded = cv2.imencode('.jpg', processed, encode_param)
                processed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                
                return processed, cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            return cv2_image, cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
            
        elif transform_option == "Grayscale":
            processed = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
            return processed, processed
            
        elif transform_option == "Edge Detection":
            gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            processed = cv2.Canny(blurred, edge_low, edge_high)
            return processed, processed
            
        return None, None
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None

def get_file_size(buf):
    """Get size of file in MB"""
    return len(buf.getvalue()) / (1024 * 1024)

def main():
    st.set_page_config(page_title="Vision App", layout="wide")
    
    st.title("Computer Vision Web App")
    st.sidebar.header("Image Processing Controls")

    # File uploader
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display original file size
        original_size = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.sidebar.write(f"Original file size: {original_size:.2f} MB")

        # Load and display original image
        image = load_image(uploaded_file)
        if image is not None:
            cv2_image = convert_to_cv2(image)
            if cv2_image is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original Image")
                    st.image(image, use_column_width=True)

                # Image processing controls
                with st.sidebar:
                    st.subheader("Transformation Controls")
                    resize_percent = st.slider("Resize %", 10, 200, 100)
                    edge_low = st.slider("Edge Detection - Low Threshold", 0, 255, 100)
                    edge_high = st.slider("Edge Detection - High Threshold", 0, 255, 200)
                    
                    st.subheader("Compression Controls")
                    compression_quality = st.slider("PNG Compression Level (0-9)", 0, 9, 6)
                    jpeg_quality = st.slider("JPEG Quality (0-100)", 0, 100, 85)

                # Process image
                with col2:
                    transform_option = st.selectbox(
                        "Choose Transformation",
                        ["Resized", "Grayscale", "Edge Detection"]
                    )

                    processed_img, display_img = process_image(
                        cv2_image, 
                        transform_option, 
                        resize_percent, 
                        edge_low, 
                        edge_high,
                        compression_quality,
                        jpeg_quality
                    )

                    if processed_img is not None:
                        st.subheader(f"{transform_option} Image")
                        st.image(display_img, use_column_width=True)

                        # Download button with compression
                        buf = io.BytesIO()
                        if isinstance(processed_img, np.ndarray):
                            if len(processed_img.shape) == 2:  # Grayscale
                                Image.fromarray(processed_img).save(
                                    buf, 
                                    format="PNG", 
                                    optimize=True,
                                    compression_level=compression_quality
                                )
                            else:  # RGB
                                Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)).save(
                                    buf, 
                                    format="JPEG", 
                                    quality=jpeg_quality,
                                    optimize=True
                                )
                            
                            # Display processed file size
                            processed_size = get_file_size(buf)
                            st.write(f"Processed file size: {processed_size:.2f} MB")
                            st.write(f"Size reduction: {((original_size - processed_size)/original_size * 100):.1f}%")
                            
                            st.download_button(
                                label="Download processed image",
                                data=buf.getvalue(),
                                file_name=f"processed_{transform_option.lower()}.jpg",
                                mime="image/jpeg"
                            )

if __name__ == "__main__":
    main()