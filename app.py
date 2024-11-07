import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

# Set up the page title and description
st.title("Fringe Pattern Intensity Analyzer")
st.write("Upload a fringe pattern image, and this app will analyze the intensity along the central row, plot it, and provide a CSV file for download.")

# Upload image file
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# Function to process the image and get the central row intensity values
def analyze_fringe_pattern(image):
    # Convert image to grayscale if needed
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    # Select the central row
    central_row = gray_image[gray_image.shape[0] // 2, :]
    return central_row

# Function to create and download CSV
def create_csv_download_link(data):
    # Convert data to CSV
    df = pd.DataFrame(data, columns=["Intensity"])
    csv = df.to_csv(index_label="Pixel Position")
    # Convert to binary for download link
    b = BytesIO()
    b.write(csv.encode())
    b.seek(0)
    return b

# If an image is uploaded
if uploaded_file is not None:
    # Read the image with OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    # Analyze the image to get intensity values along the central row
    central_row = analyze_fringe_pattern(image)
    
    # Display the intensity plot
    st.write("### Fringe Pattern Intensity Plot")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(central_row, color="green")
    ax.set_title("Fringe Pattern Intensity Plot")
    ax.set_xlabel("Pixel Position")
    ax.set_ylabel("Intensity")
    st.pyplot(fig)

    # Create a downloadable CSV file
    csv_data = create_csv_download_link(central_row)
    st.download_button(
        label="Download Intensity Data as CSV",
        data=csv_data,
        file_name="fringe_intensity.csv",
        mime="text/csv"
    )
