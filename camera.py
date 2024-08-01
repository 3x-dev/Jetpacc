import cv2
import easyocr
import numpy as np
import threading

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def preprocess_frame(frame):
    """Convert the frame to grayscale, apply Gaussian blur, and use adaptive thresholding."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Denoise the image
    denoised = cv2.fastNlMeansDenoising(gray, h=30)
    
    # Adjust contrast
    alpha = 1.5  # Contrast control (1.0-3.0)
    beta = 0    # Brightness control (0-100)
    adjusted = cv2.convertScaleAbs(denoised, alpha=alpha, beta=beta)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(adjusted, (5, 5), 0)
    
    # Use adaptive thresholding to create a binary image
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    return binary

def ocr_text_from_frame(frame, text):
    """Extract text from the preprocessed frame using EasyOCR."""
    results = reader.readtext(frame)
    if results:
        text[0] = " ".join([res[1] for res in results])
    else:
        text[0] = ""

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    text = [""]
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Resize frame to reduce processing time
        frame = cv2.resize(frame, (640, 480))

        # Preprocess the frame
        processed_frame = preprocess_frame(frame)

        # Create a thread to process the frame
        thread = threading.Thread(target=ocr_text_from_frame, args=(processed_frame, text))
        thread.start()
        
        # Print the text in the terminal
        print(text[0])
        
        # Show the frames
        cv2.imshow('Camera Feed', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Wait for the thread to finish
        thread.join()

    # Release the capture and close the windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
