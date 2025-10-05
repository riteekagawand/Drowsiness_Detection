import cv2
import numpy as np
from scipy.spatial import distance
from pygame import mixer
import time

# Initialize pygame mixer for sound alerts
mixer.init()
mixer.music.load("music.wav")

def eye_aspect_ratio(eye):
    """Calculate the Eye Aspect Ratio (EAR) for drowsiness detection"""
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detect_eyes_haar(frame, eye_cascade):
    """Detect eyes using Haar cascade classifier"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)
    return eyes

def detect_eyes_dnn(frame, net):
    """Detect eyes using DNN face detection"""
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
    net.setInput(blob)
    detections = net.forward()
    
    eyes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            x1 = int(detections[0, 0, i, 3] * frame.shape[1])
            y1 = int(detections[0, 0, i, 4] * frame.shape[0])
            x2 = int(detections[0, 0, i, 5] * frame.shape[1])
            y2 = int(detections[0, 0, i, 6] * frame.shape[0])
            eyes.append((x1, y1, x2-x1, y2-y1))
    
    return eyes

def simple_blink_detection(frame):
    """Simple blink detection based on eye region analysis"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply threshold to create binary image
    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area (eyes should be medium-sized)
    eye_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 100 < area < 5000:  # Adjust these values based on your camera resolution
            eye_contours.append(contour)
    
    return len(eye_contours)

# Configuration
thresh = 0.25  # Eye aspect ratio threshold
frame_check = 10  # Number of consecutive frames below threshold to trigger alert
flag = 0
blink_counter = 0
last_blink_time = time.time()

# Dynamic calibration for Haar-based heuristic
haar_area_threshold = None  # set after warm-up calibration
calibration_count = 0
calibration_frames = 60  # ~2 seconds at 30 FPS
open_eye_baseline_avg = None  # running avg of open-eye area during calibration
ema_avg_area = None  # exponential moving average for smoothing
ema_alpha = 0.3  # smoothing factor for avg_area
calibrated_threshold_fraction = 0.75  # fraction of open-eye baseline used as threshold

# Fallback when eyes are not detected (e.g., fully closed or occluded)
no_eyes_counter = 0
no_eyes_frame_limit = 10  # frames without eyes before we consider drowsiness

# Initialize camera
cap = cv2.VideoCapture(0)

# Try to load different eye detection methods
eye_cascade = None
dnn_net = None

# Method 1: Try Haar cascade for eye detection
try:
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    print("Loaded Haar cascade eye detector")
except:
    print("Could not load Haar cascade eye detector")

# Method 2: Try DNN face detection
try:
    # Download the model files if not present
    # You can download them from: https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector
    prototxt_path = "deploy.prototxt"
    model_path = "res10_300x300_ssd_iter_140000.caffemodel"
    
    if cv2.os.path.exists(prototxt_path) and cv2.os.path.exists(model_path):
        dnn_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        print("Loaded DNN face detector")
    else:
        print("DNN model files not found. Using simple blink detection.")
except:
    print("Could not load DNN face detector")

print("Starting drowsiness detection...")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (450, 300))
    
    # Method 1: Haar cascade eye detection
    if eye_cascade is not None:
        eyes = detect_eyes_haar(frame, eye_cascade)
        
        if len(eyes) >= 2:  # Both eyes detected
            # Calculate average eye area as a proxy for openness
            total_area = 0
            for (x, y, w, h) in eyes:
                total_area += w * h
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            avg_area = total_area / len(eyes)

            # Update exponential moving average for smoothing
            if ema_avg_area is None:
                ema_avg_area = avg_area
            else:
                ema_avg_area = ema_alpha * avg_area + (1 - ema_alpha) * ema_avg_area

            # During initial calibration, learn open-eye baseline and set threshold
            if haar_area_threshold is None and calibration_count < calibration_frames:
                calibration_count += 1
                if open_eye_baseline_avg is None:
                    open_eye_baseline_avg = avg_area
                else:
                    open_eye_baseline_avg = 0.2 * avg_area + 0.8 * open_eye_baseline_avg
                # Show calibration progress
                print(f"[HAAR-CAL] {calibration_count}/{calibration_frames} avg_area={avg_area:.1f} baseline_avg={open_eye_baseline_avg:.1f}")
                # Do not trigger alerts during calibration window
                flag = 0
            elif haar_area_threshold is None and calibration_count >= calibration_frames:
                # Set threshold at a fraction of open-eye baseline (smaller area => more closed)
                haar_area_threshold = max(200.0, calibrated_threshold_fraction * open_eye_baseline_avg)
                print(f"[HAAR-CAL] threshold set to {haar_area_threshold:.1f} from baseline {open_eye_baseline_avg:.1f}")
                flag = 0
            else:
                # Debug to help calibrate threshold based on your camera/lighting
                print(f"[HAAR] eyes={len(eyes)}, avg_area={avg_area:.1f}, ema={ema_avg_area:.1f}, thr={haar_area_threshold:.1f}")

                # Simple heuristic with smoothing: smaller area = more closed eyes
                if ema_avg_area is not None and haar_area_threshold is not None and ema_avg_area < haar_area_threshold:
                    flag += 1
                    print(f"Drowsiness detected - Frame {flag}")
                else:
                    flag = 0
            # Reset the 'no eyes' counter because eyes are detected this frame
            no_eyes_counter = 0
        else:
            # If eyes are not detected, avoid carrying over stale counts
            no_eyes_counter += 1
            print(f"[HAAR] no eyes detected ({no_eyes_counter}/{no_eyes_frame_limit})")
            if no_eyes_counter >= no_eyes_frame_limit:
                flag += 1
                print(f"Drowsiness detected (no-eyes) - Frame {flag}")
            else:
                # before limit reached, do not accumulate stale flags
                flag = 0
    
    # Method 2: DNN face detection
    elif dnn_net is not None:
        eyes = detect_eyes_dnn(frame, dnn_net)
        
        if len(eyes) >= 1:  # Face detected
            for (x, y, w, h) in eyes:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Simple area-based detection
            avg_area = sum([w*h for (x, y, w, h) in eyes]) / len(eyes)
            # Debug to help calibrate threshold based on your camera/lighting
            print(f"[DNN] boxes={len(eyes)}, avg_area={avg_area:.1f}")
            if avg_area < 10000:  # Adjust threshold
                flag += 1
                print(f"Drowsiness detected - Frame {flag}")
            else:
                flag = 0
        else:
            # If no face/eyes detected by DNN, reset counter
            flag = 0
            no_eyes_counter += 1
            print(f"[DNN] no boxes detected ({no_eyes_counter}/{no_eyes_frame_limit})")
            if no_eyes_counter >= no_eyes_frame_limit:
                flag += 1
                print(f"Drowsiness detected (no-boxes) - Frame {flag}")
    
    # Method 3: Simple blink detection fallback
    else:
        blink_count = simple_blink_detection(frame)
        current_time = time.time()
        
        # If no eyes detected for a while, consider it drowsiness
        if blink_count < 1:
            blink_counter += 1
            if blink_counter > 10:  # No eyes detected for 10 frames
                flag += 1
                print(f"Drowsiness detected - Frame {flag}")
        else:
            blink_counter = 0
            flag = 0
    
    # Alert system
    if flag >= frame_check:
        cv2.putText(frame, "****************ALERT!****************", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "****************ALERT!****************", (10, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Play alert sound
        if not mixer.music.get_busy():
            mixer.music.play()
    
    # Display frame
    cv2.imshow("Drowsiness Detection", frame)
    
    # Exit on 'q' key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
mixer.quit()
print("Drowsiness detection stopped.")

