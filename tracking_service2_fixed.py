import cv2
import dlib
import numpy as np
import base64
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from threading import Lock
from collections import deque
import socket

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration

# --- Ethernet Configuration ---
JETSON_IP = "192.168.10.2"   # Change this to Jetson's Ethernet IP
PORT = 6000
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((JETSON_IP, PORT))
print("Connected to Jetson")

def send_command(command):
    try:
        client_socket.sendall((command + "\n").encode())
        print("Sent:", command)
    except Exception as e:
        print("Error sending:", e)

# Movement thresholds
CENTER_TOLERANCE = 80  # Pixels from center to consider "centered"
MIN_BBOX_AREA = 5000  # Minimum area to consider a valid target
OPTIMAL_BBOX_HEIGHT = 200  # Target height for person in frame
HEIGHT_TOLERANCE = 80  # Tolerance for optimal height

# Tracking State
tracking_state = {
    "active": False,
    "tracker": None,
    "target_bbox": None,
    "frame_width": 640,
    "frame_height": 480,
    "lost_frames": 0,
    "max_lost_frames": 60,  # Give up after 1 second at 30fps
    "tracking_history": deque(maxlen=10),
}
state_lock = Lock()

# Face detector for initial detection
try:
    face_detector = dlib.get_frontal_face_detector()
    print("dlib face detector loaded successfully")
except Exception as e:
    print(f"Failed to load dlib face detector: {e}")
    print("Install dlib: pip install dlib")
    face_detector = None

# Tracking Functions

def detect_closest_person(frame):
    """
    Detect the closest person (largest face) in the frame
    Returns: bounding box (x, y, w, h) or None
    """
    if face_detector is None:
        print("Face detector not available")
        return None
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 1)

    if not faces:
        return None

    # Find the largest face (closest person)
    largest_face = max(faces, key=lambda f: f.width() * f.height())

    # Add padding to bbox to include body
    x, y = largest_face.left(), largest_face.top()
    w, h = largest_face.width(), largest_face.height()

    # Expand bbox to include upper body (tracking even when turned around)
    padding_x = int(w * 0.20)
    padding_y = int(h * 0.5)  # Extend downward for body

    x = max(0, x - padding_x)
    y = max(0, y - int(padding_y * 0.2))
    w = min(frame.shape[1] - x, w + 2 * padding_x)
    h = min(frame.shape[0] - y, h + padding_y)

    return (x, y, w, h)

def create_tracker():
    """Create a new tracker instance - CSRT is best for accuracy"""
    try:
        # Try new OpenCV 4.5+ syntax first
        return cv2.legacy.TrackerCSRT_create()
    except (AttributeError, ImportError):
        try:
            # Try older OpenCV syntax
            return cv2.TrackerCSRT_create()
        except (AttributeError, ImportError):
            try:
                # Try KCF tracker (more compatible)
                return cv2.legacy.TrackerKCF_create()
            except (AttributeError, ImportError):
                try:
                    # Try older KCF syntax
                    return cv2.TrackerKCF_create()
                except (AttributeError, ImportError):
                    print("No compatible tracker found!")
                    print("Install opencv-contrib-python: pip install opencv-contrib-python")
                    raise Exception("No OpenCV tracker available. Install opencv-contrib-python")

def calculate_movement_command(bbox, frame_width, frame_height):
    """
    Calculate movement command based on target position and size
    Returns: command string for Jetson
    """
    if bbox is None:
        return "STOP"

    x, y, w, h = bbox

    # Calculate center of bounding box
    bbox_center_x = x + w // 2
    frame_center_x = frame_width // 2

    # Calculate offsets
    horizontal_offset = bbox_center_x - frame_center_x

    # Determine horizontal movement (turning)
    if abs(horizontal_offset) < CENTER_TOLERANCE:
        # Person is centered, check distance
        if h < OPTIMAL_BBOX_HEIGHT - HEIGHT_TOLERANCE:
            return "FORWARD"
        elif h > OPTIMAL_BBOX_HEIGHT + HEIGHT_TOLERANCE:
            return "BACKWARD"
        else:
            return "STOP"  # Perfect position
    else:
        # Need to turn - using standard keywords for Jetson
        if horizontal_offset > 0:
            return "TURN_RIGHT"
        else:
            return "TURN_LEFT"

def process_tracking_frame(frame):
    """
    Process frame for tracking - update tracker and determine movement
    """
    global tracking_state

    with state_lock:
        if not tracking_state["active"]:
            return frame, "STOP"
        
        frame_height, frame_width = frame.shape[:2]
        tracking_state["frame_width"] = frame_width
        tracking_state["frame_height"] = frame_height
        
        # Update tracker
        success, bbox = tracking_state["tracker"].update(frame)
        
        if success:
            # Valid tracking
            tracking_state["target_bbox"] = bbox
            tracking_state["lost_frames"] = 0
            
            # Add to history for smoothing
            tracking_state["tracking_history"].append(bbox)
            
            # Draw tracking box
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(frame, "TRACKING", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Calculate movement
            command = calculate_movement_command((x, y, w, h), frame_width, frame_height)
            
            # Draw center crosshair
            cv2.circle(frame, (frame_width // 2, frame_height // 2), 5, (0, 0, 255), -1)
            cv2.circle(frame, (x + w // 2, y + h // 2), 5, (0, 255, 0), -1)
            cv2.line(frame, (frame_width // 2, frame_height // 2), 
                    (x + w // 2, y + h // 2), (255, 0, 0), 2)
            
            return frame, command
        else:
            # Lost tracking
            tracking_state["lost_frames"] += 1
            
            if tracking_state["lost_frames"] > tracking_state["max_lost_frames"]:
                # Try to re-detect person
                new_bbox = detect_closest_person(frame)
                
                if new_bbox is not None:
                    # Re-initialize tracker
                    tracking_state["tracker"] = create_tracker()
                    tracking_state["tracker"].init(frame, new_bbox)
                    tracking_state["lost_frames"] = 0
                    cv2.putText(frame, "RE-ACQUIRED TARGET", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    # Completely lost - stop and search
                    cv2.putText(frame, "TARGET LOST - SEARCHING", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    return frame, "STOP"
            else:
                # Still trying to recover
                cv2.putText(frame, f"TRACKING LOST ({tracking_state['lost_frames']})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                return frame, "STOP"
            
            return frame, "STOP"

# API Endpoints

@app.get("/")
def root():
    return {
        "message": "KAIRA Person Tracking Service",
        "status": "active",
        "tracking_active": tracking_state["active"]
    }

@app.post("/start_tracking")
async def start_tracking(file: UploadFile = File(...)):
    """
    Start tracking mode - detect closest person and initialize tracker
    """
    global tracking_state

    try:
        # Read frame
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid frame")
        
        # Detect closest person
        bbox = detect_closest_person(frame)
        
        if bbox is None:
            return JSONResponse({
                "success": False,
                "message": "No person detected in frame"
            })
        
        # Initialize tracker
        with state_lock:
            tracking_state["tracker"] = create_tracker()
            tracking_state["tracker"].init(frame, bbox)
            tracking_state["target_bbox"] = bbox
            tracking_state["active"] = True
            tracking_state["lost_frames"] = 0
            tracking_state["tracking_history"].clear()
            tracking_state["frame_width"] = frame.shape[1]
            tracking_state["frame_height"] = frame.shape[0]
        
        # Draw initial detection
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(frame, "TARGET LOCKED", (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Encode frame
        success, buffer = cv2.imencode(".jpg", frame)
        frame_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
        
        print("✅ Tracking started - target locked")
        
        return JSONResponse({
            "success": True,
            "message": "Tracking started",
            "target_bbox": bbox,
            "frame": frame_b64
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting tracking: {str(e)}")

@app.post("/stop_tracking")
async def stop_tracking():
    """
    Stop tracking mode and return to normal operation
    """
    global tracking_state

    with state_lock:
        tracking_state["active"] = False
        tracking_state["tracker"] = None
        tracking_state["target_bbox"] = None
        tracking_state["lost_frames"] = 0
        tracking_state["tracking_history"].clear()

    # Stop robot movement
    send_command("STOP")

    print("⏹ Tracking stopped")

    return JSONResponse({
        "success": True,
        "message": "Tracking stopped"
    })

@app.post("/track_frame")
async def track_frame(file: UploadFile = File(...)):
    """
    Process a frame for tracking and return movement command
    """
    global tracking_state

    try:
        # Read frame
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid frame")
        
        # Process frame
        processed_frame, command = process_tracking_frame(frame)
        
        # Send command to robot
        send_command(command)
        
        # Encode frame
        success, buffer = cv2.imencode(".jpg", processed_frame)
        frame_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
        
        return JSONResponse({
            "success": True,
            "command": command,
            "tracking_active": tracking_state["active"],
            "frame": frame_b64,
            "lost_frames": tracking_state["lost_frames"]
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing frame: {str(e)}")

@app.get("/tracking_status")
def get_tracking_status():
    """Get current tracking status"""
    with state_lock:
        return {
            "active": tracking_state["active"],
            "has_target": tracking_state["target_bbox"] is not None,
            "lost_frames": tracking_state["lost_frames"],
            "target_bbox": tracking_state["target_bbox"]
        }

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    send_command("stop")
    print("👋 Tracking Service shutdown")

# Camera Display Functions
def start_camera_display():
    """Start live camera feed with tracking visualization"""
    import threading
    import time
    
    def camera_loop():
        # Initialize camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("📹 Camera feed started - Press 'q' to quit, 's' to start tracking, 'x' to stop tracking")
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame from camera")
                break
            
            display_frame = frame.copy()
            
            # Check if tracking is active
            with state_lock:
                is_active = tracking_state["active"]
                current_bbox = tracking_state["target_bbox"]
            
            if is_active and current_bbox is not None:
                # Draw tracking box
                x, y, w, h = [int(v) for v in current_bbox]
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(display_frame, "TRACKING ACTIVE", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Calculate and display current command
                command = calculate_movement_command((x, y, w, h), frame.shape[1], frame.shape[0])
                cv2.putText(display_frame, f"Command: {command}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                # Draw center crosshair and target center
                frame_center_x, frame_center_y = frame.shape[1] // 2, frame.shape[0] // 2
                target_center_x, target_center_y = x + w // 2, y + h // 2
                
                # Frame center (red)
                cv2.circle(display_frame, (frame_center_x, frame_center_y), 8, (0, 0, 255), -1)
                cv2.putText(display_frame, "CENTER", (frame_center_x - 30, frame_center_y - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Target center (green)
                cv2.circle(display_frame, (target_center_x, target_center_y), 8, (0, 255, 0), -1)
                
                # Line connecting centers
                cv2.line(display_frame, (frame_center_x, frame_center_y), 
                        (target_center_x, target_center_y), (255, 0, 0), 2)
                
                # Distance indicator
                distance_text = f"Distance: {h}px (Target: {OPTIMAL_BBOX_HEIGHT}px)"
                cv2.putText(display_frame, distance_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
            else:
                # Show face detection even when not tracking
                detected_bbox = detect_closest_person(frame)
                if detected_bbox is not None:
                    x, y, w, h = detected_bbox
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(display_frame, "PERSON DETECTED (Press 's' to track)", (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # Show status
                cv2.putText(display_frame, "TRACKING INACTIVE", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Add status info
            status_text = f"Jetson: {JETSON_IP}:{PORT}"
            cv2.putText(display_frame, status_text, (10, display_frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow('KAIRA Person Tracking - Live Camera', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Start tracking closest person
                detected_bbox = detect_closest_person(frame)
                if detected_bbox is not None:
                    with state_lock:
                        tracking_state["tracker"] = create_tracker()
                        tracking_state["tracker"].init(frame, detected_bbox)
                        tracking_state["target_bbox"] = detected_bbox
                        tracking_state["active"] = True
                        tracking_state["lost_frames"] = 0
                        tracking_state["tracking_history"].clear()
                        tracking_state["frame_width"] = frame.shape[1]
                        tracking_state["frame_height"] = frame.shape[0]
                    print("✅ Tracking started from camera!")
                else:
                    print("❌ No person detected to track")
            elif key == ord('x'):
                # Stop tracking
                with state_lock:
                    tracking_state["active"] = False
                    tracking_state["tracker"] = None
                    tracking_state["target_bbox"] = None
                    tracking_state["lost_frames"] = 0
                    tracking_state["tracking_history"].clear()
                send_command("STOP")
                print("⏹ Tracking stopped from camera!")
            
            # Update tracking if active
            frame_count +=1
            if is_active and tracking_state["tracker"] is not None:
                with state_lock:
                    success, bbox = tracking_state["tracker"].update(frame)
                    if success:
                        tracking_state["target_bbox"] = bbox
                        tracking_state["lost_frames"] = 0
                        # Calculate and send command
                        if frame_count % 5 == 0: #send command every 0.1 s
                            command = calculate_movement_command(bbox, frame.shape[1], frame.shape[0])
                            print(f"Command #{frame_count}: {command}")  # <-- this line just prints locally
                            send_command(command)
                    else:
                        tracking_state["lost_frames"] += 1
                        if tracking_state["lost_frames"] > tracking_state["max_lost_frames"]:
                            # Try to re-detect
                            new_bbox = detect_closest_person(frame)
                            if new_bbox is not None:
                                tracking_state["tracker"] = create_tracker()
                                tracking_state["tracker"].init(frame, new_bbox)
                                tracking_state["lost_frames"] = 0
                                print("🔄 Target re-acquired!")
                            else:
                                send_command("STOP")
            
            time.sleep(0.03)  # ~30 FPS

            if frame_count > 999999: # prevent overflow of frame counter
                frame_count = 0
        
        cap.release()
        cv2.destroyAllWindows()
    
    # Start camera in separate thread
    camera_thread = threading.Thread(target=camera_loop, daemon=True)
    camera_thread.start()
    return camera_thread

if __name__ == "__main__":
    import uvicorn
    import threading
    
    print("🚀 Starting KAIRA Tracking Service with Ethernet...")
    print(f"📡 Jetson IP: {JETSON_IP}:{PORT}")
    print("Commands: FORWARD, BACKWARD, TURN_LEFT, TURN_RIGHT, STOP")
    print("📹 Camera controls: 's' = start tracking, 'x' = stop tracking, 'q' = quit")
    
    # Start camera display
    camera_thread = start_camera_display()
    
    # Start FastAPI server
    uvicorn.run(
        "tracking_service2_fixed:app",
        host="0.0.0.0",
        port=8003,
        reload=False,
        log_level="info"
    )