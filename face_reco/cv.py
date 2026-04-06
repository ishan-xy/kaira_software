import cv2
import zmq
import numpy as np
import json
import time
import threading

try:
    from facereco import process_identity_from_frame
    FACE_RECO_LOADED = True
except ImportError:
    print("Could not load face_reco.facereco. Running display only.")
    FACE_RECO_LOADED = False

socket_url = "tcp://127.0.0.1:5555"
identity_pub_url = "tcp://127.0.0.1:5558"

latest_frame = None
latest_frame_lock = threading.Lock()
running = True

current_known = "Unknown"
last_seen_identity = "Unknown"
consecutive_seen_count = 0
consecutive_unknown_count = 0

CONFIRM_NEW_FRAMES = 3
CONFIRM_LOST_FRAMES = 5

identity_publisher = None 

def identity_worker():
    global current_known, last_seen_identity, consecutive_seen_count, consecutive_unknown_count
    global latest_frame, latest_frame_lock, running
    print("Identity worker thread started...")
    
    last_published_identity = None 
    
    while running:
        frame_to_process = None
        with latest_frame_lock:
            if latest_frame is not None:
                frame_to_process = latest_frame.copy()
                
        if frame_to_process is not None:
            identity = process_identity_from_frame(frame_to_process) #type: ignore
            
            if identity != "Unknown":
                consecutive_unknown_count = 0
                if identity == last_seen_identity:
                    consecutive_seen_count += 1
                else:
                    last_seen_identity = identity
                    consecutive_seen_count = 1
                    
                if consecutive_seen_count >= CONFIRM_NEW_FRAMES and current_known != identity:
                    print(f"--- Identified: {identity} ---")
                    current_known = identity
            else:
                consecutive_seen_count = 0
                consecutive_unknown_count += 1
                if consecutive_unknown_count >= CONFIRM_LOST_FRAMES and current_known != "Unknown":
                    print(f"--- Identity lost (was {current_known}) ---")
                    current_known = "Unknown"
                    last_seen_identity = "Unknown"
            
            if identity_publisher and current_known != last_published_identity:
                try:
                    message = json.dumps({
                        "identity": current_known,
                        "timestamp": time.time()
                    }).encode('utf-8')
                    
                    identity_publisher.send_multipart([b"current_identity", message])
                    last_published_identity = current_known
                    print(f"Published identity: {current_known}")
                except Exception as e:
                    print(f"Error publishing identity: {e}")
                    
        else:
            time.sleep(0.05)
            
        time.sleep(0.05)

def main():
    global latest_frame, latest_frame_lock, running, identity_publisher
    
    context = zmq.Context()
    
    frame_socket = context.socket(zmq.SUB)
    print("Connecting to ZMQ frame publisher...")
    frame_socket.connect(socket_url)
    frame_socket.subscribe(b"camera_0")
    print("Subscribed to 'camera_0'. Waiting for frames...")
    
    identity_publisher = context.socket(zmq.PUB)
    identity_publisher.bind(identity_pub_url)
    print(f"Identity publisher bound to '{identity_pub_url}'")
    
    if FACE_RECO_LOADED:
        worker = threading.Thread(target=identity_worker, daemon=True)
        worker.start()
        
    latencies = []
    last_print_time = time.time()
    
    try:
        while True:
            topic, meta_json, img_bytes = frame_socket.recv_multipart()
            recv_time = time.time()
            
            meta = json.loads(meta_json.decode())
            frame = np.frombuffer(img_bytes, dtype=meta['dtype']).reshape(meta['shape']).copy()
            
            send_time = meta.get('send_time', recv_time)
            if send_time > 1e9:
                latency_ms = (recv_time - send_time) * 1000
                latencies.append(latency_ms)
            
            current_time = time.time()
            if current_time - last_print_time >= 1.0:
                if latencies:
                    avg_latency = np.mean(latencies)
                    print(f"ZMQ Latency (avg over 1s): {avg_latency:.2f} ms")
                    latencies = []
                last_print_time = current_time
                
            with latest_frame_lock:
                latest_frame = frame
                
            cv2.imshow("CV Script - Camera 1 (Live)", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStopping...")
        
    finally:
        running = False
        if FACE_RECO_LOADED:
            worker.join() #type: ignore
            
        cv2.destroyAllWindows()
        frame_socket.close()
        identity_publisher.close()
        context.term()
        print("Stopped.")

if __name__ == "__main__":
    main()