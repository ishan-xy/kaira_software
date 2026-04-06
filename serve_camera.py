import asyncio
import json
import cv2
import numpy as np
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame
import threading
import queue

class WebcamTrack(VideoStreamTrack):
    def __init__(self, camera_index, width=1280, height=720):
        super().__init__()
        self.running = True 
        self.camera_index = camera_index
        self.camera_name = f"Camera Index {self.camera_index}" 
        
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"Error: Could not open video capture {self.camera_index}")
            self.running = False
            raise Exception(f"Could not open video capture {self.camera_index}")
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera {camera_index}: Requested {width}x{height}, got {actual_width}x{actual_height}")
        
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.frame_ready = threading.Event()

        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        
        # Wait for first frame (with timeout)
        if not self.frame_ready.wait(timeout=5.0):
            self.stop()
            raise Exception(f"Camera {camera_index} failed to capture first frame")
        print(f"Camera {camera_index} ready!")

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print(f"Camera {self.camera_index} read failed. Stopping thread.")
                with self.frame_lock:
                    self.latest_frame = None
                break
            
            # Simply overwrite with latest frame
            with self.frame_lock:
                self.latest_frame = frame
                if not self.frame_ready.is_set():
                    self.frame_ready.set()

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        # Get the latest frame
        with self.frame_lock:
            frame = self.latest_frame

        if frame is None:
            print(f"No frame available for camera {self.camera_index}")
            self.stop()
            # Use actual dimensions from camera
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self.cap.isOpened() else 640
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self.cap.isOpened() else 480
            frame = np.zeros((actual_height, actual_width, 3), dtype=np.uint8)

        frame_av = VideoFrame.from_ndarray(frame, format="bgr24") # type: ignore
        frame_av.pts = pts
        frame_av.time_base = time_base
        return frame_av

    def stop(self):
        if self.running:
            self.running = False
            if self.cap.isOpened():
                self.cap.release()
            if hasattr(self, 'thread') and self.thread.is_alive():
                self.thread.join(timeout=1.0)

    def __del__(self):
        self.stop()

pcs = set()

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        print(f"ICE Connection State is {pc.iceConnectionState}")
        if pc.iceConnectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    try:
        webcam_track_0 = WebcamTrack(camera_index=0)
        pc.addTrack(webcam_track_0)
        print(f"Successfully added camera: {webcam_track_0.camera_name}")
    except Exception as e:
        print(f"Failed to add webcam 0: {e}")
        await pc.close()
        pcs.discard(pc)
        return web.Response(status=500, text=f"Failed to open camera 0: {e}")

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )

async def on_shutdown(app):
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

def handle_async_exception(loop, context):
    exception = context.get("exception")
    
    if isinstance(exception, asyncio.exceptions.InvalidStateError):
        print("\n[HANDLED] Suppressed known 'InvalidStateError' from aioice.\n"
              "         This is a non-fatal race condition. Server continues.\n")
    else:
        print(f"Caught unhandled async exception: {context.get('message')}")
        if exception:
            print(f"Exception: {exception}")

async def main():
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_post("/offer", offer)
    
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(handle_async_exception)
    
    print("Starting server on http://localhost:9000 (with custom exception handler)")
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 9000)
    await site.start()
    
    print("======== Running on http://0.0.0.0:9000 ========")
    
    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopping server...")