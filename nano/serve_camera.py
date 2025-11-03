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
    def __init__(self, camera_index):
        super().__init__()
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"Error: Could not open video capture {self.camera_index}")
            raise Exception(f"Could not open video capture {self.camera_index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.q = queue.Queue(maxsize=1)
        self.running = True

        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        """
        Continuously reads frames from the camera in a separate thread.
        Keeps only the latest frame in the queue.
        """
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print(f"Camera {self.camera_index} read failed. Stopping thread.")
                break
            
            # If the queue is full, discard the old frame
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            
            # Put the new, latest frame in the queue
            self.q.put(frame)
        
        # Signal that the thread is stopping
        self.q.put(None)

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        
        # Get the latest frame from the queue in an async-friendly way
        # This runs self.q.get() in a separate thread
        frame = await asyncio.to_thread(self.q.get)

        if frame is None:
            # Thread has stopped, stop the track
            self.stop()
            # Return a black frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        frame_av = VideoFrame.from_ndarray(frame, format="bgr24")
        frame_av.pts = pts
        frame_av.time_base = time_base
        return frame_av

    def __del__(self):
        self.stop()

    def stop(self):
        if self.running:
            self.running = False
            if self.cap.isOpened():
                self.cap.release()
            self.thread.join(timeout=1.0)

pcs = set()

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        if pc.iceConnectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    webcam_track_0 = WebcamTrack(camera_index=0)
    webcam_track_1 = WebcamTrack(camera_index=1)
    
    pc.addTrack(webcam_track_0)
    pc.addTrack(webcam_track_1)

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

if __name__ == "__main__":
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_post("/offer", offer)
    
    # Get the event loop and set our custom exception handler
    loop = asyncio.get_event_loop()
    loop.set_exception_handler(handle_async_exception)
    
    print("Starting server on http://localhost:8080 (with custom exception handler)")
    web.run_app(app, host="0.0.0.0", port=8080)