import asyncio
import json
import cv2
import aiohttp
import numpy as np
import zmq
import zmq.asyncio
import time
from aiortc import RTCPeerConnection, RTCSessionDescription

socket_url = "ipc:///tmp/camera_stream"
camera_server_url = "http://localhost:8080"

def create_frame_message(frame, topic, send_time):
    meta = dict(
        dtype=str(frame.dtype),
        shape=frame.shape,
        send_time=time.time()
    )
    meta_json = json.dumps(meta).encode()
    return [topic, meta_json, frame.tobytes()]

async def publish_video(track, topic, socket):
    print(f"Starting publisher for topic: {topic.decode()}")
    while True:
        try:
            frame = await track.recv()
            img = frame.to_ndarray(format="bgr24")
            send_time = time.time()
            message = create_frame_message(img, topic, send_time)
            await socket.send_multipart(message)
            
        except Exception as e:
            print(f"Error in publish_video ({topic.decode()}): {e}")
            break

async def run(pc, context):
    track_count = 0
    
    publisher = context.socket(zmq.PUB)
    publisher.set_hwm(1)  # Keep only the last message
    publisher.bind(socket_url)
    print(f"ZMQ Publisher bound to {socket_url}")

    @pc.on("track")
    def on_track(track):
        nonlocal track_count
        if track.kind == "video":
            topic = f"camera_{track_count}".encode()
            asyncio.create_task(publish_video(track, topic, publisher))
            track_count += 1

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        if pc.iceConnectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    pc.addTransceiver("video", direction="recvonly")
    pc.addTransceiver("video", direction="recvonly")
    
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    body = {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    
    session = aiohttp.ClientSession()
    try:
        async with session.post(
            f"{camera_server_url}/offer", json=body
        ) as response:
            if response.status != 200:
                print("Server error")
                await session.close()
                return

            data = await response.json()
            answer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
            await pc.setRemoteDescription(answer)
    except aiohttp.ClientConnectorError:
        print("Could not connect to server.")
        await session.close()
        return
    
    await session.close()
    await asyncio.Event().wait()

pcs = set()

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    zmq_context = zmq.asyncio.Context()
    try:
        pc = RTCPeerConnection()
        pcs.add(pc)
        loop.run_until_complete(run(pc, zmq_context))
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        print("Closing connection")
        coros = [pc.close() for pc in pcs]
        loop.run_until_complete(asyncio.gather(*coros))
        zmq_context.term()