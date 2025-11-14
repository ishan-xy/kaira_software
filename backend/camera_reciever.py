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
    print(f"Stopping publisher for topic: {topic.decode()}")


async def run(pc, context):
    track_count = 0
    
    publisher = context.socket(zmq.PUB)
    publisher.set_hwm(1)
    publisher.bind(socket_url)
    print(f"ZMQ Publisher bound to {socket_url}")

    @pc.on("track")
    def on_track(track):
        nonlocal track_count
        if track.kind == "video":
            print(f"Received track {track_count}")
            topic = f"camera_{track_count}".encode()
            asyncio.create_task(publish_video(track, topic, publisher))
            track_count += 1

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        print(f"ICE Connection State is {pc.iceConnectionState}")
        if pc.iceConnectionState == "failed":
            await pc.close()
            pcs.discard(pc)
        if pc.iceConnectionState == "closed":
            pcs.discard(pc)


    pc.addTransceiver("video", direction="recvonly")
    
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    body = {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    
    print("Attempting to connect to server...")
    session = aiohttp.ClientSession()
    try:
        async with session.post(
            f"{camera_server_url}/offer", json=body
        ) as response:
            if response.status != 200:
                print(f"Server error: {response.status}")
                data = await response.text()
                print(f"Server response: {data}")
                await session.close()
                return

            print("Server connected, processing answer...")
            data = await response.json()
            answer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
            await pc.setRemoteDescription(answer)
            print("Remote description set.")
            
    except aiohttp.ClientConnectorError:
        print("Could not connect to server. Is serve_camera.py running?")
        await session.close()
        return
    except Exception as e:
        print(f"An error occurred during connection: {e}")
        await session.close()
        return
    
    await session.close()
    
    try:
        while pc in pcs:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        print("Run task cancelled.")

pcs = set()

async def main():
    zmq_context = zmq.asyncio.Context()
    pc = RTCPeerConnection()
    pcs.add(pc)
    
    try:
        await run(pc, zmq_context)
    except asyncio.CancelledError:
        print("Main task cancelled by user.")
    finally:
        print("Closing connections...")
        coros = [p.close() for p in pcs]
        await asyncio.gather(*coros)
        zmq_context.term()
        print("Cleanup complete.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopping...")