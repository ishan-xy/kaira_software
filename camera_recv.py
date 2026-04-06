import asyncio
import json
import aiohttp
import numpy as np
import zmq.asyncio
from aiortc import RTCPeerConnection, RTCSessionDescription

socket_url = "tcp://127.0.0.1:5555"
camera_server_url = "http://localhost:9000"

def create_frame_message(frame, topic, send_time):
    """Create a ZMQ message with frame data and metadata."""
    meta = dict(
        dtype=str(frame.dtype),
        shape=frame.shape,
        send_time=send_time
    )
    meta_json = json.dumps(meta).encode()
    return [topic, meta_json, frame.tobytes()]

async def publish_video(track, topic, socket):
    """Continuously receive frames from WebRTC track and publish to ZMQ."""
    print(f"📹 Starting publisher for topic: {topic.decode()}")
    frame_count = 0
    try:
        while True:
            try:
                frame = await track.recv()
                img = frame.to_ndarray(format="bgr24")
                send_time = asyncio.get_event_loop().time()
                message = create_frame_message(img, topic, send_time)
                await socket.send_multipart(message)
                
                frame_count += 1
                if frame_count % 1000 == 0:
                    print(f"✅ Published {frame_count} frames on {topic.decode()}")
                    
            except Exception as e:
                if "closed" in str(e).lower():
                    print(f"🔌 Track {topic.decode()} closed")
                    break
                print(f"⚠️  Error receiving frame on {topic.decode()}: {e}")
                break
                
    except asyncio.CancelledError:
        print(f"🛑 Publisher task cancelled for {topic.decode()}")
    finally:
        print(f"🏁 Publisher stopped for {topic.decode()}")

async def run(pc, context):
    """Main function to set up WebRTC connection and start publishing."""
    track_count = 0
    tasks = []
    
    # Create ZMQ publisher socket
    publisher = context.socket(zmq.PUB)
    publisher.set_hwm(1)  # Keep only the last message to prevent memory buildup
    publisher.bind(socket_url)
    print(f"🔌 ZMQ Publisher bound to {socket_url}")

    @pc.on("track")
    def on_track(track):
        """Called when a new video track is received from WebRTC."""
        nonlocal track_count
        if track.kind == "video":
            topic = f"camera_{track_count}".encode()
            print(f"🎥 Received video track {track_count}")
            task = asyncio.create_task(publish_video(track, topic, publisher))
            tasks.append(task)
            track_count += 1

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        """Monitor ICE connection state."""
        print(f"🔗 ICE Connection State: {pc.iceConnectionState}")
        if pc.iceConnectionState == "failed":
            print("❌ ICE connection failed!")
            await pc.close()

    # Request only ONE video track (since server only has one camera)
    pc.addTransceiver("video", direction="recvonly")
    
    # Create and send offer
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    body = {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    
    print(f"📡 Connecting to camera server at {camera_server_url}...")
    
    session = aiohttp.ClientSession()
    try:
        async with session.post(
            f"{camera_server_url}/offer", json=body
        ) as response:
            if response.status != 200:
                print(f"❌ Server error: HTTP {response.status}")
                await session.close()
                return

            data = await response.json()
            answer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
            await pc.setRemoteDescription(answer)
            print("✅ Connected to camera server!")
            
    except aiohttp.ClientConnectorError as e:
        print(f"❌ Could not connect to server: {e}")
        await session.close()
        return
    except Exception as e:
        print(f"❌ Error during connection: {e}")
        await session.close()
        return
    finally:
        await session.close()
    
    # Wait for tasks to complete or be cancelled
    try:
        print("🎬 Camera streaming started. Press Ctrl+C to stop.")
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        print("🛑 Stopping all publisher tasks...")
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        # Clean up publisher socket
        publisher.close()
        print("🧹 Publisher socket closed")

async def main():
    """Main entry point with proper cleanup."""
    zmq_context = zmq.asyncio.Context()
    pc = RTCPeerConnection()
    
    try:
        await run(pc, zmq_context)
    except KeyboardInterrupt:
        print("\n⚡ Interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        print("🔌 Closing peer connection...")
        await pc.close()
        print("🧹 Terminating ZMQ context...")
        zmq_context.term()
        print("✅ Cleanup complete")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✅ Shutdown complete")