import asyncio
import json
import logging
import threading
import queue
import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription

logger = logging.getLogger(__name__)

class WebRTCClient:
    def __init__(self, audio_queue: queue.Queue, server_url="http://localhost:8082/offer"):
        self.server_url = server_url
        self.audio_queue = audio_queue
        self.pc = None
        self.loop = None
        self.thread = None
        self._is_running = False
        self.loop_ready_event = threading.Event()

    def _start_event_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop_ready_event.set()
        try:
            self.loop.run_forever()
        finally:
            self.loop.close()

    def start(self):
        if self._is_running:
            return
        self._is_running = True
        self.thread = threading.Thread(target=self._start_event_loop, daemon=True)
        self.thread.start()
        self.loop_ready_event.wait()
        asyncio.run_coroutine_threadsafe(self.connect(), self.loop)

    def stop(self):
        if not self._is_running or not self.loop:
            return
        self._is_running = False
        
        if self.pc and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self.pc.close(), self.loop)
        
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        
        self.thread.join(timeout=1.0)
        logger.info("WebRTC client stopped.")

    async def connect(self):
        while self._is_running:
            try:
                self.pc = RTCPeerConnection()
                
                logger.info("WebRTC: Creating data channel 'audio'")
                channel = self.pc.createDataChannel("audio")
                channel.binaryType = "arraybuffer"

                @channel.on("open")
                def on_open():
                    logger.info("WebRTC: Data Channel is OPEN.")
                    
                @channel.on("message")
                def on_message(message):
                    if isinstance(message, bytes):
                        self.audio_queue.put(message)
                
                @channel.on("close")
                def on_close():
                    logger.warn("WebRTC: Data Channel CLOSED.")

                @self.pc.on("connectionstatechange")
                async def on_connectionstatechange():
                    logger.info(f"WebRTC: Connection state is {self.pc.connectionState}")
                    if self.pc.connectionState == "failed" or self.pc.connectionState == "closed":
                        await self.pc.close()

                logger.info("WebRTC: Attempting to connect (creating offer)...")
                
                offer = await self.pc.createOffer()
                await self.pc.setLocalDescription(offer)
                
                payload = {
                    "sdp": self.pc.localDescription.sdp,
                    "type": self.pc.localDescription.type,
                }

                async with aiohttp.ClientSession() as session:
                    async with session.post(self.server_url, json=payload) as resp:
                        if resp.status != 200:
                            raise Exception(f"Signaling server returned {resp.status}")
                        answer_data = await resp.json()
                        answer = RTCSessionDescription(sdp=answer_data["sdp"], type=answer_data["type"])
                
                await self.pc.setRemoteDescription(answer)
                logger.info("WebRTC: Connection established (answer received).")
                
                return 

            except Exception as e:
                logger.error(f"WebRTC connection failed: {e}. Retrying in 3s...")
                if self.pc:
                    await self.pc.close()
                await asyncio.sleep(3)