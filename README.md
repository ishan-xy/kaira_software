# KAIRA — Knowledge-based Articulated Intelligent Robotic Assistant

RAG + Face Recognition + Real-Time Audio-to-Audio Conversational Pipeline built on a distributed microservices architecture.

KAIRA is a real-time multimodal robotic assistant built for interactive tech showcases and human interaction at our techno-cultural fest. It serves as the conversational face of the robot, capable of recognizing important staff members, answering questions about them, and interacting naturally using live voice responses.

The system is designed around low-latency communication and real-time streaming between independent services.

---
<img width="1852" height="2470" alt="ea3b89d3-488b-46a3-8686-c86559ba5388" src="https://github.com/user-attachments/assets/97a1c596-71f4-4b2b-ae02-1a730eda8909" />
---
## Architecture

### Camera Pipeline

* A camera service captures the live feed
* Frames are streamed to a `camera_recv` service using WebRTC
* `camera_recv` publishes frames over a ZMQ TCP stream

### Computer Vision Service

* Subscribes to the camera stream
* Performs real-time face recognition using stored facial embeddings
* Publishes detected identities (`person name` or `unknown`) over another ZMQ stream

### RAG Pipeline

* Important documents and staff information are converted into vector embeddings
* Relevant context is retrieved dynamically during conversations

### Speech & Conversation Flow

* UI service captures user speech
* STT service converts speech to text using Whisper
* Live API service receives the text input
* Retrieved RAG context + system instructions are sent to Gemini Live API
* Responses are received as streamed chunks

Instead of waiting for the full response, chunks are streamed back instantly to the UI using WebRTC, enabling low-latency natural voice playback.

## Features

* Real-time face recognition
* Live conversational AI interaction
* Retrieval-Augmented Generation (RAG)
* Streaming audio responses
* Distributed microservices architecture
* WebRTC-based low-latency communication
* ZMQ event-driven communication pipeline

## Tech Stack

* Python
* WebRTC
* ZeroMQ (ZMQ)
* OpenCV
* Whisper
* Gemini Live API
* Vector Embeddings / RAG
