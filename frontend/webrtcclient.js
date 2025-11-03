const pc = new RTCPeerConnection({
    iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] 
});
const statusEl = document.getElementById('status');

let audioContext;
let nextTime = 0;
const geminiSampleRate = 24000;
const webrtcRetryDelay = 3000; // 3 seconds

function initAudio() {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: geminiSampleRate 
        });
        nextTime = audioContext.currentTime;
        statusEl.textContent = 'Status: ✅ AudioContext initialized. Data Channel open.';
    }
}

async function processAudioChunk(arrayBuffer) {
    if (arrayBuffer.byteLength === 0) return;

    // 1. Convert s16le (2 bytes) to f32 (-1.0 to 1.0)
    const int16View = new Int16Array(arrayBuffer);
    
    // --- THIS WAS THE TYPO ---
    const float32Array = new Float32Array(int16View.length);
    // --- END OF FIX ---
    
    for (let i = 0; i < int16View.length; i++) {
        float32Array[i] = int16View[i] / 32768.0; 
    }

    // 2. Create an AudioBuffer
    const audioBuffer = audioContext.createBuffer(
        1, // 1 channel (mono)
        float32Array.length, // number of samples
        geminiSampleRate // sample rate
    );
    audioBuffer.getChannelData(0).set(float32Array);

    // 3. Schedule it to play
    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContext.destination);

    const playTime = Math.max(audioContext.currentTime, nextTime);
    source.start(playTime);
    
    nextTime = playTime + audioBuffer.duration;
}

pc.oniceconnectionstatechange = () => {
    const state = pc.iceConnectionState;
    if (state !== 'checking') {
        statusEl.textContent = `Status: ICE Connection State - ${state}`;
    }
    if (state === 'failed' || state === 'disconnected' || state === 'closed') {
        statusEl.textContent = `Status: ❌ Connection Closed. State: ${state}`;
    }
};

async function startConnection() {
    try {
        document.getElementById('start-button').disabled = true;
        
        // 1. Client creates the data channel
        const data_channel = pc.createDataChannel("audio");
        data_channel.binaryType = 'arraybuffer';

        // 2. We attach the listeners to this new channel object
        data_channel.onopen = () => {
            initAudio(); // Initialize audio context on open
        };

        data_channel.onmessage = (event) => {
            if (event.data instanceof ArrayBuffer) {
                processAudioChunk(event.data);
            }
        };

        data_channel.onclose = () => {
            statusEl.textContent = 'Status: 🏁 Audio stream finished.';
        };
        
        statusEl.textContent = 'Status: Creating WebRTC Offer...';
        
        // 3. Now, createOffer() will include the data channel
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);

        statusEl.textContent = 'Status: Sending Offer to backend (port 8081)...';

        const response = await fetch('http://localhost:8081/offer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                sdp: pc.localDescription.sdp,
                type: pc.localDescription.type,
            }),
        });

        const answer = await response.json();
        statusEl.textContent = 'Status: Received Answer, establishing connection...';
        
        // 4. This setRemoteDescription should now work
        await pc.setRemoteDescription(new RTCSessionDescription(answer));
        
    } catch (error) {
        console.error('WebRTC setup failed:', error);
        statusEl.textContent = `Status: ❌ Error: ${error.message}`;
        document.getElementById('start-button').disabled = false;
    }
}

document.getElementById('start-button').onclick = startConnection;