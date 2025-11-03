let socket = new WebSocket("ws://localhost:8001");
let displayDiv = document.getElementById('textDisplay');
let aiDisplayDiv = document.getElementById('aiResponseDisplay');
let server_available = false;
let mic_available = false;
let fullSentences = [];

const serverCheckInterval = 5000;

function displayAIResponse(responseText) {
    if (aiDisplayDiv) {
        let p = document.createElement('p');
        p.innerHTML = `<strong>🤖 AI:</strong> ${responseText}`; 
        aiDisplayDiv.appendChild(p);
        aiDisplayDiv.scrollTop = aiDisplayDiv.scrollHeight; 
    } else {
        console.warn("Element with id 'aiResponseDisplay' not found.");
    }
}

function handleSocketMessage(event) {
    let data = JSON.parse(event.data);

    if (data.type === 'realtime') {
        displayRealtimeText(data.text, displayDiv);
    } else if (data.type === 'fullSentence') {
        fullSentences.push(data.text);
        displayRealtimeText("", displayDiv);
    } else if (data.type === 'aiResponse') {
        displayAIResponse(data.text);
    }
}

function connectToServer() {
    socket = new WebSocket("ws://localhost:8001");

    socket.onopen = function() {
        server_available = true;
        start_msg();
    };

    socket.onmessage = handleSocketMessage; 

    socket.onclose = function() {
        server_available = false;
    };
}

socket.onmessage = handleSocketMessage;

function displayRealtimeText(realtimeText, displayDiv) {
    let displayedText = fullSentences.map((sentence, index) => {
        let span = document.createElement('span');
        span.textContent = sentence + " ";
        span.className = index % 2 === 0 ? 'yellow' : 'cyan';
        return span.outerHTML;
    }).join('') + realtimeText;

    displayDiv.innerHTML = displayedText;
}

function start_msg() {
    if (!mic_available)
        displayRealtimeText("🎤  please allow microphone access  🎤", displayDiv);
    else if (!server_available)
        displayRealtimeText("🖥️  please start server  🖥️", displayDiv);
    else
        displayRealtimeText("👄  start speaking  👄", displayDiv);
}

setInterval(() => {
    if (!server_available) connectToServer();
}, serverCheckInterval);

start_msg();

socket.onopen = function() {
    server_available = true;
    start_msg();
};

navigator.mediaDevices.getUserMedia({ audio: true })
.then(stream => {
    let audioContext = new AudioContext();
    let source = audioContext.createMediaStreamSource(stream);
    let processor = audioContext.createScriptProcessor(256, 1, 1);

    source.connect(processor);
    processor.connect(audioContext.destination);
    mic_available = true;
    start_msg();

    processor.onaudioprocess = function(e) {
        let inputData = e.inputBuffer.getChannelData(0);
        let outputData = new Int16Array(inputData.length);

        for (let i = 0; i < inputData.length; i++) {
            outputData[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
        }

        if (socket.readyState === WebSocket.OPEN) {
            let metadata = JSON.stringify({ sampleRate: audioContext.sampleRate });
            let metadataBytes = new TextEncoder().encode(metadata);
            let metadataLength = new ArrayBuffer(4);
            let metadataLengthView = new DataView(metadataLength);
            metadataLengthView.setInt32(0, metadataBytes.byteLength, true);
            let combinedData = new Blob([metadataLength, metadataBytes, outputData.buffer]);
            socket.send(combinedData);
        }
    };
})
.catch(e => console.error(e));
