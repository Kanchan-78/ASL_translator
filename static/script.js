const body = document.body;
body.classList.add('dark');

let sessionId = localStorage.getItem("sessionId");
if (!sessionId) {
    sessionId = crypto.randomUUID();
    localStorage.setItem("sessionId", sessionId);
}

const video = document.getElementById('video');
const videoContainer = document.getElementById('videoContainer');
const videoStatus = document.getElementById('videoStatus');
const detectedText = document.getElementById('detectedText');
const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');
const copyButton = document.getElementById('copyButton');

let canvas = document.createElement('canvas');
let ctx = canvas.getContext('2d');

let processedFeed = document.createElement('img');
processedFeed.id = 'processedFeed';
processedFeed.style.position = 'absolute';
processedFeed.style.top = '0';
processedFeed.style.left = '0';
processedFeed.style.width = '100%';
processedFeed.style.height = '100%';
processedFeed.style.objectFit = 'cover';
processedFeed.style.display = 'none';
processedFeed.style.borderRadius = '12px';
videoContainer.appendChild(processedFeed);

let captureInterval;
let stream = null;
let frameCounter = 0;

async function startCapture() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        await video.play();

        video.classList.add('active');
        videoContainer.classList.add('capturing');
        videoStatus.style.display = "none"; 
        processedFeed.style.display = 'block';
        
        startButton.disabled = true;
        stopButton.disabled = false;

        captureInterval = setInterval(() => {
            if (video.readyState < 2) return;

            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            frameCounter++;
            if (frameCounter % 2 === 0) {
                canvas.toBlob(async blob => {
                    const formData = new FormData();
                    formData.append("file", blob, "frame.jpg");
                    formData.append("session_id", sessionId);

                    try {
                        const response = await fetch("/process_frame", {
                            method: "POST",
                            body: formData
                        });
                        
                        if(response.ok){
                            const data = await response.json();
                            
                            if(data.sentence !== undefined) {
                                detectedText.value = data.sentence;
                            }

                            if (data.image) {
                                processedFeed.src = 'data:image/jpeg;base64,' + data.image;
                            }
                        }
                    } catch (err) {
                        console.error("Frame send error:", err);
                    }
                }, 'image/jpeg', 0.5);
            }
        }, 1000 / 24); // ~24 FPS

    } catch (err) {
        console.error("Camera access error:", err);
        showToast("Camera access denied", "error");
    }
}

function stopCapture() {
    clearInterval(captureInterval);
    
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    
    video.srcObject = null;
    video.classList.remove('active');
    videoContainer.classList.remove('capturing');
    videoStatus.style.display = "block";
    processedFeed.style.display = 'none'; // Hide the processed feed
    processedFeed.src = ""; // Clear memory
    
    startButton.disabled = false;
    stopButton.disabled = true;
}

startButton.addEventListener('click', startCapture);
stopButton.addEventListener('click', stopCapture);

copyButton.addEventListener('click', () => {
    const text = detectedText.value.trim();
    if (text) {
        navigator.clipboard.writeText(text);
        showToast("Text copied to clipboard!", "success");
    } else {
        showToast("No text to copy", "error");
    }
});

function showToast(message, type = 'success') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast ${type} show`;
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}