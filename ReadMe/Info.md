
---

### prepare_dataset

---

**collection_img.py**

As this project uses our own created dataset, where collection_img python script comes into play to capture 1k images per class so that we can easily capture every class images easily then do further processing.

This script works using open cv python library for capturing images from camera source of host device, where all captured images are first flipped then that converted image gets converted into RGB format; where the PIL library expect RGB instead of other format then creats a PIL image object from numpy array which allows working with image using PIL-compatible libraries. In here for GUI customtkinter has been used intead of regular tkinter which is an open source Python GUI library. It creates 1k images for every class needed to be predicted.

---

**create_dataset.py**

After we finish collecting images, it is then processed where every image inside the dataset gets hand detected using Mediapipe library, then from the detected hand 21 landmark points were extracted from every hand images which converts those points by shifting all points so the hand start at position (0,0) into a simple list of normalized x and y coordinates that makes every sample consistent no matter where the hand appers because all of the coordinates starts from (0,0) for every image it stores all these clean landmark values along with the gesture label after doing it for whole complete classes of image it then saves the whole data into a single file named as `dataset.pickle` which will be used later to train the ML model.

---


### model_train

---

**model_train.py**

As the dataset pickel file gets ready we then move toward the model creation part, where the script loads the pickel file containing hand-landmark dataset it splits into 2 section training set and testing set, then trains a random forest machine-learning model to recognize each gesture. It takes all the landmark coordinates for every image along with their labels, learns the patterns during training, and then checks how well the model performs by testing it on data it has never seen before. After calculating and printing the accuracy percentage, it finally saves the trained model into a file (`model.p`) so it can be used later in a real-time gesture-recognition program.

---

### random forest algorithim ML

Random Forest works by creating many small decision trees, where each tree learns a slightly different way to recognize a gesture. When you give it new landmark data, all the trees vote on what gesture they think it is, and the most common answer becomes the final prediction. 

It simply learns patterns from the 42 landmark numbers and reliably tells gestures apart without needing deep learning.

---

### main.py

After that the backend small web server which receives camera frames from the browser, finds your hand using MediaPipe, extracts landmark points, and sends those points into a pre-trained model to figure out which letter (A, B, C, etc.) you are showing. It keeps track of each user using a session ID so it can remember the ongoing sentence they are forming. When the hand is detected, the code draws the landmarks on the image, predicts the letter, shows it on the screen, and if you hold the same letter for a couple of seconds, it adds that letter to your sentence (or deletes when the “Bk” sign is shown).

Finally, it sends the processed image and the updated sentence back to the browser so the interface updates in real time.

Even if it is being shown that the video is being feed into ml model its not, because video is the sequence of images where in this case user sends 24 frames per second only for backend to process faster with better result even in low end devices.

---

### script.js

This code turns on your webcam, captures video frames, and sends some of those frames to the server to be processed. It creates a unique session ID for each user so your sentence can be tracked. Every time a frame is captured, the image is drawn onto a hidden canvas, converted into a JPEG blob, and uploaded to the server. The server sends back the predicted letter and a processed image, which replaces the live video feed with the version that has the hand landmarks drawn. As new letters come in, the text box updates with the sentence you are forming. The Start button begins the camera capture, the Stop button turns it off, and the Copy button copies the detected sentence to your clipboard.

---