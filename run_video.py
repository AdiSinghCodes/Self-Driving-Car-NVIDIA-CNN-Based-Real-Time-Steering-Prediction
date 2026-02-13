import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from PIL import Image
import numpy as np
import model
import cv2

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "save/model.ckpt")

img = cv2.imread('steering_wheel_image.jpg',0)
rows,cols = img.shape

smoothed_angle = 0

# Video file (downloaded using download_video.py)
video_path = "driving_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file. Make sure the video file exists!")
    exit()

print("Playing video... Press 'q' to quit")

while(cv2.waitKey(10) != ord('q')):
    ret, frame = cap.read()
    
    # If video ends, restart from beginning
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    
    image = np.array(Image.fromarray(frame).resize((200, 66))) / 255.0
    degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180 / np.pi
    print("Predicted steering angle: " + str(degrees) + " degrees")
    
    # Show the video frame
    cv2.imshow('Video', frame)
    
    # Rotate steering wheel based on prediction
    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imshow("Steering Wheel", dst)

cap.release()
cv2.destroyAllWindows()
