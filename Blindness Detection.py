#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2

# Load the pre-trained Haar cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Load the image
img = cv2.imread('eye_image.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect eyes in the grayscale image
eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Check if eyes are detected
if len(eyes) > 0:
    print("Eyes detected!")
else:
    print("No eyes detected.")

# Display the image with rectangles around the detected eyes
for (x, y, w, h) in eyes:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the image
cv2.imshow('Blindness Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




