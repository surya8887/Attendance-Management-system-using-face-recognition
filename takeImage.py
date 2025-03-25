import csv
import os
import cv2
import numpy as np
import pandas as pd
import datetime
import time

def TakeImage(l1, l2, haarcascade_path, trainimage_path, message, err_screen, text_to_speech):
    if not l1 and not l2:
        t = 'Please enter your Enrollment Number and Name.'
        text_to_speech(t)
        return
    elif not l1:
        t = 'Please enter your Enrollment Number.'
        text_to_speech(t)
        return
    elif not l2:
        t = 'Please enter your Name.'
        text_to_speech(t)
        return
    
    try:
        cam = cv2.VideoCapture(0)
        detector = cv2.CascadeClassifier(haarcascade_path)
        
        enrollment = l1.strip()
        name = l2.strip()
        sample_num = 0
        directory = f"{enrollment}_{name}"
        path = os.path.join(trainimage_path, directory)
        
        if not os.path.exists(path):
            os.mkdir(path)
        else:
            text_to_speech("Student Data already exists")
            return
        
        while True:
            ret, img = cam.read()
            if not ret:
                text_to_speech("Failed to capture image from camera")
                break
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                sample_num += 1
                img_path = os.path.join(path, f"{name}_{enrollment}_{sample_num}.jpg")
                cv2.imwrite(img_path, gray[y:y + h, x:x + w])
                cv2.imshow("Frame", img)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            elif sample_num >= 50:
                break
        
        cam.release()
        cv2.destroyAllWindows()
        
        row = [enrollment, name]
        student_details_path = "StudentDetails/studentdetails.csv"
        os.makedirs(os.path.dirname(student_details_path), exist_ok=True)
        
        with open(student_details_path, "a", newline="") as csvFile:
            writer = csv.writer(csvFile, delimiter=",")
            writer.writerow(row)
        
        res = f"Images Saved for ER No: {enrollment}, Name: {name}"
        message.configure(text=res)
        text_to_speech(res)
    
    except Exception as e:
        text_to_speech(f"Error: {str(e)}")
