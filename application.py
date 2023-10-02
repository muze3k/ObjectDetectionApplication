print('Starting application please wait')

import tkinter as tk
import customtkinter as ctk
import math

import os
import numpy as np

from ultralytics import YOLO
import cv2

from PIL import Image, ImageTk
from functools import partial

#import vlc

app = tk.Tk()
app.geometry("800x800")
app.title("ZeeshanCVApplication_SketricSolutions")
ctk.set_appearance_mode("dark")

#setup video to be used
video_path = 'test1.mp4'
#video_path_out = '{}_out.mp4'.format(video_path)
#cap = cv2.VideoCapture(video_path)
#cap = cv2.VideoCapture()

global videocounter
videocounter = 0

globalchoice=False
# setup application window
vidFrame = tk.Frame(height=480, width=600)
vidFrame.pack()

vid = ctk.CTkLabel(vidFrame)  # this holds our frames from video
vid.configure(text="")
#vid = ctk.CTkFrame(master=vidFrame, width=200, height=200)
vid.pack()


#load model
model_path = 'knife_yolov8.pt'

# Load a model
model = YOLO(model_path)  # load a custom model
threshold = 0.5

global center_points
center_points = []
global camcenter_points
camcenter_points = []

def detectwithvideo(cap):
	global videocounter
	global center_points


	if globalchoice == False:
		cap.release()
	ret, frame = cap.read()
	H, W, _ = frame.shape
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	if videocounter % 2 == 0:
		results = model(frame)[0]
		
		for result in results.boxes.data.tolist():
		    x1, y1, x2, y2, score, class_id = result

		    if score > threshold:
		        
		        cx = int((x1 + x2)/2)
		        cy = int((y1 + y2)/2)
		        center_points.append((cx, cy))
		        #cv2.circle(frame, (cx,cy), 5, (0,255,0), -1)
		        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
		        cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
		                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

	for pt in center_points:
		cv2.circle(frame, pt, 5, (0,255,0), -1)

	imgarr = Image.fromarray(frame)
	imgtk = ImageTk.PhotoImage(imgarr)
	vid.imgtk = imgtk
	vid.configure(image=imgtk)
	videocounter += 1

	vid.after(1, partial(detectwithvideo, cap))

def detectwithcamera(cap):
	global camcenter_points
	if globalchoice == False:
		cap.release()
	ret, frame = cap.read()
	H, W, _ = frame.shape
	results = model(frame)[0]

	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	
	for result in results.boxes.data.tolist():
		x1, y1, x2, y2, score, class_id = result
		if score > threshold:
			cx = int((x1 + x2)/2)
			cy = int((y1 + y2)/2)
			camcenter_points.append((cx, cy))

			cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
			cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
		                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

	for pt in camcenter_points:
		cv2.circle(frame, pt, 5, (0,255,0), -1)

	imgarr = Image.fromarray(frame)
	imgtk = ImageTk.PhotoImage(imgarr)
	vid.imgtk = imgtk
	vid.configure(image=imgtk)
	vid.after(1, partial(detectwithcamera, cap))


def startdetecting(x):
	global globalchoice
	#if cap.isOpened():
	#cap.release()
	if x == 1:
		globalchoice = True
		button1.configure(state="disabled")
		button2.configure(state="disabled")
		cap1 = cv2.VideoCapture(video_path)
		detectwithvideo(cap1)
	if x == 2:
		globalchoice = True
		button1.configure(state="disabled")
		button2.configure(state="disabled")
		cap2 = cv2.VideoCapture(0)
		detectwithcamera(cap2)


def stopdetection():
	global globalchoice
	global center_points
	global camcenter_points
	globalchoice = False
	button1.configure(state="normal")
	button2.configure(state="normal")
	center_points.clear()
	camcenter_points.clear()


button1 = ctk.CTkButton(master=app, text="Detection on video", command=partial(startdetecting, 1))
button1.pack(padx=20, pady=10)


button2 = ctk.CTkButton(master=app, text="Detection on webcam", command=partial(startdetecting, 2))
button2.pack(padx=20, pady=10)

button3 = ctk.CTkButton(master=app, text="stop", command=stopdetection)
button3.pack(padx=20, pady=10)

app.mainloop()
