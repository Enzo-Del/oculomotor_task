# -*- coding: utf-8 -*-
"""
Created on : 10/01/2021
@author: Enzo Delamarre
Last modified on :
Description : This code provide a open-source
              software to record simultaneously 2 videos streams
              from 2 distincts webcams.
"""
# Import
import cv2
import tkinter as tk
import time

import os

VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    # 'mp4': cv2.VideoWriter_fourcc(*'H264'),
    'mp4': cv2.VideoWriter_fourcc(*'mp4v'),
}


# Functions & classes
class CAMS():
    def __init__(self, base_filename, format_type, base_previewName, cam_IDs, FPS, acq_dur, width, height):

        self.base_filename = base_filename
        self.base_previewName = base_previewName

        self.cam_IDs = cam_IDs

        self.FPS = FPS
        self.acq_dur = acq_dur
        self.format_type = format_type

        self.width = width
        self.height = height

        if format_type == 'mp4':
            self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        elif format_type == 'avi':
            self.fourcc = cv2.VideoWriter_fourcc(*'XVID')

        self.record = False

        self.start_time = 0.0
        self.current_time = 0.0

    def run(self):

        print("Capture from  " + self.base_previewName + str(self.cam_IDs[0]) + " during " + str(
            self.acq_dur) + "s at " + str(self.FPS) + " FPS")
        print("Capture from  " + self.base_previewName + str(self.cam_IDs[1]) + " during " + str(
            self.acq_dur) + "s at " + str(self.FPS) + " FPS")
        print("Capture from  " + self.base_previewName + str(self.cam_IDs[2]) + " during " + str(
            self.acq_dur) + "s at " + str(self.FPS) + " FPS")
        print("Press r to record")
        cam1 = cv2.VideoCapture(self.cam_IDs[0])
        cam1.set(3,640)
        cam1.set(4,360)
        filename = (self.base_filename + str(self.cam_IDs[0]) + '.' + self.format_type)
        out1 = cv2.VideoWriter(r'C:\Users\opto-delamarree\Desktop\calib-camera-1.mp4', self.fourcc,
                               self.FPS, (640, 360),0)

        cam2 = cv2.VideoCapture(self.cam_IDs[1])
        cam2.set(3, 640)
        cam2.set(4, 360)
        filename = (self.base_filename + str(self.cam_IDs[1]) + '.' + self.format_type)
        out2 = cv2.VideoWriter(r'C:\Users\opto-delamarree\Desktop\calib-camera-2.mp4', self.fourcc,
                               self.FPS, (640, 360),0)
        cam3 = cv2.VideoCapture(self.cam_IDs[2])
        cam3.set(3, 640)
        cam3.set(4, 360)
        filename = (self.base_filename + str(self.cam_IDs[2]) + '.' + self.format_type)
        out3 = cv2.VideoWriter(r'C:\Users\opto-delamarree\Desktop\calib-camera-3.mp4', self.fourcc,
                               self.FPS, (640, 360), 0)

        cam4 = cv2.VideoCapture(self.cam_IDs[3])
        cam4.set(3, 640)
        cam4.set(4, 360)
        filename = (self.base_filename + str(self.cam_IDs[0]) + '.' + self.format_type)
        out4 = cv2.VideoWriter(r'C:\Users\opto-delamarree\Desktop\calib-camera-4.mp4', self.fourcc,
                               self.FPS, (640, 360), 0)
        while (True):
            # Capture frame-by-frame

            ret1, frame1 = cam1.read()
            ret2, frame2 = cam2.read()
            ret3, frame3 = cam3.read()
            ret4, frame4 = cam4.read()
            # Our operations on the frame come here
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            gray3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)
            gray4 = cv2.cvtColor(frame4, cv2.COLOR_BGR2GRAY)
            if (self.record):
                out1.write(gray1)
                out2.write(gray2)
                out3.write(gray3)
                out4.write(gray4)
                if (time.time()-self.start_time) >= self.acq_dur :
                    print('quitting')
                    break

            # Display the resulting frame
            cv2.imshow('frame1', gray1)
            cv2.imshow('frame2', gray2)
            cv2.imshow('frame3', gray3)
            cv2.imshow('frame4', gray4)
            ret = cv2.waitKey(1)

            if ret & 0xFF == ord('r'):
                self.toggleRecord()
                print('toggling')
                if (self.record):
                    print('record has started')
                else:
                    print('record has finished')

            if ret & 0xFF == ord('q'):
                print('quitting')
                break

        cam1.release()
        out1.release()
        cam2.release()
        out2.release()
        cam3.release()
        out3.release()
        cam4.release()
        out4.release()
        self.record = False

    def toggleRecord(self):
        self.record = ~self.record
        self.start_time = time.time()

    def getRecord(self):
        return self.record


def printT(string1, string2, root):
    label1 = tk.Label(root, text=string1)
    label2 = tk.Label(root, text=string2)
    label1.grid(column=0, row=1)
    label2.grid(column=0, row=3)


# Main

folder_path = r'C:\Users\opto-delamarree\Desktop'
video_name = '\test'

format_type = 'mp4'

filename = str(folder_path + video_name)

acquisition_duration = 60
frame_rate = 30
resolution = (640, 360)

cams = CAMS(filename, format_type, "Camera", [0, 1, 2, 3], frame_rate, acquisition_duration, resolution[0], resolution[1])
cams.run()
