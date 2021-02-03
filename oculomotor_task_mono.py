#-*- coding: utf-8 -*-
"""
Created on Sun Mar  8 11:48:02 2020
Last modified on
@author: Enzo Delamarre
Version : mono
"""

####################################################
# Import et définition des fonctions
####################################################

from __future__ import absolute_import, division
import math
import threading
import tkinter
from datetime import date
import tensorflow as tf
import numpy as np
import deeplabcut
import pandas as pd
import cv2
import os
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter.ttk import Frame, Label, Style
import pyfirmata
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
import os.path
from deeplabcut.pose_estimation_tensorflow.nnet import predict
from deeplabcut.pose_estimation_tensorflow.config import load_config
from tqdm import tqdm
import tensorflow as tf
from deeplabcut.utils import auxiliaryfunctions
from skimage.util import img_as_ubyte
import skimage
import _thread
import time
import psychopy
from psychopy import visual, core
from numpy.random import choice
from PIL import ImageTk, Image
#board = pyfirmata.Arduino('COM4')
#board.digital[13].write(0)
from tkinter import messagebox
tf.reset_default_graph()
from dlclive import DLCLive, Processor


def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)


STD_DIMENSIONS =  {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}

def stopwatch(seconds):
    start = time.time()
    time.clock()
    elapsed = 0
    while elapsed < seconds:
        elapsed = round(time.time() - start)
        print (elapsed)
        time.sleep(1)


def displayVid( frame):
    cv2.imshow('Calibration', frame)



def get_dims(cap, res='1080p'):
    width, height = STD_DIMENSIONS["480p"]
    if res in STD_DIMENSIONS:
        width,height = STD_DIMENSIONS[res]
    ## change the current caputre device
    ## to the resulting resolution
    change_res(cap, width, height)
    return width, height


VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    #'mp4': cv2.VideoWriter_fourcc(*'H264'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}



def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
      return  VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']


def check_cbox(event) :

   global s1
   global s2
   global s3
   global s4
   global s5
   global s6
   global s7
   if E1.get():
       s1 = E1.get()
   if E2.get():
       s2 = E2.get()
   if E3.get():
       s3 = E3.get()
   if cbox.get() == "Nasal saccade" :
       s4 = cbox.get()
   if cbox.get() == "Temporal saccade" :
       s4 = cbox.get()
   if cbox.get() == "Both" :
       s4 = cbox.get()
   if E5.get():
       s5 = E5.get()
   if E6.get():
       s6 = E6.get()
   if E7.get():
       s7 = E7.get()


def analyse_mouvements(pathProject,pathVideo) :
  deeplabcut.analyze_videos(pathProject, pathVideo, videotype='mp4',save_as_csv=True)
  #deeplabcut.create_labeled_video(pathProject, pathVideo,trailpoints=10,draw_skeleton=False,save_frames=True)
 # deeplabcut.plot_trajectories(pathProject, pathVideo)


def dfToArray(array, dataFrame, size) :
  for j in range(2, size):
    array[0, j] = float(dataFrame.iloc[j])
  array[np.isnan(array)] = 0


def velocity(array1, array2, array3, size):
  for k in range(1, size):
    if k == 1:
      array3[0, k] = 0
    elif k == size-1:
      array3[0, k] = 0
    else :
      array3[0, k] = (((array1[0, k+1]-array1[0, k-1])**(2) + (array2[0, k+1]-array2[0, k-1])**(2))**(1/2))/2


def center_Pupil(nframes, arrayC, PredictedData) :
  r1 = 0
  size =nframes
  for i in range(1, size):

    a = ((((PredictedData[i, 0]-PredictedData[i, 3])**(2) + (PredictedData[i, 1]-PredictedData[i, 4])**(2))**(1/2))
    +(((PredictedData[i, 3] - PredictedData[i, 6]) ** (2) + (PredictedData[i, 4] - PredictedData[i, 7]) ** (2)) ** (1 / 2))
    +(((PredictedData[i, 6] - PredictedData[i, 9]) ** (2) + (PredictedData[i, 7] - PredictedData[i, 10]) ** (2)) ** (1 / 2))
    +(((PredictedData[i, 9] - PredictedData[i, 12]) ** (2) + (PredictedData[i, 10] - PredictedData[i, 13]) ** (2)) ** (1 / 2))
    +(((PredictedData[i, 12] - PredictedData[i, 15]) ** (2) + (PredictedData[i, 13] - PredictedData[i, 16]) ** (2)) ** (1 / 2))
    +(((PredictedData[i, 15] - PredictedData[i, 18]) ** (2) + (PredictedData[i, 16] - PredictedData[i, 19]) ** (2)) ** (1 / 2))
    +(((PredictedData[i, 18] - PredictedData[i, 21]) ** (2) + (PredictedData[i, 19] - PredictedData[i, 22]) ** (2)) ** (1 / 2))
    +(((PredictedData[i, 21] - PredictedData[i, 0]) ** (2) + (PredictedData[i, 22] - PredictedData[i, 1]) ** (2)) ** (1 / 2)))

    A=a/8
    r = 1.3066 * A
    r1 = r1 +r
    arrayC[0, i] = PredictedData[i, 18]
    arrayC[1, i] = PredictedData[i, 19] - r
  radius = r1/size
  return radius


def center_Pupil_avg(C, size,R0):
    x=0
    y=0
    for j in range(1, size):
        x = x + C[0, j]
        y = y + C[1, j]
        if math.isnan(x) == True:
            x = 0
        if math.isnan(y) == True:
            y = 0
    R0[0, 1]= x/size
    R0[1, 1]= y/size


def echelle(nframes, PredictedData):
    size = nframes
    echelle = 0
    for i in range(1, size):
       echelle = echelle + (((PredictedData[i, 24] - PredictedData[i, 30]) ** (2) )+ ((PredictedData[i, 24] - PredictedData[i, 31]) ** (2))) ** (1 / 2)
    echelle = echelle /size
    return echelle


def angular_position(echelle, radius, C, X0, Y0, size, Eh, Ev):
    Rlens = (1.25 * echelle) / 3
    R = math.sqrt((Rlens*Rlens)- (radius*radius)) - ((0.1 * echelle) / 3)

    for i in range(1, size):

        Eh[0, i] = math.degrees(np.arcsin((C[0, i] - X0[0, 0]) / R))
        Ev[0, i] = math.degrees(np.arcsin(-(C[1, i] - Y0[0, 0]) / R))
    Eh[np.isnan(Eh)] = 0
    Ev[np.isnan(Ev)] = 0


def global_variation_rate(nFrames, PredictedData):
    size = nFrames
    saccade = 0
    blink = 0

    for i in range(2, size-4):
        Tx = ((((PredictedData[i,0] - PredictedData[i-1,0]) / (PredictedData[i-1,0])) * 100
        + ((PredictedData[i,3] - PredictedData[ i - 1,3]) / (PredictedData[i - 1,3])) * 100
        +((PredictedData[i, 6] - PredictedData[i - 1,6]) / (PredictedData[i - 1,6])) * 100
        +((PredictedData[i, 9] - PredictedData[i - 1, 9]) / (PredictedData[i - 1, 9])) * 100
        +((PredictedData[i, 12] - PredictedData[i - 1, 12]) / (PredictedData[i - 1, 12])) * 100
        +((PredictedData[i, 15] - PredictedData[i - 1, 15]) / (PredictedData[i - 1, 15])) * 100
        +((PredictedData[i, 18] - PredictedData[i - 1, 18]) / (PredictedData[i - 1, 18])) * 100
        +((PredictedData[i, 21] - PredictedData[i - 1, 22]) / (PredictedData[i-1, 22])) * 100) / 8)

        Ty = (((PredictedData[i, 1] + PredictedData[i - 1, 1]) / (PredictedData[i - 1, 1])) * 100
        + ((PredictedData[i, 4] - PredictedData[i - 1, 4]) / (PredictedData[i - 1, 4])) * 100
        + ((PredictedData[i, 7] - PredictedData[i -1, 7]) / (PredictedData[i -1, 7])) * 100
        + ((PredictedData[i, 10] - PredictedData[i - 1, 10]) / (PredictedData[i - 1, 10])) * 100
        + ((PredictedData[i, 13] - PredictedData[i - 1, 13]) / (PredictedData[i - 1, 13])) * 100
        + ((PredictedData[i, 16] - PredictedData[i - 1, 16]) / (PredictedData[i - 1, 16])) * 100
        + ((PredictedData[i, 19] - PredictedData[i - 1, 19]) / (PredictedData[i - 1, 19])) * 100
        + ((PredictedData[i, 22] - PredictedData[i - 1, 22]) / (PredictedData[i - 1, 22])) * 100) / 8

        C = (((PredictedData[i, 28] - PredictedData[i - 1, 28]) / (PredictedData[i - 1, 28])) * 100
        + ((PredictedData[i, 34] - PredictedData[i - 1, 34]) / (PredictedData[i - 1, 34])) * 100) / 2



        if C > 10 or C < -10:
          blink =+1
          print('Cligenment détecté !')
        elif (Tx >= 5.53 or Tx <= -5.53) and (Ty>=2 or Ty<=-2):
          saccade =+1

        erreur = saccade + blink

    return erreur


def DLC_analysis_DIY_init(cfg,dlc_config ):
    dlc_config['num_outputs'] = cfg.get('num_outputs', dlc_config.get('num_outputs', 1))
    start_path = os.getcwd()  # record cwd to return to this directory in the end
    trainFraction = cfg['TrainingFraction'][0]
    try:
        Snapshots = np.array(
            [fn.split('.')[0] for fn in os.listdir(os.path.join(modelfolder, 'train')) if "index" in fn])
    except FileNotFoundError:
        raise FileNotFoundError(
            "Snapshots not found! It seems the dataset for shuffle %s has not been trained/does not exist.\n Please train it before using it to analyze videos.\n Use the function 'train_network' to train the network for shuffle %s." % (
            1, 1))

    if cfg['snapshotindex'] == 'all':
        print(
            "Snapshotindex is set to 'all' in the config.yaml file. Running video analysis with all snapshots is very costly! Use the function 'evaluate_network' to choose the best the snapshot. For now, changing snapshot index to -1!")
        snapshotindex = -1
    else:
        snapshotindex = cfg['snapshotindex']

    increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
    Snapshots = Snapshots[increasing_indices]
    print("Using %s" % Snapshots[snapshotindex], "for model", modelfolder)

    tf.reset_default_graph()
    # Check if data already was generated:
    dlc_config['init_weights'] = os.path.join(modelfolder, 'train', Snapshots[snapshotindex])
    trainingsiterations = (dlc_config['init_weights'].split(os.sep)[-1]).split('-')[-1]
    # Update number of output and batchsize
    dlc_config['num_outputs'] = cfg.get('num_outputs', dlc_config.get('num_outputs', 1))
    dlc_config['batch_size'] = cfg['batch_size']
    sess, inputs, outputs = predict.setup_GPUpose_prediction(dlc_config)
    pose_tensor = predict.extract_GPUprediction(outputs, dlc_config)  # extract_output_tensor(outputs, dlc_cfg)

    x_range = list(range(0, (3 * len(dlc_config['all_joints_names'])), 3))
    y_range = list(range(1, (3 * len(dlc_config['all_joints_names'])), 3))

    return sess, inputs, outputs, pose_tensor, Snapshots


def DLC_analysis_live(path):
    dlc_proc = Processor()
    dlc_live = DLCLive(path, processor = dlc_proc)
    return dlc_live


def display_image_right(gui_c, image):
    labr = Label(gui_c, image = image).place(x = 25, y = 50)


def display_image_left(gui_c, image):
    labl = Label(gui_c, image = image).place(x = 675, y = 50)


def circle_pos(screen_dist, circle_pos_angle):
    circle_pos = (math.tan(math.degrees(circle_pos_angle)))*screen_dist
    return abs(circle_pos)

def printT(string1, string2, gui):
    # if you want the button to disappear:
    # button.destroy() or button.pack_forget()
    label1 = Label(gui, text=string1)
    label2 = Label(gui, text=string2)
    label1.grid(column=0, row=1)
    label2.grid(column=0, row=3)


####################################################
# INFOS
####################################################

filename_C = r'C:\Users\opto-delamarree\Desktop\presentation\calib.mp4'
filename_V = r'C:\Users\opto-delamarree\Desktop\presentation\WIN_20200808_14_09_02_Prodownsampled.mp4'
cfg = auxiliaryfunctions.read_config(r"C:\Users\opto-delamarree\Desktop\Eye_validation-Enzo-2020-08-08\config.yaml")
modelfolder =(r"C:\Users\opto-delamarree\Desktop\Eye_validation-Enzo-2020-08-08\dlc-models\iteration-0\Eye_validationAug8-trainset95shuffle1")
dlc_config = load_config(r"C:\Users\opto-delamarree\Desktop\Eye_validation-Enzo-2020-08-08\dlc-models\iteration-0\Eye_validationAug8-trainset95shuffle1\test\pose_cfg.yaml")
stim_dur = int(60)
screen_dist = 6 #cm
circle_pose_angle =30
win1_right = visual.Window(
    size=(1024, 600), fullscr=False, screen=1,
    winType='pyglet', allowGUI=False, allowStencil=False,
    monitor='testMonitor', color=[-1, -1, -1], colorSpace='rgb',
    blendMode='avg', useFBO=True,
    units='deg')


####################################################
# GUI - Paramètres de l'expérience
####################################################

gui= Tk()
gui.title("Training device for an oculomotor behavioral task")
gui.geometry('600x600')
#Style().configure("TFrame",background ="#343")
#tab_control = ttk.Notebook(gui)
#tab1 = ttk.Frame(tab_control)
#tab_control.add(tab1, text="Parameters")
#tab_control.pack(expand=1, fill="both")
L1 = Label(gui, text="Name of the mice :")
L1.place(x=25, y=50)
L2 = Label(gui, text="Time to wait between trials (s) :")
L2.place(x=25, y=150)
L3 = Label(gui, text="Number of trials :")
L3.place(x=25, y=250)
E2 = Entry(gui, bd= 5)
E2.insert(END,'1')
E2.place(x=250,y=120)
E3 = Entry(gui, bd= 5)
E3.insert(END,'1')
E6 = Entry(gui, bd= 5)
E6.insert(END,'1')
E7 = Entry(gui, bd= 5)
E7.insert(END,'1')
E3.place(x=250,y=195)
E5= Entry(gui, bd= 5)
E5.insert(END,'1')
E7.place(x=250,y=460)
cbox = ttk.Combobox(gui)
L5 = Label(gui, text="Radius of the circle (deg) :")
L5.place(x=25, y=375)
L6 = Label(gui, text="Duration of the response window (s) :")
L6.place(x=25, y=425)
L7 = Label(gui, text="Enter probability of punition :")
L7.place(x=25, y=475)
L8 = Label(gui, text="Make sure that the mice is well head-fixed and that the cameras and screens are centered")
L8.place(x=25, y=510)
E1 = Entry(gui, bd =5)
E1.insert(END,'310')
E1.place(x=250,y=45)
E5.place(x=250,y=365)
E6= Entry(gui, bd= 5)
E6.place(x=250,y=415)
E7= Entry(gui, bd= 10)
E6.insert(END,'1')
L4 = Label(gui, text="Type of visual stimulation :")
L4.place(x=25, y=275)
cbox['values']= ( "Temporal saccade", "Nasal saccade", "Both")
cbox.current(0)
cbox.place(x=250,y=270)
s1 = E1.get()
s2 = E2.get()
s3 = E3.get()
s7 = E7.get()
c = Checkbutton(gui, text="")
c.place(x=505, y=510)
cbox.bind("<<ComboboxSelected>>", check_cbox)
var = tk.IntVar()
button = tk.Button(gui, text="Calibrate !", command=lambda: var.set(1))
button.place(x=510,y=550)
button.wait_variable(var)
gui.update()
button.destroy()
gui.destroy()
circle_stim_right = visual.Circle(win = win1_right, radius = int(s5), units = 'deg', fillColor =[1, 1, 1], lineColor=[1,1,1], edges=128 )
circle_pose = circle_pos(screen_dist, circle_pose_angle)
name_C = str('Calibration'+s1)
today = date.today()
name_F =r'C:\Users\opto-delamarree\Desktop\Visuomotor_Task' +chr(92) + str(s1) + '_' + str(today) +'.txt'
F = open(name_F, 'w')
#filename = name +'.mp4'
#frames_per_second = 30.0
#res = '720p'

if s4 == 'Left saccade':
    circle_stim_right.pos = [-450, 0]
elif s4 == 'Right saccade':
    circle_stim_right.pos = [450, 0]


####################################################
# CALIBRATION
####################################################
gui_calibration= Tk()
gui_calibration.title("Calibration")
gui_calibration.geometry('650x600')
app = tk.Frame(gui_calibration, bg="white")
app.grid()
lmain = Label(app)
lmain.grid()
#L1 = Label(gui_calibration, text="Camera 1 :")
#L1.grid(column=0, row=1)

gui_calibration.update()
cap_right = cv2.VideoCapture(filename_C)
#change_res(cap_right, 640, 360)
#out = cv2.VideoWriter(filename_C, get_video_type(filename_C), 30, get_dims(cap_C, res))
x_range = list(range(0,(3 * len(dlc_config['all_joints_names'])),3))
y_range = list(range(1,(3 * len(dlc_config['all_joints_names'])),3))
ny, nx = int(cap_right.get(4)), int(cap_right.get(3))
batchsize = 1
batch_ind = 0
batch_num = 0
#nframes = 300 # 10 s
nframes = int(cap_right.get(7))
step=max(10,int(nframes/100))
frames_right = np.empty((batchsize, ny, nx,3), dtype='ubyte') # this keeps all frames in a batch
colors = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0), (240, 32, 160), (240, 32, 160), (240, 32, 160)]
sess, inputs, outputs, pose_tensor, Snapshots = DLC_analysis_DIY_init(cfg, dlc_config)
#dlc_live = DLC_analysis_live(r'C:\Users\opto-delamarree\Desktop\GBM3100_Final_Network-Enzo-2020-03-30\dlc-models\iteration-0\GBM3100_Final_NetworkMar30-trainset95shuffle1\test')
PredictedData_right = np.zeros((nframes, dlc_config['num_outputs'] * 3 * len(dlc_config['all_joints_names'])))
#PredictedData_left = np.zeros((nframes, dlc_config['num_outputs'] * 3 * len(dlc_config['all_joints_names'])))
counter = 0
pbar=tqdm(total=nframes)
while counter<= nframes-2:

     if counter%step==0:
         pbar.update(step)
     circle_stim_right.draw()
     win1_right.flip()
     retr, frame_right = cap_right.read()
     if retr :

        #frame_right =cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
        #frame_right = cv2.rotate(frame_right, cv2.ROTATE_180)
        if cfg['cropping']:
            frames_right[batch_ind]= img_as_ubyte(frame_right[cfg['y1']:cfg['y2'],cfg['x1']:cfg['x2']])
        else:
            frames_right[batch_ind] = img_as_ubyte(frame_right)
        pose = sess.run(pose_tensor, feed_dict={inputs: np.expand_dims(frame_right, axis=0).astype(float)})
        pose[:, [0, 1, 2]] = pose[:, [1, 0, 2]]
        # pose = predict.getpose(frame, dlc_config, sess, inputs, outputs)
        PredictedData_right[counter, :] = pose.flatten()
        frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
        gui_calibration.update()


     else:
         nframes=counter

         break


     for  x_plt, y_plt, c in zip(x_range, y_range, colors):

         image = cv2.circle(frame_right, (int(PredictedData_right[counter, :][x_plt]), int(PredictedData_right[counter, :][y_plt])), 3, c,
                            -1)
         img = Image.fromarray(image)
         imgtk = ImageTk.PhotoImage(image=img)
         lmain.imgtk = imgtk
         lmain.configure(image=imgtk)

     counter+= 1


circle_stim_right.draw()
win1_right.flip()
pbar.close()
cap_right.release()
check1 = global_variation_rate(nframes, PredictedData_right)
#if check1  > 0:
    #print('Calibration failed, please restart')
    #sys.exit(1)

C_c = np.zeros([2, nframes])
radius = float(center_Pupil(nframes, C_c, PredictedData_right))
R0 = np.zeros([2, 2])
center_Pupil_avg(C_c, nframes,R0)
print(R0)
echelle = float(echelle(nframes,PredictedData_right))
Rlens = (1.25 * echelle) / 3
#print(Rlens)
#print(radius)
R = math.sqrt((Rlens * Rlens) - (radius * radius)) - ((0.1 * echelle) / 3)
F.write("Calibration : \r\n")
F.write("Rayon moyen de la pupille : %d pixels \r\n" % radius)
F.write("Echelle utilisée : %d pixels  --> 3mm \r\n" % echelle)
L3 = Label(gui_calibration, text="Calibration Done! Press the Go! button to start the trials ! ")
L3.place(x = 50, y = 450)
L4 = Label(gui_calibration, text=("Mean radius of the pupil : %d pixels \r\n" % radius))
L4.place(x = 50, y = 500)
L5 = Label(gui_calibration, text=("Used scale : %d pixels  --> 3mm \r\n" % echelle))
L5.place(x = 50, y = 550)
var = tk.IntVar()
button = tk.Button(gui_calibration, text="Go !", command=lambda: var.set(1))
button.place(x=510,y=550)
button.wait_variable(var)
gui_calibration.update()
button.destroy()
gui_calibration.destroy()
reward_negatif = 0
reward_positif = 0
compteurG=0
compteurD=0
time.sleep(int(s2))

####################################################
# ANALYSE TEMPS REEL
####################################################
cap_right = cv2.VideoCapture(filename_V)
gui_trials= Tk()
gui_trials.title("Trials")
gui_trials.geometry('650x600')
L1 = Label(gui_trials, text="Camera 1 :")
L1.place(x=10, y=25)
app2 = tk.Frame(gui_trials, bg="white")
app2.grid()
lmain2 = Label(app2)
lmain2.grid()
gui_trials.update()

for k in range(0, int(s3)):
    win1_right.color = [-1, -1, -1]
    s7 = 10
    #if k == 0 :
        #filename_V = r'C:\Users\taches-comportements\Desktop\GBM3100_Final_Network-Enzo-2020-03-30\Tests\Demo.mp4'
    #if k==1 :
        #filename_V = r'C:\Users\taches-comportements\Desktop\GBM3100_Final_Network-Enzo-2020-03-30\Tests\1.mp4'
    #if k==2 :
        #filename_V = r'C:\Users\taches-comportements\Desktop\GBM3100_Final_Network-Enzo-2020-03-30\Tests\2.mp4'

    #centered cercle
    circle_stim_right.pos = (0, 0)
    circle_stim_right.radius = int(s5)
    win1_right.flip()

    colors = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0), (240, 32, 160), (240, 32, 160), (240, 32, 160)]
    dlc_config['num_outputs'] = cfg.get('num_outputs', dlc_config.get('num_outputs', 1))
    start_path = os.getcwd()  # record cwd to return to this directory in the end
    trainFraction = cfg['TrainingFraction'][0]
    try:
        Snapshots = np.array(
            [fn.split('.')[0] for fn in os.listdir(os.path.join(modelfolder, 'train')) if "index" in fn])
    except FileNotFoundError:
        raise FileNotFoundError(
            "Snapshots not found! It seems the dataset for shuffle %s has not been trained/does not exist.\n Please train it before using it to analyze videos.\n Use the function 'train_network' to train the network for shuffle %s." % (
            1, 1))

    if cfg['snapshotindex'] == 'all':
        print(
            "Snapshotindex is set to 'all' in the config.yaml file. Running video analysis with all snapshots is very costly! Use the function 'evaluate_network' to choose the best the snapshot. For now, changing snapshot index to -1!")
        snapshotindex = -1
    else:
        snapshotindex = cfg['snapshotindex']

    increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
    Snapshots = Snapshots[increasing_indices]
    print("Using %s" % Snapshots[snapshotindex], "for model", modelfolder)
    tf.reset_default_graph()
    # Check if data already was generated:
    dlc_config['init_weights'] = os.path.join(modelfolder, 'train', Snapshots[snapshotindex])
    trainingsiterations = (dlc_config['init_weights'].split(os.sep)[-1]).split('-')[-1]
    # Update number of output and batchsize
    dlc_config['num_outputs'] = cfg.get('num_outputs', dlc_config.get('num_outputs', 1))
    batchsize = 1
    dlc_config['batch_size'] = cfg['batch_size']
    sess, inputs, outputs = predict.setup_GPUpose_prediction(dlc_config)
    pose_tensor = predict.extract_GPUprediction(outputs, dlc_config)  # extract_output_tensor(outputs, dlc_cfg)
    x_range = list(range(0, (3 * len(dlc_config['all_joints_names'])), 3))
    y_range = list(range(1, (3 * len(dlc_config['all_joints_names'])), 3))
    centered = False
    counter_centered = 0
    saccade_V = False
    #filename = s1 +'.mp4'
    #cap = cv2.VideoCapture(filename_V)
    #out = cv2.VideoWriter(filename, get_video_type(filename), 30, get_dims(cap, res))
    nframes_V = int(cap_right.get(7))
    #nframes_V = (30 * int(s6))
    #nframes_V = 5000
    PredictedData_V = np.zeros((nframes_V, dlc_config['num_outputs'] * 3 * len(dlc_config['all_joints_names'])))
    tf.reset_default_graph()
    counter = 0
    counter_V = 0
    step_V=max(10, int(nframes_V/100))
    batch_ind = 0
    batch_num = 0
    ny, nx = int(cap_right.get(4)), int(cap_right.get(3))
    pbar=tqdm(total=nframes_V)
    draw = np.zeros([1, 1])
    draw_S = np.zeros([1,1])
    if s4 == 'Both':
        draw_S[0,0] = choice([0, 1], 1, p=[0.5,  0.5])
        if draw_S[0,0]  == 1:
            stim = 'Temporal saccade'
        else :
            stim ='Nasal saccade'

    else:
        stim = str(s4)
    print(stim)
    F.write(" Stimulation :  %s   \r\n" % stim)
    if stim == 'Temporal saccade':
        circle_stim_right.pos = (-450, 0)
    else:
        circle_stim_right.pos = (+450, 0)

    circle_stim_right.radius = int(s5)
    saccadeD = 0
    saccadeG = 0
    L=0
    stopwatch(int(s2))
    start_S = time.time()
    message = 'waiting for a saccade'
    while counter <= nframes_V -1:
        if counter % step == 0:
            pbar.update(step)
        #board.digital[13].write(0)
        ret, frame = cap_right.read()
        #if (time.time() - start_S)  >= (int(s6) + stim_dur):
            #ret = False
        if centered == True:
            circle_stim_right.draw()
            win1_right.flip()
        if ret:
            start_T = time.time()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if cfg['cropping']:
                frame = img_as_ubyte(frame[cfg['y1']:cfg['y2'], cfg['x1']:cfg['x2']])
            else:
                frame = img_as_ubyte(frame)

            pose = sess.run(pose_tensor, feed_dict={inputs: np.expand_dims(frame, axis=0).astype(float)})
            pose[:, [0, 1, 2]] = pose[:, [1, 0, 2]]
            # pose = predict.getpose(frame, dlc_config, sess, inputs, outputs)
            PredictedData_V[counter, :] = pose.flatten()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if counter > 0:

                # Variation rate estimation
                Tx = abs(((((PredictedData_V[counter, 0] - PredictedData_V[counter - 1, 0]) / (PredictedData_V[counter - 1, 0])) * 100
                + ((PredictedData_V[counter, 3] - PredictedData_V[counter - 1, 3]) / (PredictedData_V[counter - 1, 3])) * 100
                + ((PredictedData_V[counter, 6] - PredictedData_V[counter - 1, 6]) / (PredictedData_V[counter - 1, 6])) * 100
                + ((PredictedData_V[counter, 9] - PredictedData_V[counter - 1, 9]) / (PredictedData_V[counter - 1, 9])) * 100
                + ((PredictedData_V[counter, 12] - PredictedData_V[counter - 1, 12]) / (PredictedData_V[counter - 1, 12])) * 100
                + ((PredictedData_V[counter, 15] - PredictedData_V[counter - 1, 15]) / (PredictedData_V[counter - 1, 15])) * 100
                + ((PredictedData_V[counter, 18] - PredictedData_V[counter - 1, 18]) / (PredictedData_V[counter - 1, 18])) * 100
                + ((PredictedData_V[counter, 21] - PredictedData_V[counter - 1, 21]) / (PredictedData_V[counter - 1, 21])) * 100) / 8))

                # Velocity estimation
                V = (((((PredictedData_V[counter, 0] - PredictedData_V[counter - 1, 0]) ** (2) + (PredictedData_V[counter, 1] - PredictedData_V[counter - 1, 1]) ** (2)) ** (1 / 2))
                + (((PredictedData_V[counter, 3] - PredictedData_V[counter - 1, 3]) ** (2) + (PredictedData_V[counter, 4] - PredictedData_V[counter - 1, 4]) ** (2)) ** (1 / 2))
                + (((PredictedData_V[counter, 6] - PredictedData_V[counter - 1, 6]) ** (2) + (PredictedData_V[counter, 7] - PredictedData_V[counter - 1, 7]) ** (2)) ** (1 / 2))
                + (((PredictedData_V[counter - 1, 9] - PredictedData_V[counter - 1, 9]) ** (2) + (PredictedData_V[counter, 10] - PredictedData_V[counter - 1, 10]) ** (2)) ** (1 / 2))
                + (((PredictedData_V[counter, 12] - PredictedData_V[counter - 1, 12]) ** (2) + (PredictedData_V[counter, 13] - PredictedData_V[counter - 1, 13]) ** (2)) ** (1 / 2))
                + (((PredictedData_V[counter, 15] - PredictedData_V[counter - 1, 15]) ** (2) + (PredictedData_V[counter, 16] - PredictedData_V[counter - 1, 16]) ** (2)) ** (1 / 2))
                + (((PredictedData_V[counter, 18] - PredictedData_V[counter - 1, 18]) ** (2) + (PredictedData_V[counter, 19] - PredictedData_V[counter - 1, 19]) ** (2)) ** (1 / 2))
                + (((PredictedData_V[counter, 21] - PredictedData_V[counter - 1, 21]) ** (2) + (PredictedData_V[counter, 22] - PredictedData_V[counter - 1, 22]) ** (2)) ** (1 / 2))) / 16)

                # Blink verification
                TB = ((PredictedData_V[counter, 28] - PredictedData_V[counter - 1, 28]) / (PredictedData_V[counter - 1, 28])) * 100
                if TB>10:
                    print('Blink !!')
                    timestp = float(time.time() - start_S)
                    F.write(" %f s : Blink! \r\n" % timestp)

                #Block the algo for 3 frames to analyse the detected saccade
                if Tx >= 2 and V > 2.5 and TB <10:
                    L=1
                    saccade_V = True
                    counter_V = counter

                if (counter - counter_V) <= 3 and L==1:
                    saccade_V = True
                else:
                    saccade_V = False
                    message = 'Waiting for a saccade'

                # Absolute angle estimation
                Eh = np.zeros([nframes_V, 2])
                Ev = np.zeros([nframes_V, 2])

                a = ((((PredictedData_V[counter, 0] - PredictedData_V[counter, 3]) ** (2) + (PredictedData_V[counter, 1] - PredictedData_V[counter, 2]) ** (2)) ** (1 / 2))
                + (((PredictedData_V[counter, 3] - PredictedData_V[counter, 6]) ** (2) + (PredictedData_V[counter, 4] - PredictedData_V[counter, 7]) ** (2)) ** (1 / 2))
                + (((PredictedData_V[counter, 6] - PredictedData_V[counter, 9]) ** (2) + (PredictedData_V[counter, 7] - PredictedData_V[counter, 10]) ** (2)) ** (1 / 2))
                + (((PredictedData_V[counter, 9] - PredictedData_V[counter, 12]) ** (2) + (PredictedData_V[counter, 10] - PredictedData_V[counter, 13]) ** (2)) ** (1 / 2))
                + (((PredictedData_V[counter, 12] - PredictedData_V[counter, 15]) ** (2) + (PredictedData_V[counter, 13] - PredictedData_V[counter, 16]) ** (2)) ** (1 / 2))
                + (((PredictedData_V[counter, 15] - PredictedData_V[counter, 18]) ** (2) + (PredictedData_V[counter, 16] - PredictedData_V[counter, 19]) ** (2)) ** (1 / 2))
                + (((PredictedData_V[counter, 18] - PredictedData_V[counter, 21]) ** (2) + (PredictedData_V[counter, 19] - PredictedData_V[counter, 22]) ** (2)) ** (1 / 2))
                + (((PredictedData_V[counter, 21] - PredictedData_V[counter, 0]) ** (2) + (PredictedData_V[counter, 22] - PredictedData_V[counter, 1]) ** (2)) ** (1 / 2)))

                A = a / 8
                r = 1.3066 * A
                #print(r)
                C = np.zeros([nframes_V, 2])
                C[counter, 0] = PredictedData_V[counter, 18]
                C[counter, 1] = PredictedData_V[counter, 19] - r

                #print(C[counter, 0])
                #print(C[counter, 1])
                #print('-----')
                Eh[counter, 0] = math.degrees(np.arcsin((C[counter, 0] - R0[0, 1]) / R))
                #Ev[counter, 0] = math.degrees(np.arcsin(-(C[counter, 1] - R0[1, 1]) / R))
                #print(Eh[counter, 0])
                printT('Azimutal angular absolute position = ' + str(round(Eh[counter, 0])) +str( 'degrees'), message, gui_trials)
                gui_trials.update()
                # Check if eye is centered on the central point before starting trial
                if Eh[counter, 0] >= -1 and Eh[counter, 0]<= 1 and centered == False:
                    counter_centered =+1
                    if counter_centered >= 30:
                        centered = True

                # Direction of saccade estimation
                if Eh[counter, 0] <= -circle_pose_angle and saccade_V == True and saccadeD<1 and saccadeG<1 and centered == True:
                    saccadeD = saccadeD + 1
                    compteurD = compteurD+1
                    print('Saccade direction nasale !')
                    message = 'Nasal saccade !'

                    if stim == 'Nasal saccade':
                        timeT = time.time()-start_T
                        print(timeT)
                        reward_positif =reward_positif + 1
                        #board.digital[13].write(1)
                        print('Récompense!')
                        timestp = float(time.time() - start_S)
                        F.write(" %f s : Récompense ! \r\n" % timestp)
                    else :
                        draw[0,0]  = choice([0, 1], 1, p=[(int(s7)/10), (1-(int(s7)/10))])
                        if draw[0,0]  == 0:
                            timeT = time.time() - start_T
                            print(timeT)
                            reward_negatif = reward_negatif + 1
                            print('Punition!')
                           # win.color = [1, 1, 1]

                            timestp = float(time.time() - start_S)
                            F.write(" %f s : Punition ! \r\n" % timestp)
                        else:
                            print('Bad saccade but no punition')

                if Eh[counter, 0] >= circle_pose_angle and saccade_V == True and saccadeG<1 and saccadeD <1 and centered == True:
                    saccadeG = saccadeG + 1
                    compteurG = compteurG+1
                    print('Saccade direction temporale !')
                    message = 'Temporal saccade !'


                    if stim == 'Temporal saccade':
                        timeT = time.time() - start_T
                        print(timeT)
                        reward_positif =reward_positif + 1
                        #board.digital[13].write(1)
                        print('Récompense!')
                        timestp = float(time.time() - start_S)
                        F.write(" %f s : Récompense ! \r\n" % timestp)
                    else :
                        draw[0,0]  = choice([0, 1], 1,  p=[(int(s7)/10), (1-(int(s7)/10))])
                        if draw[0,0]  == 0:
                            timeT = time.time() - start_T
                            print(timeT)
                            reward_negatif = reward_negatif + 1
                            print('Punition!')
                            #win. color=[1, 1, 1]
                            timestp = float(time.time() - start_S)
                            F.write(" %f s : Punition ! \r\n" % timestp)
                        else:
                            print('Bad saccade but no punition')
                            timestp = float(time.time() - start_S)
                            F.write(" %f s : Bad saccade but no punition \r\n" % timestp)

                printT('Azimutal angular absolute position = ' + str(round(Eh[counter, 0])) + str('degrees'), message, gui_trials)
                gui_trials.update()
        else:
            nframes = counter
            circle_stim_right.radius = 0
            if counter_V <1 :
                draw[0, 0] = choice([0, 1], 1,  p=[(int(s7)/10), (1-(int(s7)/10))])
                if draw[0, 0] == 0:
                    reward_negatif = reward_negatif + 1
                    print('Punition!')
                    win1_right.color = [1, 1, 1]
                    timestp = float(time.time() - start_S)
                    F.write(" %f s : Punition ! \r\n" % timestp)
                else:
                    print('No saccade but no punition')
                    timestp = float(time.time() - start_S)
                    F.write(" %f s : No saccade but no punition \r\n" % timestp)

            win1_right.colors = [-1, -1, -1]
            circle_stim_right.draw()
            win1_right.flip()

            break

        for x_plt, y_plt, c in zip(x_range, y_range, colors):
            image = cv2.circle(frame, (
            int(PredictedData_right[counter, :][x_plt]), int(PredictedData_right[counter, :][y_plt])), 3, c,
                               -1)
            img = Image.fromarray(image)
            imgtk = ImageTk.PhotoImage(image=img)
            lmain2.imgtk = imgtk
            lmain2.configure(image=imgtk)

        counter += 1

    pbar.close()
    cap_right.release()


FPS = nframes_V/(time.time()-start_S)
print(FPS)
win1_right.colors = [-1, -1, -1]
win1_right.flip()
sucess_Rate =int( (reward_positif/int(s3))*100)
gui_F= Tk()
gui_F.title("Training device for a visuomotor behavioral task")
gui_F.geometry('500x600')
Style().configure("TFrame",background ="#49A")
tab_control = ttk.Notebook(gui_F)
tab1_F = ttk.Frame(tab_control)
#tab_control.add(tab1, text="Results")
tab_control.pack(expand=1, fill="both")
L1 = Label(tab1_F, text="Success rate (%):")
L1.place(x=25, y=25)
L2 = Label(gui_F, text="Nasal direction saccades :")
L2.place(x=25, y=125)
L3 = Label(gui_F, text="Temporal direction saccades :")
L3.place(x=25, y=225)
E1 = Entry(gui_F, bd =5)
E1.insert(END,sucess_Rate)
E1.place(x=200,y=45)
E2 = Entry(gui_F, bd =5)
E2.insert(END,compteurD)
E2.place(x=200,y=125)
E3 = Entry(gui_F, bd =5)
E3.insert(END,compteurG)
E3.place(x=200,y=225)
gui_F.mainloop()

####################################################
# Sauvegarde des données
####################################################
F.write("Sucess rate : %d /100 \r\n" % sucess_Rate)
F.close()
#win.close()
core.quit()
sys.exit()
