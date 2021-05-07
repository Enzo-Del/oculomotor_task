# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 11:48:02 2020
Last modified on 19/04/2021
@author: Enzo Delamarre -- Lab Vanni
Version : mono
"""

########################################################################################################################
#                 LIBRAIRIES IMPORT                 #
########################################################################################################################

from __future__ import absolute_import, division
import math
from datetime import date

print('Importing TensorFlow ...')
import tensorflow as tf

print('Done')
import numpy as np
import cv2
import os
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter.ttk import Label, Style
import pyfirmata

_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)
print('Loading GPU ...')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
print('Done')
import os.path
from dlclive import DLCLive
from tqdm import tqdm
import tensorflow as tf
from skimage.util import img_as_ubyte
import time
import psychopy
from psychopy import visual, core
from numpy.random import choice
from PIL import ImageTk, Image
from tkinter import messagebox
import MiceEyeMathsTools
from datetime import datetime
from tkinter import scrolledtext
from threading import Thread
from deeplabcut.pose_estimation_tensorflow.config import load_config
import matplotlib.pyplot as plt


########################################################################################################################
#                 CLASS DEFINITION                 #
########################################################################################################################

class Camera():
    """ Class for the webcams
    """

    def __init__(self, cam_id, mice_name, res_width, res_height, FPS, grayscale):
        self.camID = cam_id
        self.mice_name = mice_name
        self.res_width = res_width
        self.res_height = res_height
        self.FPS = FPS
        self.grayscale = grayscale
        try:
            self.cam = cv2.VideoCapture(self.camID)
            print("Communication Successfully started with camera {}".format(self, self.camID))
        except:
            print('The camera is not connected or occupied, please check')
        self.cam.set(3, self.res_width)  # Resolution change
        self.cam.set(4, self.res_height)
        self.video_type = cv2.VideoWriter_fourcc(*'mp4v')  # Video format
        self.record = False
        today = date.today()
        today = today.strftime('%d%m%Y')
        self.out = cv2.VideoWriter(
            r'C:\Users\taches-comportements\Desktop\Oculomotor_task_v2.1' + chr(92) + str(today) + '_' + str(
                self.mice_name) + '.mp4', self.video_type, self.FPS, (self.res_width, self.res_height), grayscale)

        # Thread start
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        print('Acquisition thread started')
        time.sleep(1)
        (self.status, self.frame) = self.cam.read()

    def getImage(self):
        # global frame
        # try :
        # ret, self.frame = self.cam.read()
        # except :
        # print('Camera object is not reading any frame, check if camera is wired or already occupied')

        # self.out.write(self.frame)
        # [200:550, 300:800]
        # [85: 550, 250: 900]
        # [25:300,100:550]
        return self.frame

    def update(self):
        while True:
            if self.cam.isOpened():
                (self.status, self.frame) = self.cam.read()
                if self.grayscale:
                    self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                self.out.write(self.frame)

    def close(self):
        self.cam.release()
        self.out.release()


class PosePrediction():
    """ Class for DLC prediction with DLC functions
    """

    def __init__(self, model_folder):
        self.model_folder = model_folder

        self.colors = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0), (240, 32, 160),
                       (240, 32, 160),
                       (240, 32, 160), (246, 30, 160), (246, 30, 160), (246, 30, 160)]
        self.dlc_live = DLCLive(self.model_folder)

    def init_infer(self, frame):
        frame = img_as_ubyte(frame)
        pose = self.dlc_live.init_inference(frame)
        # pose[:, [0, 1, 2]] = pose[:, [1, 0, 2]]
        self.pose = pose.flatten()
        return self.pose

    def get(self, frame):
        frame = img_as_ubyte(frame)
        pose = self.dlc_live.get_pose(frame)
        # pose[:, [0, 1, 2]] = pose[:, [1, 0, 2]]
        self.pose = pose.flatten()
        return self.pose

    def dispPredictions(self, frame, pose, counter):
        for x_plt, y_plt, c in zip(self.x_range, self.y_range, self.colors):
            image = cv2.drawMarker(frame, (int(pose[counter, :][x_plt]), int(pose[counter, :][y_plt])), c,
                                   cv2.MARKER_STAR, 5, 2)
            # Essayer cv2 line + point central de la pupille
        img = Image.fromarray(image)
        imgtk = ImageTk.PhotoImage(image=img)
        return imgtk

    def dispPredictionsv2(self, frame, counter, pupil_center, radius):
        # image = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        image = cv2.drawMarker(frame, (int(pupil_center[0][counter]), int(pupil_center[1][counter])), (255, 0, 255),
                               cv2.MARKER_STAR, 3, 1)
        image = cv2.circle(image, (int(pupil_center[0][counter]), int(pupil_center[1][counter])), radius, (0, 255, 255),
                           1)
        img = Image.fromarray(image)
        imgtk = ImageTk.PhotoImage(image=img)
        return imgtk

    def checkProbability(self, pose, counter):
        # Check the probabilty of the pose prediciton
        if all(float(p) >= 0.90 for p in pose[counter, 2:3:340]):
            self.probability = True
        else:
            self.probability = False

        return self.probability


class ArduinoNano():

    def __init__(self, com_port):
        self.com_port = com_port
        try:
            self.board = pyfirmata.Arduino(com_port, baudrate=57600)
            print("Communication Successfully started with arduinoNano at {}".format(self, com_port))
        except:
            print("ArduinoNano not connected or wrong COM port, please check")


class PeristalticPump(ArduinoNano):

    def __init__(self, com_port, water_spout_qty):
        ArduinoNano.__init__(self, com_port)
        self.water_spout_qty = water_spout_qty  # in micro-liter
        self.lick_GPIO = self.board.digital[2]
        self.reward_GPIO = self.board.digital[6]
        # self.reward_GPIO.write(0)

        self.it = pyfirmata.util.Iterator(self.board)
        self.it.start()
        self.lick_GPIO.mode = pyfirmata.INPUT
        self.time_ON = 0.0075  # TODO : Change with pump calibration

    def reward(self):
        self.reward_GPIO.write(1)
        time.sleep(self.time_ON)
        self.reward_GPIO.write(0)

    def reward_flush(self):
        self.reward_GPIO.write(1)
        time.sleep(1)
        self.reward_GPIO.write(0)

    def lick(self):
        if self.lick_GPIO.read() == 1:
            lick = True
            time.sleep(0.01)
        else:
            lick = False
        return lick

    def close_ard(self):
        self.board.exit()


class VisualStimulation():
    def __init__(self, window, stim_angle_pos, screen_dist, stim_radius):

        self.win = visual.Window(
            size=(1024, 600), fullscr=True, screen=window,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[-1, -1, -1], colorSpace='rgb',
            blendMode='avg', useFBO=True,
            units='pix')
        self.stim_pos, self.stim_radius = MiceEyeMathsTools.circle_pos(screen_dist, stim_angle_pos, stim_radius)
        self.screen_dist = screen_dist
        self.visual_stim = visual.Circle(win=self.win, radius=stim_radius, units='pix', fillColor=[1, 1, 1],
                                         lineColor=[1, 1, 1],
                                         edges=128)
        self.win.flip()

    def stim(self, direction):
        self.win.color = [-1, -1, -1]
        if direction == 'Temporal':
            self.visual_stim.pos = [-self.stim_pos, 120]
        if direction == 'Nasal':
            self.visual_stim.pos = [self.stim_pos, 120]
        if direction == 'Center':
            self.visual_stim.pos = [0, 120]

        self.visual_stim.draw()
        self.win.flip()

    def whiteScreen(self):
        self.win.color = [1, 1, 1]
        self.win.flip()

    def blackScreen(self):
        self.win.color = [-1, -1, -1]
        self.win.flip()

    def gambleStim(self, past_stim1, past_stim2):
        draw = np.zeros([1, 1])
        draw[0, 0] = choice([0, 1], 1, p=[0.5, 0.5])
        if draw[0, 0] == 1:
            if past_stim1 == past_stim2 == 'Temporal':
                stim = 'Nasal'
            else:
                stim = 'Temporal'
        else:
            if past_stim1 == past_stim2 == 'Nasal':
                stim = 'Temporal'
            else:
                stim = 'Nasal'

        return stim

    def gamble_punition(self, probability):
        draw = np.zeros([1, 1])
        draw[0, 0] = choice([0, 1], 1, p=[(probability / 10), (1 - (probability / 10))])
        return draw[0, 0]


class Gui(PeristalticPump):
    def __init__(self, com_port, water_spout_qty):
        PeristalticPump.__init__(self, com_port, water_spout_qty)

        self.thread = Thread(target=self.report_lick, args=())
        self.thread.daemon = True
        # self.threadL = Thread(target=self.rewardLive, args=())
        # self.threadL.daemon = True
        # self.threadL.start()
        self.exp_started = False
        self.flag_preview = False
        self.rewardL = False
        self.root = tk.Tk()
        self.root.title("Training device for an oculomotor behavioral task")
        self.root.geometry('700x720+20+20')
        self.root.tk.call('wm', 'iconphoto', self.root._w, tk.PhotoImage(file='icon.PNG'))
        tk.Label(self.root, text="Mice name :").grid(row=0, column=0, sticky=W)
        tk.Label(self.root, text="Trials number :").grid(row=1, column=0, sticky=W)
        tk.Label(self.root, text="Probability of punition (/10) :").grid(row=2, column=0, sticky=W)
        tk.Label(self.root, text="Answer window duration (> or = to stim duration) [s] :").grid(row=3, column=0,
                                                                                                sticky=W)
        tk.Label(self.root, text="Stim duration [s] :").grid(row=4, column=0, sticky=W)
        tk.Label(self.root, text="Type of visual stimulation :").grid(row=5, column=0, sticky=W)
        tk.Label(self.root, text="Maximum duration of visual centered stim [s] :").grid(row=7, column=0, sticky=W)
        tk.Label(self.root, text="Centered fixation time [s] :").grid(row=8, column=0, sticky=W)
        tk.Label(self.root, text="Point radius [degrees] :").grid(row=9, column=0, sticky=W)
        tk.Label(self.root, text="Accepted precision by the mice [degrees]:").grid(row=10, column=0, sticky=W)
        tk.Label(self.root, text="Time to wait between trials [s] :").grid(row=11, column=0, sticky=W)
        tk.Label(self.root, text="Visual stimuli position [degrees] :").grid(row=12, column=0, sticky=W)
        tk.Label(self.root, text="Variation rate threshold [%] :").grid(row=14, column=0, sticky=W)
        tk.Label(self.root, text="Mice-screen distance [cm] :").grid(row=15, column=0, sticky=W)
        tk.Label(self.root, text="WARNING: Make sure that the mice").grid(row=16, column=0, sticky=W)
        tk.Label(self.root, text="is well head-fixed and that the ").grid(row=17, column=0, sticky=W)
        tk.Label(self.root, text="cameras and screens are centered !").grid(row=18, column=0, sticky=W)
        self.label1 = tk.Label(self.root, text="Si tout va mal, c'est que tout va bien. MV")
        self.label1.place(x=1, y=700)
        self.label2 = tk.Label(self.root, text='Version 2.1 - Neurophotonic VanniLab - 2021')
        self.label2.place(x=450, y=700)
        self.check_var = tk.IntVar()
        self.button_var = tk.IntVar()
        self.mice_name_entry = tk.Entry(self.root)
        # self.mice_name_entry.insert(END, '310')
        self.mice_name_entry.grid(row=0, column=1)
        self.trials_number_entry = tk.Entry(self.root)
        self.trials_number_entry.insert(END, '30')
        self.trials_number_entry.grid(row=1, column=1)
        self.punition_proba_entry = tk.Entry(self.root)
        self.punition_proba_entry.insert(END, '0')
        self.punition_proba_entry.grid(row=2, column=1)
        self.answer_win_entry = tk.Entry(self.root)
        self.answer_win_entry.insert(END, '2')
        self.answer_win_entry.grid(row=3, column=1)
        self.stim_dur_entry = tk.Entry(self.root)
        self.stim_dur_entry.insert(END, '2')
        self.stim_dur_entry.grid(row=4, column=1)
        self.max_center_dur_entry = tk.Entry(self.root)
        self.max_center_dur_entry.insert(END, '2')
        self.max_center_dur_entry.grid(row=7, column=1)
        self.centered_dur_entry = tk.Entry(self.root)
        self.centered_dur_entry.insert(END, '0.5')
        self.check_var1 = tk.IntVar()
        self.check_var2 = tk.IntVar()
        self.inactivity_pun_widget = tk.Checkbutton(self.root, text='Inactivity punition', variable=self.check_var1,
                                                    onvalue=1, offvalue=0)
        self.inactivity_pun_widget.grid(row=6, column=0, sticky=W)
        self.training_widget = tk.Checkbutton(self.root, text='Training', variable=self.check_var2,
                                              onvalue=1, offvalue=0)
        self.training_widget.grid(row=13, column=0, sticky=W)
        self.centered_dur_entry.grid(row=8, column=1)
        self.point_radius_entry = tk.Entry(self.root)
        self.point_radius_entry.insert(END, '40')
        self.point_radius_entry.grid(row=9, column=1)
        self.precision_entry = tk.Entry(self.root)
        self.precision_entry.insert(END, '5')
        self.precision_entry.grid(row=10, column=1)
        self.time_between_trials_entry = tk.Entry(self.root)
        self.time_between_trials_entry.insert(END, '4')
        self.time_between_trials_entry.grid(row=11, column=1)
        self.visual_stim_angle_pos_entry = tk.Entry(self.root)
        self.visual_stim_angle_pos_entry.insert(END, '20')
        self.visual_stim_angle_pos_entry.grid(row=12, column=1)
        self.mice_dist_entry = tk.Entry(self.root)
        self.mice_dist_entry.insert(END, '9')
        self.mice_dist_entry.grid(row=15, column=1)
        self.stim_type_cbox = ttk.Combobox(self.root, values=['Both', 'Temporal', 'nasal'])
        self.stim_type_cbox.current(0)
        self.stim_type_cbox.grid(row=5, column=1)
        self.go_button = tk.Button(self.root, text="Start experiment", command=self.start)
        self.go_button.grid(column=1, row=26, sticky=S)
        self.prevstim_button = tk.Button(self.root, text="Preview stim", command=self.previewStim)
        self.prevstim_button.grid(column=1, row=25, sticky=S)
        self.thresh_entry = tk.Entry(self.root)
        self.thresh_entry.insert(END, '20')
        self.thresh_entry.grid(row=14, column=1)
        self.water_button = tk.Button(self.root, text="Dispense sequence of water", command=self.dispense)
        self.water_button.grid(column=1, row=16)
        self.preview_button = tk.Button(self.root, text="Preview video", command=self.displayVideoStream)
        self.preview_button.grid(column=1, row=17)
        self.flush_button = tk.Button(self.root, text="Flush water", command=self.flush)
        self.flush_button.grid(column=1, row=18)
        self.go_button_var = tk.IntVar()
        self.water_button_var = tk.IntVar()
        self.flush_button_var = tk.IntVar()
        self.preview_button_var = tk.IntVar()
        self.go_button.wait_variable(self.go_button_var)
        self.label1.destroy()
        self.label2.destroy()
        self.root.children.clear()
        self.root.update()
        self.root.geometry('650x850')
        self.root.children.clear()
        self.root.update()
        tk.Label(self.root, text="Si tout va mal, c'est que tout va bien. MV").place(x=1, y=830)
        tk.Label(self.root, text='Version 2.1 - Neurophotonic VanniLab - 2021').place(x=405, y=830)
        # Video live update
        self.app = Frame(self.root, bg="white")
        self.app.grid(row=0, column=0)
        self.lmain = Label(self.app)
        self.lmain.grid(row=0, column=0)
        self.az_pos_label = tk.Label(self.root, text='Azimutal angle :')
        self.el_pos_label = tk.Label(self.root, text='Elevation angle :')
        self.pa_pos_label = tk.Label(self.root, text='Pupil area :')
        self.FPS_label = tk.Label(self.root, text='FPS :')
        self.az_pos_label.grid(row=2, column=0)
        self.el_pos_label.grid(row=3, column=0)
        self.pa_pos_label.grid(row=4, column=0)
        self.FPS_label.grid(row=5, column=0)
        self.water_button = tk.Button(self.root, text="Dispense sequence of water", command=self.dispense)
        self.water_button.grid(column=0, row=4, sticky=E)
        self.root.update()
        # Init log box
        self.log = scrolledtext.ScrolledText(self.root, width=75, height=15)
        self.log.grid(row=6, column=0)
        self.log.see(END)

        # Init log text file
        try:
            self.text_file = open(
                r"C:\Users\taches-comportements\Desktop\Oculomotor_task_v2.1" + chr(92) + str(date.today()) + '_' + str(
                    self.mice_name) + '.txt', 'w')
            self.text_file.write("Experiment started on : " + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "\r\n")
        except:
            print('An error as occured while trying to create a new text file.')

    def start(self):
        # Get experience parameters
        self.exp_started = True
        self.go_button_var.set(0)
        self.mice_name = str(self.mice_name_entry.get())

        self.trials_number = int(self.trials_number_entry.get())
        self.punition_proba = int(self.punition_proba_entry.get())
        self.answer_win = int(self.answer_win_entry.get())
        self.stim_dur = int(self.stim_dur_entry.get())
        self.stim_type = str(self.stim_type_cbox.get())
        self.max_center_dur = int(self.max_center_dur_entry.get())
        self.point_radius = int(self.point_radius_entry.get())
        self.precision = int(self.precision_entry.get())
        self.centered_dur = float(self.centered_dur_entry.get())
        self.time_between_trials = int(self.time_between_trials_entry.get())
        self.visual_stim_angle_pos = int(self.visual_stim_angle_pos_entry.get())
        self.inactivity_pun = bool(self.check_var1.get())
        self.training = bool(self.check_var2.get())
        self.thresh = int(self.thresh_entry.get())
        self.mice_dist = int(self.mice_dist_entry.get())
        # self.close_ard() #Close connection with arduino
        if self.flag_preview:
            self.win.close()
        self.clear()  # Clear widgets

    def dispense(self):
        self.reward()  # Reward

    def dispenseLive(self):
        self.rewardL = True  # Reward

    def flush(self):
        self.reward_flush()

    def dispFrame(self, frame):
        if self.exp_started:
            self.lmain.imgtk = frame
            self.lmain.configure(image=frame)
            self.root.update()
        else:
            self.lmain1.imgtk = frame
            self.lmain1.configure(image=frame)
            self.rootp.update()

    def updateAngle(self, az, el, pa, fps):
        self.az_pos_label.configure(text='Azimutal angle :   ' + str(az) + ' degrees')
        self.el_pos_label.configure(text='Elevation angle :   ' + str(el) + ' degrees')
        self.pa_pos_label.configure(text='Pupil area :   ' + str(pa) + ' mm^2')
        self.FPS_label.configure(text='FPS : ' + str(fps) + ' frames/s')

    def updateLog(self, content):
        self.log.insert(END, datetime.now().strftime('%H:%M:%S.%f : ') + str(content) + "\r\n")
        self.log.see(END)
        self.root.update()

    def updatelogText(self, content):

        # Save content to text file
        self.text_file.write(datetime.now().strftime('%H:%M:%S.%f')[:-3] + str(' ' + content) + "\r\n")

    def displayVideoStream(self):
        self.rootp = tk.Toplevel()
        self.rootp.title("Video preview")
        self.rootp.geometry('650x460')
        self.cam = cv2.VideoCapture(0)
        self.cam.set(3, 640)  # Resolution change
        self.cam.set(4, 360)
        self.app1 = Frame(self.rootp, bg="white")
        self.app1.grid(row=0, column=0, sticky=N)
        self.lmain1 = Label(self.app1)
        self.lmain1.grid(row=0, column=0, sticky=N)
        self.placement_button = tk.Button(self.rootp, text="Mice is well-placed", command=self.close_preview)
        self.placement_button.grid(column=0, row=3)
        tk.Label(self.rootp, text="Make sure the head with the eye is in the square").grid(row=1, column=0)
        self.water_button = tk.Button(self.rootp, text="Dispense sequence of water", command=self.dispenseLive)
        self.water_button.grid(column=0, row=2)
        self.go_placement_var = tk.IntVar()
        self.dispense_button_var = tk.IntVar()
        while self.exp_started == False:
            self.ret, self.frame = self.cam.read()
            cv2.rectangle(self.frame, (100, 25), (550, 300), (0, 255, 0), 2)
            img = Image.fromarray(self.frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.dispFrame(imgtk)

        return None

    def report_lick(self):
        while True:
            if self.lick_GPIO.read() == 1:
                lick = True
                self.updateLog('Mice lick')
                time.sleep(0.05)
            else:
                lick = False

    def start_lick_reporting(self):
        self.thread.start()
        print('Lick thread started')

    def rewardLive(self):
        if self.rewardL == True:
            self.reward_GPIO.write(1)
            self.board.pass_time(self.time_ON)
            self.reward_GPIO.write(0)

        self.rewardL = False

    def previewStim(self):
        if self.flag_preview:
            self.win.close()
        self.screen_dist = int(self.mice_dist_entry.get())
        self.flag_preview = True
        self.point_radius = int(self.point_radius_entry.get())
        self.visual_stim_angle_pos = int(self.visual_stim_angle_pos_entry.get())
        self.win = visual.Window(
            size=(1024, 600), fullscr=True, screen=2,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[-1, -1, -1], colorSpace='rgb',
            blendMode='avg', useFBO=True,
            units='pix')
        self.stim_pos, self.stim_radius = MiceEyeMathsTools.circle_pos(self.screen_dist, self.visual_stim_angle_pos,
                                                                       self.point_radius)
        self.screen_dist = self.screen_dist
        self.visual_stim1 = visual.Circle(win=self.win, radius=self.point_radius, units='pix', fillColor=[1, 1, 1],
                                          lineColor=[1, 1, 1], edges=128)
        self.visual_stim2 = visual.Circle(win=self.win, radius=self.point_radius, units='pix', fillColor=[1, 1, 1],
                                          lineColor=[1, 1, 1], edges=128)
        self.visual_stim3 = visual.Circle(win=self.win, radius=self.point_radius, units='pix', fillColor=[1, 1, 1],
                                          lineColor=[1, 1, 1], edges=128)

        self.visual_stim1.pos = [-self.stim_pos, 120]
        self.visual_stim2.pos = [self.stim_pos, 120]
        self.visual_stim3.pos = [0, 120]
        self.visual_stim1.draw()
        self.visual_stim2.draw()
        self.visual_stim3.draw()
        self.win.flip()

    def close_preview(self):
        self.rootp.destroy()
        self.cam.release()

    def clear(self):
        list = self.root.grid_slaves()
        for l in list:
            l.destroy()

    def close(self):
        self.root.destroy()
        self.text_file.close()


########################################################################################################################
#                 MAIN PROGRAMM                 #
########################################################################################################################

filename_V = r'C:\Users\taches-comportements\Pictures\Camera Roll\clignement.mp4'
# cfg = auxiliaryfunctions.read_config(r"C:\Users\taches-comportements\Desktop\TOM-net_v0.9-Enzo-2021-03-20\config.yaml")
modelfolder = (
    r'C:\Users\taches-comportements\Downloads\TOM_netv2-Enzo-2021-04-20 (1)\TOM_netv2-Enzo-2021-04-20\exported-models\DLC_TOM_netv2_mobilenet_v2_1.0_iteration-0_shuffle-1')
dlc_config = load_config(
    r"C:\Users\taches-comportements\Desktop\TOM-net_v0.9-Enzo-2021-03-20\dlc-models\iteration-0\TOM-net_v0.9Mar20-trainset95shuffle1\test\pose_cfg.yaml")
# pose_file_config['num_outputs'] = config_file.get('num_outputs', pose_file_config.get('num_outputs', 1))
today = date.today()
today = today.strftime('%d%m%Y')
pose_obj = PosePrediction(modelfolder)
user_interface = Gui('COM9', 2)
# water_reward = PeristalticPump('COM9', 2)
finished = False
trial_started = False
centered = False
start_calib = False
saccade_detected = False
inactivity_punition = False
punished = False
stim_finished = False
centered_stim_started = False
punish_finished = False
ioi = True  # TODO : Make a function to handle IOI via BNC cable
cam = Camera(0, user_interface.mice_name, 640, 360, 25, 0)
predicted_data = np.zeros((100000, 42))

# Parameters dictionnary
parameters = {
    "Mice name": user_interface.mice_name,
    "Number of trials": user_interface.trials_number,
    "Punition probability": user_interface.punition_proba,

    "Answer window duration": user_interface.answer_win,
    "Visual stim duration": user_interface.stim_dur,
    "Visual stim type": user_interface.stim_type,
    "Max duration of centered stim": user_interface.max_center_dur,
    "Visual stim radius": user_interface.point_radius,
    "Precision": user_interface.precision,
    "Centered fixation duration": user_interface.centered_dur,
    "Time between trials": user_interface.time_between_trials,
    "Visual stim angular position": user_interface.visual_stim_angle_pos,
    "Punish if inactivity": user_interface.inactivity_pun,
    "Punition duration": int(2),
    "training": user_interface.training,
    'Threshold': user_interface.thresh,
    'Screen distance': user_interface.mice_dist
}

screen = VisualStimulation(2, int(user_interface.visual_stim_angle_pos), int(user_interface.mice_dist),
                           int(user_interface.point_radius))
screen.blackScreen()
counter = 0
trial_counter = 0
start_centered_time = 0
pun_time_stmp = 0
stims = []
pupil_center = [[], []]
pupil_radius = []
pupil_velocity = []
variation_rate = []
azimutal_angle = []
elevation_angle = []

start_stmp = time.time()
centered_time_stmp = time.time()
time_stp_start = time.time()
time_stp_start_fps = time.time()
FPS = 0
reward = 0
# user_interface.start_lick_reporting()
while finished == False:

    # if counter == 116 :
    # finished = True
    frame = cam.getImage()
    if counter == 0:
        predicted_data[counter, :] = pose_obj.init_infer(frame)
        time_stp_start = time.time()
    time_stp = time.time() - time_stp_start
    time_stp_start = time.time()
    if counter > 1:
        FPS = int(1 / (time_stp))

    predicted_data[counter, :] = pose_obj.get(frame)
    # probability = pose_obj.checkProbability(predicted_data, counter)
    probability = True
    if probability:
        # Computation
        scale_f = MiceEyeMathsTools.scale_factor(counter, predicted_data)  # Scale factor
        pupil_radius.append(MiceEyeMathsTools.pupil_center_radius(counter, pupil_center, predicted_data))  # Radius of
        # the pupil + x,y coordinates of pupil center
        # MiceEyeMathsTools.variation_rate(counter, variation_rate, pupil_center)

        MiceEyeMathsTools.velocity(counter, pupil_velocity, pupil_center, time_stp)
        blink = MiceEyeMathsTools.global_variation_rate_blink(counter, predicted_data)
        if blink:
            azimutal_angle.append(math.nan)
            elevation_angle.append(math.nan)
        else:
            MiceEyeMathsTools.angular_position(counter, scale_f, pupil_radius, pupil_center, azimutal_angle,
                                               elevation_angle,
                                               predicted_data)

        if blink == True:  # Leave the trial if the mice blinks her eye to avoid false saccades detections
            trial_started = False
            user_interface.updatelogText('Mice blinked, restarting trial')
            user_interface.updateLog('Mice blinked, restarting trial...')
        # MiceEyeMathsTools.angular_position(counter, scale_f, pupil_radius, pupil_center, azimutal_angle, elevation_angle,
        # predicted_data)

        MiceEyeMathsTools.variation_rate_az(counter, variation_rate, azimutal_angle)
        # Save all the data in a file
        measures_str = "Pupil area : {} mm^2 / Azimutal angle : {}  degrees / Elevation angle : {}  degrees".format(
            round(((pupil_radius[counter] * (1 / scale_f)) ** 2) * math.pi, 3), round(azimutal_angle[counter], 3),
            round(elevation_angle[counter], 3))
        user_interface.updatelogText(measures_str)
        # Display the frame and angles on the GUI
        img = pose_obj.dispPredictionsv2(frame, counter, pupil_center, int(pupil_radius[counter]))
        user_interface.dispFrame(img)

        user_interface.updateAngle(round(azimutal_angle[counter], 3), round(elevation_angle[counter], 3),
                                   round(((pupil_radius[counter] * (1 / scale_f)) ** 2) * math.pi, 3), round(FPS, 2))
        # print(variation_rate[counter])
        if trial_started == False and time.time() - centered_time_stmp < parameters[
            "Max duration of centered stim"] and centered == False:
            if centered_stim_started == False:
                screen.stim('Center')
                centered_stim_started = True
                print('Centering start')
                user_interface.updatelogText(
                    'Centering start')
                user_interface.updateLog(
                    'Centering start')
            if azimutal_angle[counter] < 5 and azimutal_angle[counter] > -5 and elevation_angle[counter] < 5 and \
                    elevation_angle[counter] > -5:
                if start_centered_time == 0:
                    start_centered_time = time.time()
                elif time.time() - start_centered_time > parameters["Centered fixation duration"]:
                    centered = True
                    centered_stim_started = False
                    start_centered_time = 0
                    user_interface.updatelogText(
                        'Mice has fixed the center during {} s'.format(parameters["Centered fixation duration"]))
                    user_interface.updateLog(
                        'Mice has fixed the center during {} s'.format(parameters["Centered fixation duration"]))
            elif centered == False and trial_started == False:
                start_centered_time = 0


        elif trial_started == False and centered == False and (time.time() - centered_time_stmp) > parameters[
            "Max duration of centered stim"]:
            start_centered_time = 0
            centered = True
            centered_stim_started = False
            user_interface.updatelogText(
                'Mice did not fix the center during {} s, trial start'.format(parameters["Centered fixation duration"]))
            user_interface.updateLog(
                "Mice did not fix the center during {} s, trial start".format(parameters["Centered fixation duration"]))

        if trial_started == False and centered == True and ioi == True:
            if trial_counter > 1 and parameters["Visual stim type"] == 'Both':
                stims.append(screen.gambleStim(stims[trial_counter - 1], stims[trial_counter - 2]))

            elif parameters["Visual stim type"] == 'Both':
                stims.append(screen.gambleStim('Nasal', 'Temporal'))


            else:
                stims.append(parameters["Visual stim type"])

            trial_tmsp = time.time()
            screen.stim(str(stims[trial_counter]))
            trial_started = True
            user_interface.updatelogText('Trial {} started : '.format(trial_counter) + str(stims[trial_counter]))
            user_interface.updateLog('Trial {} started : '.format(trial_counter) + str(stims[trial_counter]))
            saccade_counter = 0
            stim_finished = False

        if trial_started == True and (time.time() - trial_tmsp) > parameters[
            "Visual stim duration"] and stim_finished == False and saccade_detected == True:
            # Remove stim
            print('test')
            screen.blackScreen()
            screen.win.flip()
            user_interface.updatelogText('Stim finished')
            user_interface.updateLog('Stim finished')
            stim_finished = True

        if trial_started == True and (time.time() - pun_time_stmp) > parameters[
            "Punition duration"] and punished == True and punish_finished == False and saccade_detected == True:
            # Remove blank screen punition if finished
            print('Solenn')
            screen.blackScreen()
            screen.win.flip()
            user_interface.updatelogText('Punition finished')
            user_interface.updateLog('Punition finished')
            punish_finished = True
            punished = False
            pun_time_stmp = 0

        if trial_started == True and (time.time() - trial_tmsp) < parameters["Answer window duration"]:

            # Saccade detection

            if variation_rate[counter] > parameters[
                "Threshold"]:  # and (azimutal_angle[counter] <= -1 or azimutal_angle[counter] >= 5) :
                saccade_counter += 1

                print('seuil déclanché : {}'.format(round(variation_rate[counter], 2)))
                print(pupil_velocity[counter])
                # if training mode is enabled, check if the mice did a saccade : TODO : Revoir cette fonction apres pratique en fonction de l'apprentissage
                if parameters["training"] and saccade_detected == False:
                    saccade_detected = True
                    user_interface.updatelogText('Rewarded')
                    user_interface.updateLog('Rewarded')
                    user_interface.dispense()
                    reward += 1
                    if azimutal_angle[counter] <= -5:
                        if stims[trial_counter] == 'Nasal saccade':
                            user_interface.dispense()
                            user_interface.dispense()
                            saccade_detected = True
                            user_interface.updatelogText('Rewarded')
                            user_interface.updateLog('Rewarded')
                            reward += 1
                    if azimutal_angle[counter] >= 5:
                        if stims[trial_counter] == 'Temporal saccade':
                            user_interface.dispense()
                            user_interface.dispense()
                            saccade_detected = True
                            user_interface.updatelogText('Rewarded')
                            user_interface.updateLog('Rewarded')
                            reward += 1
                    if (time.time() - trial_tmsp) > int(
                            (parameters["Answer window duration"])) and saccade_detected == True:
                        trial_started = False
                        saccade_detected = False
                        centered = False
                        punish_finished = False
                        punished = False
                        centered_time_stmp = time.time()
                        user_interface.updatelogText('Trial finished')
                        trial_counter += 1

            elif saccade_counter != 0 and saccade_detected == False and parameters["training"] == False:
                if azimutal_angle[counter] <= (
                        -parameters["Visual stim angular position"] + parameters["Precision"]) and azimutal_angle[
                    counter] >= (-parameters["Visual stim angular position"] - parameters["Precision"]):
                    user_interface.updatelogText('Nasal saccade')
                    if stims[trial_counter] == 'Nasal saccade':
                        user_interface.dispense()
                        saccade_detected = True
                        user_interface.updatelogText('Rewarded')
                        user_interface.updateLog('Rewarded')
                        reward += 1
                    elif screen.gamble_punition(user_interface.punition_proba) == 0:
                        saccade_detected = True
                        user_interface.updatelogText('Punition')
                        user_interface.updateLog('Punition')
                        screen.whiteScreen()
                        pun_time_stmp = time.time()
                    else:
                        saccade_detected = True
                        user_interface.updatelogText('Bad saccade but no punition')

                elif azimutal_angle[counter] <= (
                        parameters["Visual stim angular position"] + parameters["Precision"]) and azimutal_angle[
                    counter] >= (parameters["Visual stim angular position"] - parameters["Precision"]):
                    user_interface.updatelogText('Temporal saccade')
                    if stims[trial_counter] == 'Temporal saccade':
                        user_interface.dispense()
                        saccade_detected = True
                        user_interface.updatelogText('Rewarded')
                        user_interface.updateLog('Rewarded')
                        reward += 1
                    elif screen.gamble_punition(user_interface.punition_proba) == 0:
                        saccade_detected = True
                        user_interface.updatelogText('Punition')
                        user_interface.updateLog('Punition')
                        screen.whiteScreen()
                        pun_time_stmp = time.time()
                        punished = True
                    else:
                        saccade_detected = True
                        user_interface.updatelogText('Bad saccade but no punition')
                        user_interface.updateLog('Bad saccade but no punition')
                elif parameters["Punish if inactivity"] == True:
                    user_interface.updatelogText('Incomplete saccade : Punition')
                    user_interface.updateLog('Incomplete saccade : Punition')
                    screen.whiteScreen()
                    pun_time_stmp = time.time()
                    saccade_detected = True
                    punished = True
                else:
                    user_interface.updatelogText('Incomplete saccade but no punition')
                    user_interface.updateLog('Incomplete saccade but no punition')
                    saccade_detected = True


        elif trial_started == True and (time.time() - trial_tmsp) >= parameters["Answer window duration"]:
            if saccade_detected == False and punished == False and punish_finished == False and parameters[
                "Punish if inactivity"] == True:
                screen.whiteScreen()
                screen.win.flip()
                user_interface.updatelogText('No saccade : Punition')
                user_interface.updateLog('No saccade : Punition')
                punished = True
                pun_time_stmp = time.time()
            elif saccade_detected == False and punished == False and punish_finished == False and parameters[
                "Punish if inactivity"] == False and parameters["training"] == False:
                user_interface.updatelogText('No saccade : No Punition')
                user_interface.updateLog('No saccade : No Punition')
                punish_finished = True

            if (time.time() - trial_tmsp) > int((parameters["Answer window duration"] + parameters[
                "Time between trials"])) and saccade_detected == True:
                if saccade_detected == False:
                    user_interface.updatelogText('Mice did nothing during trial')

                trial_started = False
                saccade_detected = False
                centered = False
                punish_finished = False
                punished = False
                centered_time_stmp = time.time()
                user_interface.updatelogText('Trial finished')
                trial_counter += 1

    else:
        print('There was a probability of prediction below 90 %, frame not counted in the trial... ')

    if trial_counter == parameters["Number of trials"]:
        # Close the loop if all the trials are made
        finished = True

    counter += 1
    time.sleep(0.008)

    # input('t')

    # licks = water_reward.lick()# Check if mice licks
    # if licks == 1:
    # user_interface.updateLog('Mouse lick')
    # user_interface.updatelogText('Mouse lick')

cam.close()
user_interface.close()
print('Trials finished, please consult .txt files to get infos')
print('Number of rewards : {}'.format(reward))

# Stats
plt.plot(np.linspace(0, counter / 30, counter), azimutal_angle)
plt.xlabel('Time [s]')
plt.ylabel('Angle [degrees]')
plt.title(str(parameters['Mice name'] + ' : Azimutal angle of the eye'))
plt.grid()
plt.show()

# plt.plot(np.linspace(0,counter/30, counter), variation_rate)
# plt.xlabel('Time [s]')
# plt.ylabel('%')
# plt.title(str(parameters['Mice name'] +' : Var rate'))
# plt.grid()

# plt.show()

sys.exit()
