#-*- coding: utf-8 -*-
"""
Created on Sun Mar  8 11:48:02 2020
Last modified on
@author: Enzo Delamarre
Version : mono
"""

########################################################################################################################
                          #                 LIBRAIRIES IMPORT                 #
########################################################################################################################

from __future__ import absolute_import, division
import math
from datetime import date
import tensorflow as tf
import numpy as np
import deeplabcut
import cv2
import os
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter.ttk import Label, Style
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
import time
import psychopy
from psychopy import visual, core
from numpy.random import choice
from PIL import ImageTk, Image
from tkinter import messagebox
import reward
import MiceEyeMathsTools

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
        self.cam.set(3,self.res_width) # Resolution change
        self.cam.set(4, self.res_height)
        self.video_type = cv2.VideoWriter_fourcc(*'mp4v') # Video format
        self.record = False
        self.out = cv2.VideoWriter(r'C:\Users\opto-delamarree\Desktop\TOM')

    def get_image(self):
        global frame
        try :
            ret, frame = self.cam.read()
        except :
            print('Camera object is not reading any frame, check if camera is wired or already occupied')
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.out.write(frame)
        return frame

    def close(self):
        self.cam.release()
        self.out.release()


class PosePrediction():
    """ Class for DLC prediction with DLC functions
    """
    def __init__(self, config_file, model_folder, pose_file_config, res_width, res_height):
        # Hard coded variables
        batchsize = 1
        batch_ind = 0
        batch_num = 0

        self.config_file = config_file
        self.model_folder = model_folder
        self.pose_file_config = pose_file_config
        self.x_range = list(range(0, (3 * len(self.pose_file_config['all_joints_names'])), 3))
        self.y_range = list(range(1, (3 * len(self.pose_file_config['all_joints_names'])), 3))
        self.pose_file_config['num_outputs'] = self.config_file.get('num_outputs', self.pose_file_config.get('num_outputs', 1))
        try:
            Snapshots = np.array(
                [fn.split('.')[0] for fn in os.listdir(os.path.join(self.model_folder, 'train')) if "index" in fn])
        except FileNotFoundError:
            raise FileNotFoundError(
                "Snapshots not found! It seems the dataset for shuffle %s has not been trained/does not exist.\n "
                "Please train it before using it to analyze videos.\n Use the function 'train_network' to train the network for shuffle %s." % (
                    1, 1))

        if self.pose_file_config['snapshotindex'] == 'all':
            print(
                "Snapshotindex is set to 'all' in the config.yaml file. Running video analysis with all snapshots is "
                "very costly! Use the function 'evaluate_network' to choose the best the snapshot. For now, changing snapshot index to -1!")
            snapshotindex = -1
        else:
            snapshotindex = pose_file_config['snapshotindex']

        increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
        Snapshots = Snapshots[increasing_indices]
        print("Using %s" % Snapshots[snapshotindex], "for model", self.model_folder)
        self.config_file['init_weights'] = os.path.join(self.model_folder, 'train', Snapshots[snapshotindex])
        self.config_file['batch_size'] = self.pose_file_config['batch_size']
        self.frame_batch = np.empty((batchsize, res_width, res_height, 3), dtype='ubyte')
        self.sess, self.inputs, self.outputs = predict.setup_GPUpose_prediction(self.config_file)
        self.pose_tensor = predict.extract_GPUprediction(self.outputs, self.config_file)  # extract_output_tensor(outputs, dlc_cfg)


    def get(self, frame):
        frame = img_as_ubyte(frame)
        pose = self.sess.run(self.pose_tensor,feed_dict={self.inputs: np.expand_dims(frame, axis=0).astype(float)})
        pose[:, [0, 1, 2]] = pose[:, [1, 0, 2]]
        pose = pose.flatten()
        return pose

class ArduinoNano():

    def __init__(self, com_port):
        self.com_port = com_port
        try :
            self.board = pyfirmata.Arduino(com_port)
            print("Communication Successfully started with arduinoNano at {}".format(self, com_port))
        except :
            print("ArduinoNano not connected or wrong COM port, please check")

class PeristalticPump(ArduinoNano):


    def __init__(self, com_port, water_spout_qty):
        ArduinoNano.__init__(self, com_port)
        self.water_spout_qty= water_spout_qty
        self.lick_GPIO = self.board.digital[2]
        self.reward_GPIO = self.board.digital[6]
        self.reward_GPIO.write(0)
        self.it = pyfirmata.util.Iterator(self.board)
        self.it.start()
        self.lick_GPIO.mode = pyfirmata.INPUT
        self.time_ON = 0.05 # Change with pump calibration

    def reward(self):
        self.reward_GPIO.write(1)
        time.sleep(self.time_ON)
        self.reward_GPIO.write(0)

    def lick(self):
        if self.lick_GPIO.read() == 1:
            lick = True
            time.sleep(0.05)
        else :
            lick = False
        return lick



class VisualStimulation():
    def __init__(self, window, stim_angle_pos, screen_dist, stim_radius):

        self.win = visual.Window(
            size=(1024, 600), fullscr=False, screen=window,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[-1, -1, -1], colorSpace='rgb',
            blendMode='avg', useFBO=True,
            units='deg')
        self.stim_pos = MiceEyeMathsTools.circle_pos(screen_dist, stim_angle_pos)
        self.screen_dist = screen_dist
        self.stim_radius = stim_radius
        self.stim = visual.Circle(win=self.win, radius=stim_radius, units='deg', fillColor=[1, 1, 1], lineColor=[1, 1, 1],
                             edges=128)
        self.win.flip()


    def stim(self, direction):
        self.win.color = [-1, -1, -1]
        if 'temporal' in direction:
            self.stim.pose = [-self.stim_pos, 0]
        if 'nasal' in direction:
            self.stim.pose = [self.stim_pos, 0]
        if 'center' in direction:
            self.stim.pose = [0, 0]

        self.stim.draw()
        self.stim.flip()

    def punition(self, pun_dur):

        self.win.color = [1, 1, 1]
        time.sleep(pun_dur)
        self.win.color = [-1, -1, -1]
        self.win.flip()

    def black_screen(self):
        self.win.color = [-1, -1, -1]
        self.win.flip()


########################################################################################################################
                          #                 FUNCTIONS DEFINITION                 #
########################################################################################################################








########################################################################################################################
                             #                 MAIN PROGRAMM                 #
########################################################################################################################