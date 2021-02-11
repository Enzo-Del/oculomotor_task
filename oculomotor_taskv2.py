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
print('Loading TensorFlow ...')
import tensorflow as tf
print('Done')
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
print('Loading GPU ...')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
print('Done')
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
        today = date.today()
        today = today.strftime('%d%m%Y')
        self.out = cv2.VideoWriter(r'C:\Users\opto-delamarree\Desktop\TOM' +chr(92)+ str(today)+'_'+str(self.mice_name) +'.mp4' , self.video_type, (res_width, res_height),0)

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

        if self.config_file['snapshotindex'] == 'all':
            print(
                "Snapshotindex is set to 'all' in the config.yaml file. Running video analysis with all snapshots is "
                "very costly! Use the function 'evaluate_network' to choose the best the snapshot. For now, changing snapshot index to -1!")
            snapshotindex = -1
        else:
            snapshotindex = config_file['snapshotindex']

        increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
        Snapshots = Snapshots[increasing_indices]
        print("Using %s" % Snapshots[snapshotindex], "for model", self.model_folder)
        self.pose_file_config['init_weights'] = os.path.join(self.model_folder, 'train', Snapshots[snapshotindex])
        self.pose_file_config['batch_size'] = self.config_file['batch_size']
        self.pose_file_config['num_outputs'] = self.config_file.get('num_outputs', self.pose_file_config.get('num_outputs', 1))
        self.frame_batch = np.empty((batchsize, res_width, res_height, 3), dtype='ubyte')
        self.sess, self.inputs, self.outputs = predict.setup_GPUpose_prediction(self.pose_file_config)
        self.pose_tensor = predict.extract_GPUprediction(self.outputs, self.pose_file_config)  # extract_output_tensor(outputs, dlc_cfg)
        self.colors = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0), (240, 32, 160), (240, 32, 160),
                  (240, 32, 160)]

    def get(self, frame):
        frame = img_as_ubyte(frame)
        pose = self.sess.run(self.pose_tensor,feed_dict={self.inputs: np.expand_dims(frame, axis=0).astype(float)})
        pose[:, [0, 1, 2]] = pose[:, [1, 0, 2]]
        pose = pose.flatten()
        return pose

    def disp_predictions(self, frame, pose, counter):
        for x_plt, y_plt, c in zip(self.x_range, self.y_range, self.colors):
            image = cv2.drawMarker(frame, (int(pose[counter, :][x_plt]), int(pose[counter, :][y_plt])), c,cv2.MARKER_STAR, 5, 2)

        img = Image.fromarray(image)
        imgtk = ImageTk.PhotoImage(image=img)
        return imgtk


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

    def gamble_stim(self, past_stim1, past_stim2):
        draw = np.zeros([1, 1])
        draw[0, 0] = choice([0, 1], 1, p=[0.5, 0.5])
        if draw[0, 0] ==1 :
            if past_stim1 == past_stim2 =='Temporal' :
                stim = 'Nasal'
            else :
                stim = 'Temporal'
        else :
            if past_stim1 == past_stim2 == 'Nasal':
                stim = 'Temporal'
            else :
                stim = 'Nasal'

        return stim



class Gui():
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Training device for an oculomotor behavioral task")
        self.root.geometry('600x600')
        self.root.tk.call('wm', 'iconphoto', self.root._w, tk.PhotoImage(file='icon.PNG'))
        tk.Label(self.root, text="Mice name :").grid(row=0, column=0,sticky = W)
        tk.Label(self.root, text="Trials number :").grid(row=1, column=0,sticky = W)
        tk.Label(self.root, text="Probability of punition :").grid(row=2, column=0,sticky = W)
        tk.Label(self.root, text="Answer window duration [s] :").grid(row=3, column=0,sticky = W)
        tk.Label(self.root, text="Stim duration [s] :").grid(row=4, column=0,sticky = W)
        tk.Label(self.root, text="Type of visual stimulation :").grid(row=5, column=0,sticky = W)
        tk.Label(self.root, text="Maximum duration of visual centered stim [s] :").grid(row=6, column=0, sticky=W)
        tk.Label(self.root, text="Point radius [degrees] :").grid(row=8, column=0, sticky=W)
        tk.Label(self.root, text="WARNING : Make sure that the mice").grid(row=9, column=0,sticky = W)
        tk.Label(self.root, text= "is well head-fixed and that the ").grid(row=10, column=0,sticky = W)
        tk.Label(self.root, text= "cameras and screens are centered !").grid(row=11, column=0,sticky = W)
        self.check_var = tk.IntVar()
        self.pun_check = tk.Checkbutton(self.root, text='Punition ?', variable=self.check_var, onvalue=1, offvalue=0,
                                      command=self.punition)
        self.pun_check.grid(row = 7, column = 0,sticky = W)
        self.button_var = tk.IntVar()
        self.mice_name_entry = tk.Entry(self.root)
        self.mice_name_entry.grid(row=0, column=1)
        self.trials_number_entry = tk.Entry(self.root)
        self.trials_number_entry.grid(row=1, column=1)
        self.punition_proba_entry = tk.Entry(self.root)
        self.punition_proba_entry.grid(row=2, column=1)
        self.answer_win_entry = tk.Entry(self.root)
        self.answer_win_entry.grid(row=3, column=1)
        self.stim_dur_entry = tk.Entry(self.root)
        self.stim_dur_entry.grid(row=4, column=1)
        self.center_dur_entry = tk.Entry(self.root)
        self.center_dur_entry.grid(row=6, column=1)
        self.point_radius_entry = tk.Entry(self.root)
        self.point_radius_entry.grid(row=8, column=1)
        self.stim_type_cbox = ttk.Combobox(self.root, values = ['Both', 'Temporal','nasal'])
        self.stim_type_cbox.current(0)
        self.stim_type_cbox.grid(row=5, column=1)
        self.go_button = tk.Button(self.root, text="Go !", command=self.start)
        self.go_button.grid(column=3, row=9)
        self.go_button_var = tk.IntVar()
        self.go_button.wait_variable(self.go_button_var)
        self.root.children.clear()
        self.root.update()

        # Video live update
        self.app = Frame(self.root, bg="white")
        self.app.grid()
        self.lmain = Label(self.app)
        self.lmain.grid()



    def start(self):

        self.go_button_var.set(0)
        self.mice_name = str(self.mice_name_entry.get())
        self.trials_number = int(self.trials_number_entry.get())
        self.punition_proba = int(self.punition_proba_entry.get())
        self.answer_win = int(self.answer_win_entry.get())
        self.stim_dur = int(self.stim_dur_entry.get())
        self.stim_type = str(self.stim_type_cbox.get())
        self.center_dur = int(self.center_dur_entry.get())
        self.point_radius = int(self.center_dur_entry.get())
        self.stim_type = str(self.stim_type_cbox.get())

    def punition(self):

        if self.check_var.get() == 1:
            self.punition = True
        elif self.check_var.get() == 0:
            self.punition = False


    def disp_frame(self, frame):
        self.lmain.imgtk = frame
        self.lmain.configure(image=frame)

    def close(self):
        self.root.destroy()



########################################################################################################################
                          #                 FUNCTIONS DEFINITION                 #
########################################################################################################################







########################################################################################################################
                             #                 MAIN PROGRAMM                 #
########################################################################################################################

filename_V = r'C:\Users\opto-delamarree\Desktop\presentation\WIN_20200808_14_09_02_Prodownsampled.mp4'
cfg = auxiliaryfunctions.read_config(r"C:\Users\opto-delamarree\Desktop\Eye_validation-Enzo-2020-08-08\config.yaml")
modelfolder =(r"C:\Users\opto-delamarree\Desktop\Eye_validation-Enzo-2020-08-08\dlc-models\iteration-0\Eye_validationAug8-trainset95shuffle1")
dlc_config = load_config(r"C:\Users\opto-delamarree\Desktop\Eye_validation-Enzo-2020-08-08\dlc-models\iteration-0\Eye_validationAug8-trainset95shuffle1\test\pose_cfg.yaml")
user_interface = Gui()
today = date.today()
today = today.strftime('%d%m%Y')
finished = False
trial_started = False
centered = False
start_calib = False
ioi = True #TODO : Make a function to handle IOI via BNC cable
pose_obj = PosePrediction(cfg,modelfolder, dlc_config, 640, 360)
cam = Camera(filename_V, user_interface.mice_name, 640, 360,30,0)
predicted_data = np.zeros((100000, dlc_config['num_outputs'] * 3 * len(dlc_config['all_joints_names'])))
screen = VisualStimulation(1, 5, 6, user_interface.point_radius)
screen.black_screen()
#Open log file
text_file= open(r"C:\Users\opto-delamarree\PycharmProjects\oculomotor_task" +chr(92)+ str(today) + '_' + str(user_interface.mice_name) + '.txt' , 'w')
counter = 0
trial_counter = 0
stims = []
pupil_center = []
pupil_radius = []
pupil_velocity = []
azimutal_angle = []
elevation_angle = []
start_tmsp = time.time()

while finished ==False:
    frame = cam.get_image()
    predicted_data[counter, :] = pose_obj.get(frame)
    #Computation
    scale_f = MiceEyeMathsTools.scale_factor(counter, predicted_data) #Scale factor
    pupil_radius[counter] = MiceEyeMathsTools.pupil_center_radius(counter,pupil_center, predicted_data)#Radius of
        # the pupil + x,y coordinates of pupil center

    MiceEyeMathsTools.velocity(counter,pupil_velocity, pupil_center)
    MiceEyeMathsTools.angular_position(counter, scale_f, pupil_radius, pupil_center, azimutal_angle, elevation_angle,
                                       predicted_data)
    blink = MiceEyeMathsTools.global_variation_rate_blink(counter, predicted_data)
    #Display the frame on the GUI
    img = pose_obj.disp_predictions(frame, predicted_data, counter)
    user_interface.disp_frame(img)
    counter += 1
    #TODO : Afficher position angulaire absolue

    if azimutal_angle[counter] < 2 and azimutal_angle[counter] > -2 and elevation_angle[counter] < 2 and elevation_angle[counter] > -2 :
        pass
    if trial_started == False and centered == True and ioi ==True:
        if trial_counter > 1 and user_interface.stim_type =='Both':
            stim = screen.gamble_stim(stims[trial_counter-1], stims[trial_counter-2])
            stims[trial_counter] = stim
        elif user_interface.stim_type =='Both' :
            stim = screen.gamble_stim('Nasal', 'Temporal')
            stims[trial_counter] = stim
        else :
            stim = user_interface.stim_type
            stims[trial_counter] = stim

        trial_tmsp = time.time()
        screen.stim(stim)
        trial_started = True

cam.close()
user_interface.close()