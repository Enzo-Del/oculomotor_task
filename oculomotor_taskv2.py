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
print('Importing TensorFlow ...')
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
import MiceEyeMathsTools
from datetime import datetime
from tkinter import scrolledtext
from threading import Thread

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
        self.out = cv2.VideoWriter(r'C:\Users\opto-delamarree\Desktop\TOM'+chr(92)+ str(today)+'_'+str(self.mice_name) +'.mp4', self.video_type, self.FPS, (self.res_width, self.res_height), grayscale)

    def getImage(self):
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
        self.pose = pose.flatten()
        return self.pose

    def dispPredictions(self, frame, pose, counter):
        for x_plt, y_plt, c in zip(self.x_range, self.y_range, self.colors):
            image = cv2.drawMarker(frame, (int(pose[counter, :][x_plt]), int(pose[counter, :][y_plt])), c,cv2.MARKER_STAR, 5, 2)

        img = Image.fromarray(image)
        imgtk = ImageTk.PhotoImage(image=img)
        return imgtk

    def checkProbability(self, pose, counter):
        # Check the probabilty of the pose prediciton
        if all(float(p) >= 0.9 for p in pose[counter, 2:3:47]):
            self.probability = True
        else:
            self.probability = False

        return self.probability

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
        self.water_spout_qty= water_spout_qty # in micro-liter
        self.lick_GPIO = self.board.digital[2]
        self.reward_GPIO = self.board.digital[6]
        self.reward_GPIO.write(0)
        self.it = pyfirmata.util.Iterator(self.board)
        self.it.start()
        self.lick_GPIO.mode = pyfirmata.INPUT
        self.time_ON = 0.05 #TODO : Change with pump calibration

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
            size=(1024, 600), fullscr=True, screen=window,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[-1, -1, -1], colorSpace='rgb',
            blendMode='avg', useFBO=True,
            units='pix')
        self.stim_pos, self.stim_radius = MiceEyeMathsTools.circle_pos(screen_dist, stim_angle_pos, stim_radius)
        self.screen_dist = screen_dist
        self.visual_stim = visual.Circle(win=self.win, radius=stim_radius, units='pix', fillColor=[1, 1, 1], lineColor=[1, 1, 1],
                             edges=128)
        self.win.flip()


    def stim(self, direction):
        self.win.color = [-1, -1, -1]
        if direction == 'Temporal' :
            self.visual_stim.pos = [-self.stim_pos, 0]
        if direction == 'Nasal':
            self.visual_stim.pos = [self.stim_pos, 0]
        if direction =='Center':
            self.visual_stim.pos = [0, 0]

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

    def gamble_punition(self, probability):
        draw = np.zeros([1, 1])
        draw[0, 0] = choice([0, 1], 1, p=[(probability/10), (1-(probability/10))])
        return draw[0, 0]


class Gui():
    def __init__(self, pump):
        self.pump = pump
        self.root = tk.Tk()
        self.root.title("Training device for an oculomotor behavioral task")
        self.root.geometry('700x700')
        self.root.tk.call('wm', 'iconphoto', self.root._w, tk.PhotoImage(file='icon.PNG'))
        tk.Label(self.root, text="Mice name :").grid(row=0, column=0,sticky = W)
        tk.Label(self.root, text="Trials number :").grid(row=1, column=0,sticky = W)
        tk.Label(self.root, text="Probability of punition (/10) :").grid(row=2, column=0,sticky = W)
        tk.Label(self.root, text="Answer window duration (> or = to stim duration) [s] :").grid(row=3, column=0,sticky = W)
        tk.Label(self.root, text="Stim duration [s] :").grid(row=4, column=0,sticky = W)
        tk.Label(self.root, text="Type of visual stimulation :").grid(row=5, column=0,sticky = W)
        tk.Label(self.root, text="Maximum duration of visual centered stim [s] :").grid(row=7, column=0, sticky=W)
        tk.Label(self.root, text="Centered fixation time [s] :").grid(row=8, column=0, sticky=W)
        tk.Label(self.root, text="Point radius [degrees] :").grid(row=9, column=0, sticky=W)
        tk.Label(self.root, text="Accepted precision by the mice [degrees]:").grid(row=10, column=0, sticky=W)
        tk.Label(self.root, text="Time to wait between trials [s] :").grid(row=11, column=0, sticky=W)
        tk.Label(self.root, text="Visual stimuli position [degrees] :").grid(row=12, column=0, sticky=W)
        tk.Label(self.root, text="WARNING: Make sure that the mice").grid(row=14, column=0,sticky = W)
        tk.Label(self.root, text= "is well head-fixed and that the ").grid(row=15, column=0,sticky = W)
        tk.Label(self.root, text= "cameras and screens are centered !").grid(row=16, column=0,sticky = W)
        self.check_var = tk.IntVar()
        self.button_var = tk.IntVar()
        self.mice_name_entry = tk.Entry(self.root)
        self.mice_name_entry.insert(END, '310')
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
        self.inactivity_pun_widget = tk.Checkbutton(self.root, text='Inactivity punition', variable=self.check_var1, onvalue=1, offvalue=0)
        self.inactivity_pun_widget.grid(row=6, column=0, sticky=W)
        self.training_widget = tk.Checkbutton(self.root, text='Training', variable=self.check_var2,
                                                    onvalue=1, offvalue=0)
        self.training_widget.grid(row=13, column=0, sticky=W)
        self.centered_dur_entry.grid(row=8, column=1)
        self.point_radius_entry = tk.Entry(self.root)
        self.point_radius_entry.insert(END, '25')
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
        self.stim_type_cbox = ttk.Combobox(self.root, values = ['Both', 'Temporal','nasal'])
        self.stim_type_cbox.current(0)
        self.stim_type_cbox.grid(row=5, column=1)
        self.go_button = tk.Button(self.root, text="Go !", command=self.start)
        self.go_button.grid(column=3, row=15)
        self.water_button = tk.Button(self.root, text="Dispense sequence of water", command=self.dispense)
        self.water_button.grid(column=3, row=15)
        self.go_button_var = tk.IntVar()
        self.water_button_var = tk.IntVar()
        self.go_button.wait_variable(self.go_button_var)
        self.root.children.clear()
        self.root.update()

        # Video live update
        self.app = Frame(self.root, bg="white")
        self.app.grid(row=0, column=0)
        self.lmain = Label(self.app)
        self.lmain.grid(row=0, column=0)
        self.az_pos_label = tk.Label(self.root, text = 'Azimutal angle :')
        self.el_pos_label = tk.Label(self.root, text = 'Elevation angle :')
        self.az_pos_label.grid(row=2, column=0)
        self.el_pos_label.grid(row=3, column=0)
        self.root.update()
        # Init log box
        self.log = scrolledtext.ScrolledText(self.root, width=70, height=15)
        self.log.grid(row=4, column=0)
        self.log.see(END)

        # Init log text file
        try :
            self.text_file = open(
                r"C:\Users\opto-delamarree\PycharmProjects\oculomotor_task" + chr(92) + str(date.today()) + '_' + str(
                    self.mice_name) + '.txt', 'w')
            self.text_file.write("Experiment started on : " + datetime.now().strftime("%d/%m/%Y %H:%M:%S") +"\r\n")
        except :
            print('An error as occured while trying to create a new text file.')



    def start(self):
        # Get experience parameters
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
        self.clear() # Clear widgets

    def dispense(self):
        self.pump.reward()


    def dispFrame(self, frame):
        self.lmain.imgtk = frame
        self.lmain.configure(image=frame)
        self.root.update()

    def updateAngle(self, az, el):
        self.az_pos_label.configure(text='Azimutal angle :' + str(az) + ' degrees')
        self.el_pos_label.configure(text='Elevation angle :' + str(el)+ ' degrees')

    def updateLog(self, content):
        self.log.insert(END, datetime.now().strftime('%H:%M:%S.%f : ') + str(content) + "\r\n")
        self.log.see(END)
        self.root.update()

    def updatelogText(self, content):

        # Save content to text file
        self.text_file.write(datetime.now().strftime('%H:%M:%S.%f')[:-3] + str(content) + "\r\n" )

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

filename_V = r'C:\Users\opto-delamarree\Desktop\presentation\WIN_20200808_14_09_02_Prodownsampled.mp4'
cfg = auxiliaryfunctions.read_config(r"C:\Users\opto-delamarree\Desktop\Eye_validation-Enzo-2020-08-08\config.yaml")
modelfolder =(r"C:\Users\opto-delamarree\Desktop\Eye_validation-Enzo-2020-08-08\dlc-models\iteration-0\Eye_validationAug8-trainset95shuffle1")
dlc_config = load_config(r"C:\Users\opto-delamarree\Desktop\Eye_validation-Enzo-2020-08-08\dlc-models\iteration-0\Eye_validationAug8-trainset95shuffle1\test\pose_cfg.yaml")
today = date.today()
today = today.strftime('%d%m%Y')
#water_reward = PeristalticPump('COM9', 2)
user_interface = Gui(water_reward)
finished = False
trial_started = False
centered = False
start_calib = False
saccade_detected =False
inactivity_punition = False
punished = False
stim_finished = False
centered_stim_started = False
punish_finished = False
ioi = True #TODO : Make a function to handle IOI via BNC cable
pose_obj = PosePrediction(cfg,modelfolder, dlc_config, 640, 360)
cam = Camera(filename_V, user_interface.mice_name, 640, 360,30,0)
predicted_data = np.zeros((100000, dlc_config['num_outputs'] * 3 * len(dlc_config['all_joints_names'])))
screen = VisualStimulation(1, int(user_interface.visual_stim_angle_pos), 6, int(user_interface.point_radius))
screen.blackScreen()


# Parameters dictionnary
parameters = {
    "Mice name" : user_interface.mice_name,
    "Number of trials" : user_interface.trials_number,
    "Punition probability" : user_interface.punition_proba,
    "Answer window duration" : user_interface.answer_win,
    "Visual stim duration" : user_interface.stim_dur,
    "Visual stim type" : user_interface.stim_type,
    "Max duration of centered stim" : user_interface.max_center_dur,
    "Visual stim radius" : user_interface.point_radius,
    "Precision" : user_interface.precision,
    "Centered fixation duration" : user_interface.centered_dur,
    "Time between trials" : user_interface.time_between_trials,
    "Visual stim angular position" : user_interface.visual_stim_angle_pos,
    "Punish if inactivity" : user_interface.inactivity_pun,
    "Punition duration" : int(2),
    "training" : user_interface.training
}

counter = 0
trial_counter = 0
start_centered_time = 0
pun_time_stmp = 0
stims = []
pupil_center = [[],[]]
pupil_radius = []
pupil_velocity = []
variation_rate= []
azimutal_angle = []
elevation_angle = []
licks = []
start_stmp = time.time()
centered_time_stmp = time.time()
time_stp_start = 0

while finished ==False:

    time_stp = time.time() - time_stp_start
    time_stp_start = time_stp
    frame = cam.getImage()
    predicted_data[counter, :] = pose_obj.get(frame)
    probability = pose_obj.checkProbability(predicted_data, counter)
    if probability :
        #Computation
        scale_f = MiceEyeMathsTools.scale_factor(counter, predicted_data) #Scale factor
        pupil_radius.append(MiceEyeMathsTools.pupil_center_radius(counter,pupil_center, predicted_data)) #Radius of
            # the pupil + x,y coordinates of pupil center
        MiceEyeMathsTools.variation_rate(counter, variation_rate, pupil_center)
        MiceEyeMathsTools.velocity(counter,pupil_velocity, pupil_center, time_stp)

        MiceEyeMathsTools.angular_position(counter, scale_f, pupil_radius, pupil_center, azimutal_angle, elevation_angle,
                                           predicted_data)

        blink = MiceEyeMathsTools.global_variation_rate_blink(counter, predicted_data)
        #Save all the data in a file
        measures_str = "Pupil area : {} mm^2 / Azimutal angle : {}  degrees / Elevation angle : {}  degrees".format(round(((pupil_radius[counter]*(10/scale_f))**2)*math.pi, 3), round(azimutal_angle[counter], 3), round(elevation_angle[counter], 3))
        user_interface.updatelogText(measures_str)
        #Display the frame and angles on the GUI
        img = pose_obj.dispPredictions(frame, predicted_data, counter)
        user_interface.dispFrame(img)
        #licks[counter] = water_reward.lick()# Check if mice licks
        user_interface.updateAngle(round(azimutal_angle[counter],3), round(elevation_angle[counter], 3))

        if  trial_started ==False and time.time()-centered_time_stmp < parameters["Max duration of centered stim"] and centered ==False:
            if centered_stim_started == False:
                screen.stim('Center')
                centered_stim_started = True
                print('Centering start')
                user_interface.updatelogText(
                    'Centering start')
                user_interface.updateLog(
                    'Centering start')
            if azimutal_angle[counter] < 2 and azimutal_angle[counter] > -2 and elevation_angle[counter] < 2 and elevation_angle[counter] > -2  :
                if start_centered_time == 0:
                    start_centered_time = time.time()
                elif time.time()-start_centered_time > parameters["Centered fixation duration"]:
                    centered = True
                    centered_stim_started = False
                    start_centered_time = 0
                    user_interface.updatelogText(
                        'Mice has fixed the center during {} s'.format(parameters["Centered fixation duration"]))
                    user_interface.updateLog('Mice has fixed the center during {} s'.format(parameters["Centered fixation duration"]))
            elif centered ==False and trial_started ==False :
                start_centered_time = 0


        elif trial_started ==False and centered == False and (time.time()-centered_time_stmp) > parameters["Max duration of centered stim"]:
                start_centered_time =0
                centered =True
                centered_stim_started = False
                user_interface.updatelogText(
                    'Mice did not fix the center during {} s, trial start anyway'.format(parameters["Centered fixation duration"]))
                user_interface.updateLog("Mice did not fix the center during {} s, trial start anyway".format(parameters["Centered fixation duration"]))


        if trial_started == False and centered == True and ioi ==True:
            if trial_counter > 1 and parameters["Visual stim type"] =='Both':
                stims.append(screen.gambleStim(stims[trial_counter-1], stims[trial_counter-2]))

            elif parameters["Visual stim type"] =='Both' :
                stims.append(screen.gambleStim('Nasal', 'Temporal'))


            else :
                stims.append(parameters["Visual stim type"])

            trial_tmsp = time.time()
            screen.stim(str(stims[trial_counter]))
            trial_started = True
            user_interface.updatelogText('Trial {} started : '.format(trial_counter) + str(stims[trial_counter]))
            user_interface.updateLog('Trial {} started : '.format(trial_counter) + str(stims[trial_counter]))
            saccade_counter = 0
            stim_finished = False

        if trial_started == True and (time.time() - trial_tmsp) > parameters["Visual stim duration"] and stim_finished == False :
            #Remove stim
            screen.blackScreen()
            screen.win.flip()
            user_interface.updatelogText('Stim finished')
            user_interface.updateLog('Stim finished')
            stim_finished = True


        if trial_started ==True and (time.time() - pun_time_stmp) > parameters["Punition duration"] and punished == True and punish_finished == False:
            #Remove blank screen punition if finished

            screen.blackScreen()
            screen.win.flip()
            user_interface.updatelogText('Punition finished')
            user_interface.updateLog('Punition finished')
            punish_finished = True
            punished = False
            pun_time_stmp = 0

        if trial_started == True and (time.time()-trial_tmsp) < parameters["Answer window duration"] :
            if blink ==True: #Leave the trial if the mice blinks her eye to avoid false saccades detections
                trial_started=False
                user_interface.updatelogText('Mice blinked, restarting trial')
                user_interface.updateLog('Mice blinked, restarting trial')

            #Saccade detection
            elif variation_rate[counter] > 5 and pupil_velocity[counter] > 2 :
                saccade_counter += 1
                # if training mode is enabled, check if the mice did a saccade : TODO : Revoir cette fonction apres pratique en fonction de l'apprentissage
                if parameters["training"] and saccade_detected == False:
                    saccade_detected = True
                    user_interface.updatelogText('Rewarded')
                    user_interface.updateLog('Rewarded')
                    # water_reward.reward()

            elif saccade_counter !=0 and saccade_detected == False and parameters["training"] == False :
                if azimutal_angle[counter] <= (-parameters["Visual stim angular position"] + parameters["Precision"]) and azimutal_angle[counter] >= (-parameters["Visual stim angular position"] - parameters["Precision"]) :
                    user_interface.updatelogText('Nasal saccade')
                    if stims[trial_counter] == 'Nasal saccade' :
                        #water_reward.reward()
                        saccade_detected = True
                        user_interface.updatelogText('Rewarded')
                        user_interface.updateLog('Rewarded')
                    elif screen.gamble_punition(user_interface.punition_proba) == 0:
                        saccade_detected = True
                        user_interface.updatelogText('Punition')
                        user_interface.updateLog('Punition')
                        screen.whiteScreen()
                        pun_time_stmp = time.time()
                    else :
                        saccade_detected = True
                        user_interface.updatelogText('Bad saccade but no punition')

                elif azimutal_angle[counter] <= (parameters["Visual stim angular position"] + parameters["Precision"]) and azimutal_angle[counter] >= (parameters["Visual stim angular position"] - parameters["Precision"]) :
                    user_interface.updatelogText('Temporal saccade')
                    if stims[trial_counter] == 'Temporal saccade' :
                        #water_reward.reward()
                        saccade_detected = True
                        user_interface.updatelogText('Rewarded')
                        user_interface.updateLog('Rewarded')
                    elif screen.gamble_punition(user_interface.punition_proba) == 0:
                        saccade_detected = True
                        user_interface.updatelogText('Punition')
                        user_interface.updateLog('Punition')
                        screen.whiteScreen()
                        pun_time_stmp = time.time()
                        punished = True
                    else :
                        saccade_detected = True
                        user_interface.updatelogText('Bad saccade but no punition')
                        user_interface.updateLog('Bad saccade but no punition')
                elif parameters["Punish if inactivity"] == True :
                    user_interface.updatelogText('Incomplete saccade : Punition')
                    user_interface.updateLog('Incomplete saccade : Punition')
                    screen.whiteScreen()
                    pun_time_stmp = time.time()
                    saccade_detected = True
                    punished = True
                else :
                    user_interface.updatelogText('Incomplete saccade but no punition')
                    user_interface.updateLog('Incomplete saccade but no punition')
                    saccade_detected = True


        elif trial_started ==True and (time.time()-trial_tmsp) >= parameters["Answer window duration"]:
            if saccade_detected == False and punished ==False and punish_finished == False and parameters["Punish if inactivity"] == True:
                screen.whiteScreen()
                screen.win.flip()
                user_interface.updatelogText('No saccade : Punition')
                user_interface.updateLog('No saccade : Punition')
                punished = True
                pun_time_stmp = time.time()
            elif saccade_detected == False and punished ==False and punish_finished == False and parameters["Punish if inactivity"] == False :
                user_interface.updatelogText('No saccade : No Punition')
                user_interface.updateLog('No saccade : No Punition')
                punish_finished = True

            if (time.time()-trial_tmsp) > int((parameters["Answer window duration"] + parameters["Time between trials"])) :
                if saccade_detected == False :

                    user_interface.updatelogText('Mice did nothing during trial')

                trial_started = False
                saccade_detected =False
                centered = False
                punish_finished = False
                punished = False
                centered_time_stmp = time.time()
                user_interface.updatelogText('Trial finished')
                trial_counter += 1

    else :
        print('There was a probability of prediction below 90 %, frame not counted in the trial. ')

    if trial_counter == parameters["Number of trials"]:
        #Close the loop of all the trials are made
        finished =True
    counter += 1

cam.close()
user_interface.close()


