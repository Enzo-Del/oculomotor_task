from psychopy import visual, core
from numpy.random import choice
import numpy as np
import time
import MiceEyeMathsTools
class VisualStimulation():
    def __init__(self, window, stim_angle_pos, screen_dist, stim_radius):

        self.win = visual.Window(
            size=(1024, 600), fullscr=True, screen=window,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[-1, -1, -1], colorSpace='rgb',
            blendMode='avg', useFBO=True,
            units='pix')
        self.stim_pos = int(MiceEyeMathsTools.circle_pos(screen_dist, stim_angle_pos))
        print(self.stim_pos)
        self.screen_dist = screen_dist
        self.stim_radius = stim_radius
        self.visual_stim = visual.Circle(win=self.win, radius=stim_radius, units='pix', fillColor=[1, 1, 1], lineColor=[1, 1, 1],
                             edges=128)
        self.win.flip()


    def stim(self, direction):
        self.win.color = [-1, -1, -1]
        if direction == 'temporal' :
            self.visual_stim.pos = [-self.stim_pos, 0]
        if direction == 'nasal':
            self.visual_stim.pos = [self.stim_pos, 0]
        if direction =='center':
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




screen = VisualStimulation(1, 5, 6, 50)
screen.stim('nasal')
time.sleep(10)


