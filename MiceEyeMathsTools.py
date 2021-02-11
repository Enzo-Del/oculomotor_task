#-*- coding: utf-8 -*-
"""
Created on Sun Mar  8 11:48:02 2020
Last modified on
@author: Enzo Delamarre
Functions for eye measurements calculation
"""

import numpy as np
import math


def dfToArray(array, dataFrame, size) :
# Convert pandas dataFrame to python array. Usefull when you manipulate deeplabcut csv files
    for j in range(2, size):
        array[0, j] = float(dataFrame.iloc[j])
        array[np.isnan(array)] = 0


def velocity(i, V, pupil_center):
# Gets the velocity of the pupil center for one frame

    if i == 1:
        V[0, i] = 0
    elif i ==2:
        V[0, i] = 0
    else :
        V[0, i] = (((pupil_center[i, 0] - pupil_center[i - 2, 0]) ** (2) + (
                    pupil_center[i, 1] - pupil_center[i - 2, 1]) ** (2)) ** (1 / 2))



def pupil_center_radius(i, arrayC, PredictedData) :
# Computes the position (x,y) of the center of the pupil and its radius for one frame

    # Pupil center coordinates computation, based on an octogon representation of the eye
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

    arrayC[0, i] = PredictedData[i, 18]
    arrayC[1, i] = PredictedData[i, 19] - r

    return  r


def center_Pupil_avg(C, size,R0):
# Gets the position (x,y) of the pupil center for a batch of frames

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


def scale_factor(i, PredictedData):
    # Scale factor = horizontal 2D lenght of the eyball, equals 3cm in c57/BL6 mice
    scale = (((PredictedData[i, 24] - PredictedData[i, 30]) ** (2) )+ ((PredictedData[i, 24] - PredictedData[i, 31]) ** (2))) ** (1 / 2)
    scale_factor = scale/3
    return scale_factor


def angular_position(i,scale_factor, radius, C, Eh, Ev, predicted_data):
    # Computes the absolute angle of the eye for one frame, calulation based on Sakatani and Isa, 2007 model
    #TODO : Ajouter conition if vérifiant la probabilité de présence du point
    #Cornea center
    X0 = abs(predicted_data[i, 38]- predicted_data[i, 44])
    Y0 = abs(predicted_data[i, 42]- predicted_data[i, 48])
    # R effective
    Rlens = 1.25 * scale_factor # Rlens = 1.25 cm in c57/BL6 mices
    R = math.sqrt((Rlens*Rlens)- (radius*radius)) - (0.1 * scale_factor) # Epsilon = 0.1 cm in c57/BL6 mices

    # Azimutal (Eh) and elevation (Ev) angle computation
    Eh[0, i] = math.degrees(np.arcsin((C[0, i] - X0) / R))
    Ev[0, i] = math.degrees(np.arcsin(-(C[1, i] - Y0) / R))
    Eh[np.isnan(Eh)] = 0
    Ev[np.isnan(Ev)] = 0


def global_variation_rate_blink(i, PredictedData):
# Check if the eye is not moving too strongly and if there is no eye blink
    blink = 0
    if i == 1 :
        Tx = 0
    else :

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

        eye_height_var = (((PredictedData[i, 28] - PredictedData[i - 1, 28]) / (PredictedData[i - 1, 28])) * 100
        + ((PredictedData[i, 34] - PredictedData[i - 1, 34]) / (PredictedData[i - 1, 34])) * 100) / 2

        if (Tx >= 5.53 or Tx <= -5.53) and (Ty>=3 or Ty<=-3):
          blink =+1
        if eye_height_var > 10 or eye_height_var < -10:
          blink =+1

    return blink

def circle_pos(screen_dist, circle_pos_angle):
    circle_pos = (math.tan(math.degrees(circle_pos_angle)))*screen_dist
    return abs(circle_pos)



