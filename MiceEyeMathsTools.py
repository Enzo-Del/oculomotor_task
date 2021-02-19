# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 11:48:02 2020
Last modified on
@author: Enzo Delamarre
Functions for eye measurements calculation
"""

import numpy as np
import math


def dfToArray(array, data_frame, size):
    # Convert pandas dataFrame to python array. Usefull when you manipulate deeplabcut csv files
    for j in range(2, size):
        array[0, j] = float(data_frame.iloc[j])
        array[np.isnan(array)] = 0


def velocity(i, V, pupil_center, time_stp):
    # Gets the velocity of the pupil center for one frame

    if i == 0:
        V.append(0)
    elif i == 1:
        V.append(0)
    elif i == 2:
        V.append(0)
    else:
        V.append((((pupil_center[0][i] - pupil_center[0][i-2]) ** (2) + (
                pupil_center[1][i] - pupil_center[1][i-2]) ** (2)) ** (1 / 2)) / (2 * time_stp))


def pupil_center_radius(i, array_c, predicted_data):
    # Computes the position (x,y) of the center of the pupil and its radius for one frame
    dx = 0
    dy = 0
    j = 0
    # Pupil radius estimation, based on an octogon representation of the pupil
    a = ((((predicted_data[i, 0] - predicted_data[i, 3]) ** (2) + (predicted_data[i, 1] - predicted_data[i, 4]) ** (2)) ** (
                1 / 2))
         + (((predicted_data[i, 3] - predicted_data[i, 6]) ** (2) + (predicted_data[i, 4] - predicted_data[i, 7]) ** (
                2)) ** (1 / 2))
         + (((predicted_data[i, 6] - predicted_data[i, 9]) ** (2) + (predicted_data[i, 7] - predicted_data[i, 10]) ** (
                2)) ** (1 / 2))
         + (((predicted_data[i, 9] - predicted_data[i, 12]) ** (2) + (predicted_data[i, 10] - predicted_data[i, 13]) ** (
                2)) ** (1 / 2))
         + (((predicted_data[i, 12] - predicted_data[i, 15]) ** (2) + (predicted_data[i, 13] - predicted_data[i, 16]) ** (
                2)) ** (1 / 2))
         + (((predicted_data[i, 15] - predicted_data[i, 18]) ** (2) + (predicted_data[i, 16] - predicted_data[i, 19]) ** (
                2)) ** (1 / 2))
         + (((predicted_data[i, 18] - predicted_data[i, 21]) ** (2) + (predicted_data[i, 19] - predicted_data[i, 22]) ** (
                2)) ** (1 / 2))
         + (((predicted_data[i, 21] - predicted_data[i, 0]) ** (2) + (predicted_data[i, 22] - predicted_data[i, 1]) ** (
                2)) ** (1 / 2)))

    A = a / 8
    r = 1.3066 * A
    # Pupil center estimation

    for k in range(4):
        dx = dx + (predicted_data[i, j] + predicted_data[i, j + 12]) / 2
        dy = dy + (predicted_data[i, j + 1] + predicted_data[i, j + 13]) / 2
        j += 3

    pupil_center_x = dx / 4
    pupil_center_y = dy / 4
    array_c[0].append(pupil_center_x)
    array_c[1].append(pupil_center_y)

    return r


def center_Pupil_avg(array_c, size, array_c_avg):
    # Gets the position (x,y) of the pupil center for a batch of frames

    x = 0
    y = 0
    for j in range(1, size):
        x = x + array_c[0, j]
        y = y + array_c[1, j]
        if math.isnan(x) == True:
            x = 0
        if math.isnan(y) == True:
            y = 0
    array_c_avg.append(x / size)
    array_c_avg.append(y / size)


def scale_factor(i, predicted_data):
    # Scale factor = horizontal 2D lenght of the eyball, equals 3cm in c57/BL6 mice
    scale = (((predicted_data[i, 24] - predicted_data[i, 30]) ** (2)) + (
            (predicted_data[i, 25] - predicted_data[i, 31]) ** (2))) ** (1 / 2)

    scale_factor = scale / 3
    return scale_factor


def angular_position(i, scale_factor, radius, pupil_center, azimut, elev, predicted_data):
    # Computes the absolute angle of the eye for one frame, calulation based on Sakatani and Isa, 2007 model
    # Cornea center
    j = 0
    dx = 0
    dy = 0
    for k in range(2):
        dx = dx + (predicted_data[i, j + 36] + predicted_data[i, j + 42])/2
        dy = dy + (predicted_data[i, j + 37] + predicted_data[i, j + 43])/2
        j +=3

    cornea_center_x = dx/2
    cornea_center_y = dy/2
    # R effective
    r_lens = 1.25 * (scale_factor)  # Rlens = 1.25 cm in c57/BL6 mices
    r_factor = math.sqrt((r_lens**2) - (radius[i]**2)) - (0.1 * scale_factor)  # Epsilon = 0.1 cm in c57/BL6 mices

    # Azimutal (Eh) and elevation (Ev) angle computation
    azimut.append(round(math.degrees(np.arcsin((pupil_center[0][i] - cornea_center_x) / r_factor)),3))
    elev.append(round(math.degrees(np.arcsin(-(pupil_center[1][i] - cornea_center_y) / r_factor)),3))
    #azimut[np.isnan(int(azimut))] = 0
    #elev[np.isnan(elev)] = 0


def variation_rate(i, var_rate, pupil_center):
    if i == 0:
        var_rate.append(0)
    else:
        var_rate.append(abs((((pupil_center[0][i] - pupil_center[0][i-1]) /pupil_center[0][i-1]) + ((
            pupil_center[1][i] - pupil_center[1][i-1]) / pupil_center[1][i-1])) / 2))


def global_variation_rate_blink(i, predicted_data):
    # Check if the eye is not moving too strongly and if there is no eye blink
    # Function from the v1 version, might be obsolete
    blink = 0
    if i> 0:
        eye_height_var = (abs(predicted_data[i-1, 28] - predicted_data[i-1, 34])) / (abs(predicted_data[i, 28] -
                                                                                       predicted_data[i, 34]))

    else : eye_height_var =0
    if eye_height_var > 2 :
        blink = +1

    return blink


def circle_pos(screen_dist, circle_pos_angle, stim_radius):
    circle_pos = ((math.tan(math.radians(circle_pos_angle))) * screen_dist)*(1024/5)
    stim_radius_pos = ((math.tan(math.radians(stim_radius))) * screen_dist)*(1024/5)
    return abs(int(circle_pos)), abs(int(stim_radius_pos))
