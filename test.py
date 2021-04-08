import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
import deeplabcut
#config_path3d = deeplabcut.create_new_project_3d('P4_final','Enzo', num_cameras = 2, working_directory = r'C:\Users\opto-delamarree\Desktop')
config_path3d = r'C:\Users\opto-delamarree\Desktop\P4_final-Enzo-2021-04-03-3d\config.yaml'
deeplabcut.calibrate_cameras(config_path3d, cbrow =9, cbcol = 7, calibrate = False, alpha = 0.9)
#deeplabcut.calibrate_cameras(config_path3d, cbrow = 9, cbcol = 7, calibrate = True, alpha = 0.9)
#deeplabcut.check_undistortion(config_path3d, cbrow = 9, cbcol = 7)
#deeplabcut.triangulate(config_path3d, r'C:\Users\opto-delamarree\Desktop\what3', videotype = '.mp4', filterpredictions = True, save_as_csv = True)

import matplotlib as plt
from mpl_toolkits.mplot3d import Axes3D
#deeplabcut.create_labeled_video_3d(config_path3d,[r'C:\Users\opto-delamarree\Desktop\what3'],start = 400, end = 430, videotype = '.mp4',view = [-150,45])