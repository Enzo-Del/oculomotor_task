import deeplabcut
config_path3d = r'C:\Users\opto-delamarree\Desktop\p4_testFev_3D-Enzo-2021-02-25-3d/config.yaml'
#deeplabcut.calibrate_cameras(config_path3d, cbrow =6, cbcol = 8, calibrate = False, alpha = 0.9)
#deeplabcut.calibrate_cameras(config_path3d, cbrow = 6, cbcol = 8, calibrate = True, alpha = 0.9)
deeplabcut.check_undistortion(config_path3d, cbrow = 6, cbcol = 8)
#deeplabcut.triangulate(config_path3d, r'C:\Users\opto-delamarree\Desktop\p4_videos_3D_test4', videotype = '.mp4', filterpredictions = True, save_as_csv = True)

import matplotlib as plt
from mpl_toolkits.mplot3d import Axes3D
#deeplabcut.create_labeled_video_3d(config_path3d,[r'C:\Users\opto-delamarree\Desktop\p4_videos_3D_test4'],start = 0, end = 40, trailpoints= 0, videotype = '.mp4',view = [-143,225])