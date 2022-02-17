import cv2
import numpy as np
import os
from tqdm import tqdm

"""
Script to convert frames into a video for submission
"""

# * Logfile to be processed
logfile = 2
base_path = r"D:\CMU\Academics\SLAM\Homeworks\HW1\hw1_code_data_assets\results"
log_path = base_path + "-robotdata" + str(logfile)
print("Log path : ", log_path)

# * Read frames for corresponding logfile
img_array = []
for filename in tqdm(os.listdir(log_path), desc="Reading Frames"):
    img = cv2.imread(os.path.join(log_path, filename))
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

# * Create directory for storing videos
os.makedirs(base_path, exist_ok=True)
print("Created directory : ", base_path)

# * Save frames into video
video_path = os.path.join(base_path, "robotlog" + str(logfile) + ".avi")
print("Video Path : ", video_path)
out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"DIVX"), 60, size)


for i in tqdm(range(len(img_array)), desc="Writing Video"):
    out.write(img_array[i])
out.release()
