import cv2
import numpy as np
import os
from tqdm import tqdm

img_array = []

logfile = 4
base_path = r"D:\CMU\Academics\SLAM\Homeworks\HW1\hw1_code_data_assets\results"
log_path = base_path + "-robotdata" + str(logfile)
print(log_path)

for filename in tqdm(os.listdir(log_path), desc="Reading Frames"):
    img = cv2.imread(os.path.join(log_path, filename))
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

os.makedirs(base_path, exist_ok=True)
print("Created directory : ", base_path)
video_path = os.path.join(base_path, "robotlog" + str(logfile) + ".avi")
print("Video Path : ", video_path)
out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"DIVX"), 30, size)

for i in tqdm(range(len(img_array)), desc="Writing Video"):
    out.write(img_array[i])
out.release()
