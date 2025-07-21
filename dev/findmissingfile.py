# from email.mime import audio
# import os

# video_path ="/home/mt0/22CS60R39/BM-NAS_dataset/NTU/nturgb+d_rgb_256x256_30/"

# audio_path = "/home/mt0/22CS60R39/BM-NAS_dataset/NTU/nturgb+d_rgb_256x256_30_audio/"

# video_lis = os.listdir(video_path)
# audio_lis = os.listdir(audio_path)
# print(len(video_lis))
# print(len(audio_lis))

# x = []
# for i in video_lis:
#     if ".mp4" in i: 
#         x.append(i.split('.')[0])
# y = []
# for i in audio_lis:
#     if ".npy" in i: 
#         y.append(i.split('.')[0])
# print(len(x))
# print(len(y))
# print("Diff list:",[item for item in x if item not in y])

import numpy as np
x = np.load("mean.npy")
std = np.load("std.npy")
print(x.shape)
print(std.shape)