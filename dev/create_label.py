import os
import random

root_dir = "/media/user1/8089c1c4-5bde-4b0d-abd1-ac7131e6ceb4/Aravind_data/codes/vinod/BM-NAS_dataset/NTU/nturgb+d_rgb_256x256_30"
label_path = "/media/user1/8089c1c4-5bde-4b0d-abd1-ac7131e6ceb4/Aravind_data/codes/vinod/BM-NAS_dataset/NTU/label2.txt"

files = os.listdir(root_dir)

fptr = open(label_path,"w")

for file in files:
    #file_path = os.path.join(root_dir, file)
    s = file + " "+ str(random.randint(0,1))+"\n"
    fptr.write(s)
fptr.close()
