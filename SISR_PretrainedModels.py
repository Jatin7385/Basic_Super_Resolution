# Importing packages
import time
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np


# Loading the data
print("Loading the frames")
# imgg = np.load("../Launch_Videos/Launch_Frames.npy")

print("Frames Loaded")
frame_no = 1815

# np.save("1815.npy",imgg[1815, 180 : 692, 700 : 1212])
test_img = np.load("1815.npy")
# test_img = np.load("400.npy")
# test_img = np.load("740.npy")
# test_img = np.load("750.npy")
# test_img = np.load("789.npy")
# test_img = np.load("890.npy")
# test_img = np.load("1145.npy")
# test_img = np.load("2213.npy")



# test_img = imgg[frame_no, 180 : 692, 700 : 1212]


super_resolution = cv2.dnn_superres.DnnSuperResImpl_create()

# Define the path to the pretrained models
# path = "./Models/EDSR_x4.pb"
# path = "./Models/ESPCN_x4.pb"
# path = "./Models/FSRCNN_x4.pb"
path = "./Models/LapSRN_x4.pb"

super_resolution.readModel(path)
# super_resolution.setModel("edsr",4)
# super_resolution.setModel("espcn",4)
# super_resolution.setModel("fsrcnn",4)
super_resolution.setModel("lapsrn",4)

result = super_resolution.upsample(test_img)

# Resized image
resized = cv2.resize(test_img, dsize = None, fx=4,fy=4)

# Displaying
fig, ax = plt.subplots(1,2, figsize=(12,8), sharex = True, sharey = True)
# ax[0].imshow(test_img[:,:,::-1], cmap="gray")
ax[0].imshow(result[870:1188,720:1080,::-1], cmap="gray")
ax[1].imshow(resized[870:1188,720:1080,::-1], cmap="gray")

# ax[1][0].imshow(img1[:,:,::-1], cmap="gray")
# ax[1][1].imshow(img1[:,:,::-1], cmap="gray")
# ax[1][2].imshow(img1[:,:,::-1], cmap="gray")

# ax[2][0].imshow(img1[:,:,::-1], cmap="gray")
# ax[2][1].imshow(img1[:,:,::-1], cmap="gray")
# ax[2][2].imshow(img1[:,:,::-1], cmap="gray")

# plt.figure(figsize=(12,8))
# plt.subplot(1,3,1)
# # Original image
# plt.imshow(test_img[:,:,::-1])
# plt.subplot(1,3,2)
# # SR upscaled
# plt.imshow(result[:,:,::-1])
# plt.subplot(1,3,3)
# # OpenCV upscaled
# plt.imshow(resized[:,:,::-1])
plt.show()