"""
This file is used for creating a movie from png files.
"""

import imageio
import os

img_dir = 'segmented_images/' # directory storing images
video_name = 'segmentation.mp4' # file name for the video to be created

# create a list storing complete paths for the png images
list_img_path = []
for img_name in os.listdir(img_dir):
    img_path = img_dir + img_name
    # print(img_path)
    list_img_path.append(img_path)

# create a video from the images
writer = imageio.get_writer(video_name)
for img_path in list_img_path:
    writer.append_data(imageio.imread(img_path))
writer.close()
