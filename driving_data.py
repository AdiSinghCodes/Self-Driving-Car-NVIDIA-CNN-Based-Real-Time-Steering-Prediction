# Loading and preprocessing data for the Nvidia model. The images are cropped to the bottom 150 pixels, resized to 66x200, and normalized to be between 0 and 1. The steering wheel angles are converted from degrees to radians.

import numpy as np
from PIL import Image


xs = []
ys = []

#points to the end of the last batch
train_batch_pointer = 0
val_batch_pointer = 0

# read data.txt - converting the steerling angle to radians and storing the image paths and steering angles in xs and ys respectively
with open("driving_dataset/data.txt") as f:
    for line in f:
        xs.append("driving_dataset/" + line.split()[0])
        #the paper by Nvidia uses the inverse of the turning radius,
        #but steering wheel angle is proportional to the inverse of turning radius
        #so the steering wheel angle in radians is used as the output
        ys.append(float(line.split()[1]) * np.pi / 180)

#get number of images
num_images = len(xs)

# Splitting into training and validation sets - 80% for training and 20% for validation
train_xs = xs[:int(len(xs) * 0.8)]
train_ys = ys[:int(len(xs) * 0.8)]

val_xs = xs[-int(len(xs) * 0.2):]
val_ys = ys[-int(len(xs) * 0.2):]

num_train_images = len(train_xs)
num_val_images = len(val_xs)


"""LoadTrainBatch & LoadValBatch: Load a batch of images from disk, preprocess them (crop bottom 150px, resize to 66×200, normalize to 0-1 range), and return them along with their corresponding steering angles for training/validation.

Even shorter version:
"Load, preprocess, and return a batch of road images with their steering angles."""
def LoadTrainBatch(batch_size):
    global train_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        img = Image.open(train_xs[(train_batch_pointer + i) % num_train_images])
        img_array = np.array(img)
        img_cropped = img_array[-150:]
        img_resized = Image.fromarray(img_cropped).resize((200, 66))
        x_out.append(np.array(img_resized) / 255.0)
        y_out.append([train_ys[(train_batch_pointer + i) % num_train_images]])
    train_batch_pointer += batch_size
    return x_out, y_out

def LoadValBatch(batch_size):
    global val_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        img = Image.open(val_xs[(val_batch_pointer + i) % num_val_images])
        img_array = np.array(img)
        img_cropped = img_array[-150:]
        img_resized = Image.fromarray(img_cropped).resize((200, 66))
        x_out.append(np.array(img_resized) / 255.0)
        y_out.append([val_ys[(val_batch_pointer + i) % num_val_images]])
    val_batch_pointer += batch_size
    return x_out, y_out
