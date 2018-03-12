# # The Implementation of our Project
# # with datasets from CIFAR-10
# # using TensorFlow(ML library from Google)
# 
# by [Hammani Elasri Elkamali]

import tensorflow as tf
import numpy as np
import math
import os
from tkinter import *
from PIL import Image
from PIL import ImageTk
import cv2
import prettytensor as pt
import cifar10

class_names = cifar10.load_class_names()

from cifar10 import img_size, num_channels, num_classes

img_size_cropped = 24

x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')

y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')

y_true_cls = tf.argmax(y_true, dimension=1)

def pre_process_image(image, training):
    # This function takes a single image as input,
    # and a boolean whether to build the training or testing graph.
    
    if training:
        # For training, add the following to the TensorFlow graph.

        # Randomly crop the input image.
        image = tf.random_crop(image, size=[img_size_cropped, img_size_cropped, num_channels])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)
        
        # Randomly adjust hue, contrast and saturation.
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

        # Some of these functions may overflow and result in pixel
        # values beyond the [0, 1] range. It is unclear from the
        # documentation of TensorFlow 0.10.0rc0 whether this is
        # intended. A simple solution is to limit the range.

        # Limit the image pixels between [0, 1] in case of overflow.
        image = tf.minimum(image, 1.0)
        image = tf.maximum(image, 0.0)
    else:
        # For training, add the following to the TensorFlow graph.

        # Crop the input image around the centre so it is the same
        # size as images that are randomly cropped during training.
        image = tf.image.resize_image_with_crop_or_pad(image,target_height=img_size_cropped,target_width=img_size_cropped)

    return image


def pre_process(images, training):
    # Use TensorFlow to loop over all the input images and call
    # the function above which takes a single image as input.
    images = tf.map_fn(lambda image: pre_process_image(image, training), images)

    return images


def main_network(images, training):
    # Wrap the input images as a Pretty Tensor object.
    x_pretty = pt.wrap(images)

    if training:
        phase = pt.Phase.train
    else:
        phase = pt.Phase.infer

    # Create the convolutional neural network using Pretty Tensor.
    with pt.defaults_scope(activation_fn=tf.nn.relu, phase=phase):
        y_pred, loss = x_pretty.\
        conv2d(kernel=5, depth=96, name='layer_conv1', batch_normalize=True).\
        max_pool(kernel=3, stride=2).\
        conv2d(kernel=3, depth=192, name='layer_conv2', batch_normalize=True).\
        max_pool(kernel=3, stride=2).\
        conv2d(kernel=3, depth=192, name='layer_conv3', batch_normalize=True).\
        max_pool(kernel=3, stride=2).\
        conv2d(kernel=3, depth=192, name='layer_conv4', batch_normalize=True).\
        max_pool(kernel=3, stride=2).\
        flatten().\
        fully_connected(size=512, name='layer_fc1').\
        softmax_classifier(num_classes=num_classes, labels=y_true)
    return y_pred, loss
#(N-F)/S+1 output size!


def create_network(training):
    # Wrap the neural network in the scope named 'network'.
    # Create new variables during training, and re-use during testing.
    with tf.variable_scope('network', reuse=not training):
        # Just rename the input placeholder variable for convenience.
        images = x

        # Create TensorFlow graph for pre-processing.
        images = pre_process(images=images, training=training)

        # Create TensorFlow graph for the main processing.
        y_pred, loss = main_network(images=images, training=training)

    return y_pred, loss


global_step = tf.Variable(initial_value=0,name='global_step', trainable=False)

_, loss = create_network(training=True)


y_pred, _ = create_network(training=False)


y_pred_cls = tf.argmax(y_pred, dimension=1)


saver = tf.train.Saver()


session = tf.Session()


save_dir = './checkpoints(copy)/'


if not os.path.exists(save_dir):
    os.makedirs(save_dir)


save_path = os.path.join(save_dir, 'cifar10_cnn')


try:
    print("Trying to restore last checkpoint ...")

    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)

    saver.restore(session, save_path=last_chk_path)

    print("Restored checkpoint from:", last_chk_path)
except:
    print("Failed to restore checkpoint. Initializing variables instead.")
    session.run(tf.global_variables_initializer())


def predict_cls_test():
    return predict_cls(images = images_test,labels = labels_test,cls_true = cls_test)

def normalize(array):
    newArray = array / 255.0
    return newArray

def predict_one_image(image):

#    print("\nOn Prediction")
    prediction ,indice = session.run([y_pred, y_pred_cls],feed_dict={x: [image]})

    return prediction[0],indice[0]
    

def Show_prediction(Img):
    global class_name_label
    try:
        _vect ,index = predict_one_image(Img)
       
        #np.set_printoptions(precision=3, suppress=True)
        
        arr = np.copy(_vect)
        
        first = arr.max()
        firstidx = np.argmax(arr)
        arr[firstidx] = -1
        
        second = arr.max()
        secondidx = np.argmax(arr)
        arr[secondidx] = -1
        
        third = arr.max()
        thirdidx = np.argmax(arr)
        
        var = str(round(first*100,2))+"% "+class_names[firstidx]+"\n"+ str(round(second*100,2))+"% "+ class_names[secondidx]+"\n"+ str(round(third*100,2))+"% "+ class_names[thirdidx]
       
        
        class_name_label.configure(text=var,fg="black",font=("Tahoma", 30))

    except ValueError as e:
        print("Problem!! fix this ", e)


def import_resize_predict_image(myImage):
    size32 = (32,32)
    original = myImage.copy()
    w,h = myImage.size
    if w==h:
        lastImg = myImage.copy()
        lastImg.thumbnail(size32)
    elif w>h:
        lastImg = myImage.crop((0, 0, w,w))
        lastImg.thumbnail((size32))
    else:
        lastImg = myImage.crop((0, 0, h,h))
        lastImg.thumbnail((size32))
        
    Img = np.asarray(lastImg,dtype=np.float64)
    
    Img=normalize(Img)
    
    Show_prediction(Img)


import tkinter as tk

#Commentes By ELASRI 

def showw_frame():
    _, frame = cap.read() 
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    img = img.convert("RGB")
    
    imgtk = ImageTk.PhotoImage(image=img) 
    lmain.imgtk = imgtk #lmain image is set to the image extracted from cv2image
    lmain.configure(image=imgtk)

    img.save("./test.jpeg")
    test = Image.open("./test.jpeg")
    import_resize_predict_image(test)

    lmain.after(1,showw_frame)  # after 1 miliSecond 


#Set up GUI
color = "grey"
window = tk.Tk()  #Makes main window
window.wm_title("CIFAR WEB-CAM") #Title of the window !
window.config(background=color)
window.resizable(0, 0)


#webcam Graphics window
imageFrame = tk.Frame(window, width=600, height=600) #ImageFrameShower 
imageFrame.grid(row=0, column=0, padx=10, pady=10) #add padding to the imageframe

#Capture video frames
lmain = tk.Label(imageFrame) #Add Label to imageFrame
lmain.grid(row=0, column=0)  #Label Grids


#Showing LABEL
class_name_label = tk.Label(window,text="!!!!!!!!!!!!!!!!!",font=('Tahoma',20),background=color) 
class_name_label.grid(row=1,column = 0)

cap = cv2.VideoCapture(0) #Open web Cam by default

showw_frame()

window.mainloop()    

#excution time 1 second 

session.close()

