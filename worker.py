import sys, os, io
from os.path import expanduser
home = expanduser("~")
#sys.path.insert(0,os.path.join( home, "codes/keras-multiprocess-image-data-generator"))
sys.path.insert(0,os.path.join( home, "codes/keras-preprocessing"))
sys.path.insert(0,os.path.join( home, "codes/keras-applications"))
sys.path.insert(0,os.path.join( home, "codes/DLWorkspace-Utils"))

import argparse
import os
import random
import shutil
import time
import warnings

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import os.path

from keras.applications.densenet import DenseNet121, DenseNet201, DenseNet169
from keras.applications.resnet50 import ResNet50

import tensorflow as tf
from keras.applications import Xception
from keras.utils import multi_gpu_model
from keras.regularizers import * 
from keras.callbacks import LearningRateScheduler, Callback
from keras.optimizers import Adam
from keras.optimizers import SGD

import dlwstools.utils as U

import keras_preprocessing.image as T
from keras_preprocessing import datasets
import multiprocessing.dummy

import sys, inspect
import math
import datetime
import re
import time
from sklearn.metrics import *
import itertools
import matplotlib.pyplot as plt

import numpy as np
import json



from keras_applications import inception_v4  
from keras_applications.densenet import DenseNet  


from keras.models import load_model
import keras.models as KM

from keras_preprocessing.image import img_to_array


import itertools
from sklearn.metrics import *

import gc

import sys
import json
import os

import base64
import yaml

import logging
from logging.config import dictConfig

import queue
import base64
import uuid
import multiprocessing
from threading import Thread, Lock

from storage import Azure_Storage

try:
    from PIL import ImageEnhance
    from PIL import Image as pil_image
except ImportError:
    pil_image = None
    ImageEnhance = None

from io import BytesIO

#os.environ["CUDA_VISIBLE_DEVICES"]=""

if pil_image is not None:
    _PIL_INTERPOLATION_METHODS = {
        'nearest': pil_image.NEAREST,
        'bilinear': pil_image.BILINEAR,
        'bicubic': pil_image.BICUBIC,
    }
    # These methods were only introduced in version 3.4.0 (2016).
    if hasattr(pil_image, 'HAMMING'):
        _PIL_INTERPOLATION_METHODS['hamming'] = pil_image.HAMMING
    if hasattr(pil_image, 'BOX'):
        _PIL_INTERPOLATION_METHODS['box'] = pil_image.BOX
    # This method is new in version 1.1.3 (2013).
    if hasattr(pil_image, 'LANCZOS'):
        _PIL_INTERPOLATION_METHODS['lanczos'] = pil_image.LANCZOS    


models = []
batch_size = 16
azure_storage = Azure_Storage()
image_size = (600,450)
dia_str = ["Melanoma",
"Melanocytic nevus",
"Basal cell carcinoma",
"Actinic keratosis / Bowen’s disease (intraepithelial carcinoma)", 
"Benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis)",
"Dermatofibroma",
"Vascular lesion"]


def inference(tasks):
    batch = []
    for t in tasks:
        img = t["image_data"]
        batch.append(img)
    batch = np.array(batch)
    results = []
    for model in models:
        results.append(model.predict(batch))
    results = np.array(results)
    scores = np.average(results,axis=0)
    pred = np.argmax(scores,axis=1)
    return pred, scores


def load_img(path, grayscale=False, color_mode='rgb', target_size=None,
             interpolation='nearest'):
    """Loads an image into PIL format.
    # Arguments
        path: Path to image file.
        color_mode: One of "grayscale", "rbg", "rgba". Default: "rgb".
            The desired image format.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
    """
    if grayscale is True:
        warnings.warn('grayscale is deprecated. Please use '
                      'color_mode = "grayscale"')
        color_mode = 'grayscale'
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    img = pil_image.open(path)
    if color_mode == 'grayscale':
        if img.mode != 'L':
            img = img.convert('L')
    elif color_mode == 'rgba':
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
    elif color_mode == 'rgb':
        if img.mode != 'RGB':
            img = img.convert('RGB')
    else:
        raise ValueError('color_mode must be "grayscale", "rbg", or "rgba"')



    width, height = img.size
    if width < height:
        img = img.rotate( 90, expand=1 )

    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            if interpolation not in _PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    'Invalid interpolation method {} specified. Supported '
                    'methods are {}'.format(
                        interpolation,
                        ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
            resample = _PIL_INTERPOLATION_METHODS[interpolation]
            img = img.resize(width_height_tuple, resample)
    return img



def run():
    while True:
        try:
            tasks = []
            taken_msgs = []

            messages = azure_storage.get_task(batch_size)
            for message in messages:
                try:
                    meta = json.loads(message.content)
                    print(meta)

                    task_id = meta["task_id"]

                    image_data = azure_storage.get_image(task_id)

                    image_data_bytes = BytesIO(image_data)
                    image_data = load_img(image_data_bytes, target_size = image_size)
                    image_data = img_to_array(image_data)
                    image_data = image_data / 255.0
                    print(image_data.shape)
                    task = {"task_id":task_id, "image_data":image_data}
                    tasks.append(task)
                    taken_msgs.append(message)
                except Exception as e:
                    print(str(e))
                    pass

                ## delete the message by default
                ## TODO: should handle the failure cases more carefully. 
                try:
                    azure_storage.delete_task(message)
                except Exception as e:
                    pass


            if len(tasks) > 0:
                print("inference: %d images" % len(tasks))
                time1 = time.perf_counter()
                pred, scores = inference(tasks)
                time2 = time.perf_counter()
                print("inferenced %d images in %.2f seconds" % (len(tasks),(time2-time1)))
                for i in range(len(pred)):

                    result = {}
                    result["diagnosis"] = dia_str[pred[i]]
                    result["diagnosis_index"] = str(pred[i])
                    result["scores"] = scores[i].tolist()
                    result["diagnosis_report"] = "The diagnosis is '%s' (with %.2f%% confidence)" % (dia_str[pred[i]],scores[i][pred[i]]*100)
                    print(result)
                    azure_storage.put_classification_result(tasks[i]["task_id"],json.dumps(result))
                    #azure_storage.delete_task(taken_msgs[i])
        
        except Exception as e:
            print(str(e))

def init_models():
    print("Loading models")
    model_pattern = "/home/hongzl/densenet201-600/isic_final_split%d.h5"
    #model_pattern = "/home/hongzl/train/keras/isic_dense201_batch128_split%d/final_weights_split%d.h5"
    model_files=[]

    for i in range(6):
        model_files.append(model_pattern % (i))
    for i,model_file in enumerate(model_files):
        print("loading models from %s" % model_file)
        #with tf.device('/gpu:%d' %i):
        with tf.device('/cpu:0'):
            model = load_model(model_file)
            #model = DenseNet([6, 12, 48, 32], True, None, None, None, None, 7)
            #model = inception_v4.create_model(num_classes=7,include_top=True)
        #model.load_weights(model_file,by_name=True)        

        models.append(model)
    if len(models) >0 :
        models[0].summary()        
    print("%d models are loaded" % len(models))


if __name__ == '__main__':

     
    init_models()
    run()


