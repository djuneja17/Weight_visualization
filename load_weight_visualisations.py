#
# Cloud for ML Final Project
# Divya Juneja
# load_weight_visualisations.py
#

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
from torchvision import transforms, utils

import numpy as np
import scipy.misc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import json

import base64
import os
import uuid

from model.alexnet import predict as predict_alexnet

def to_grayscale(image):
    """
    input is (d,w,h)
    converts 3D image tensor to grayscale images corresponding to each channel
    """
    image = torch.sum(image, dim=0)
    image = torch.div(image, image.shape[0])
    return image

def normalize(image):
    normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    normalize
    ])
    # image = Variable(preprocess(image).unsqueeze(0).cuda())
    image = Variable(preprocess(image).unsqueeze(0))
    return image


def predict(image):
    _, index = alexnet(image).data[0].max(0)
    return str(index.numpy()), labels[str(index.numpy())][1]

def load_image(path):
    image = Image.open(path)
    plt.imshow(image)
    plt.title("Image loaded successfully")
    return image

def filter_outputs(image, layer_to_visualize, modulelist, output_path):
    if layer_to_visualize < 0:
        layer_to_visualize += 31
    output = None
    name = None
    for count, layer in enumerate(modulelist[1:]):
        image = layer(image)
        if count == layer_to_visualize: 
            output = image
            name = str(layer)
    
    filters = []
    output = output.data.squeeze()
    for i in range(output.shape[0]):
        filters.append(output[i,:,:])
        
    fig = plt.figure()
    plt.rcParams["figure.figsize"] = (10, 10)

    #for i in range(int(np.sqrt(len(filters))) * int(np.sqrt(len(filters)))):
    for i in range(int(np.sqrt(25)) * int(np.sqrt(25))):
       # fig.add_subplot(np.sqrt(len(filters)), np.sqrt(len(filters)),i+1)
        fig.add_subplot(np.sqrt(25), np.sqrt(25),i+1)
        imgplot = plt.imshow(filters[i])
        plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight')

def get_weight_visualisations(image, layer_to_visualize):
    """
    Returns in JSON format the class, probability of class, the
    weight visualisations for the given layer

    @param img:
    @return:
    """
    # Write down to tmp file
    filename = str(uuid.uuid4()) + ".jpg"
    filepath = os.path.join("tmp", filename)
    image.save(filepath)

    alexnet = models.alexnet(pretrained=True)

    print(alexnet)

    # Infer class
    (cls, conf, idx) = sorted(predict_alexnet(image, alexnet), reverse=True)[0]

    # Write to output file
    layer_weight_output = os.path.join("output", 'layer_' + layer_to_visualize + "_" + filename)

    image = normalize(image)

    modulelist = list(alexnet.features.modules())
    print(modulelist)

    filter_outputs(image, int(layer_to_visualize), modulelist, layer_weight_output)

    # Output file to base64
    with open(layer_weight_output, "rb") as fp:
        encoded_img = base64.b64encode(fp.read())

    # Remove tmp and output files
    os.remove(filepath)
    os.remove(layer_weight_output)

    response = {'class': str(cls),
                'probability': float(conf),
                'layer_weight_output': encoded_img.decode('utf-8')
                }

    return response





