#!/usr/bin/env python
import io
import logging
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import sys
import json
from datetime import datetime
import boto3

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from PIL import ImageOps
from torchvision import transforms
from yacs.config import CfgNode as CN

# sys.path.append('./ssm')
from ssm.model_files.data_parallel import async_copy_to
from ssm.model_files.models import ModelBuilder
from ssm.model_files.models import SegmentationModule
from ssm.model_files.th import as_numpy

# from IQA import iqa 
from IQA.mp_detector import MediaPipeFacialDetector

NN_KEY = "L2_AGLSS_max"
NN_REST_KEY = "L2_AGLSS_rest"
DECODER_KEY = "decoder_"
ENCODER_KEY = "encoder_"
IMAGE_RESIZE_INTERP = "bilinear"

__gpu = 0 # TODO: pull value from env vars
__gpu1 = 1

# __image_file_path = None
__log_level = logging.getLevelName(os.getenv("LOG_LEVEL", "INFO").upper())
__logger = logging.getLogger("AGLSS SSM_Facial_Zone_Classifier_Frontal_AND_Severity .inference")
__ssm_cfg = None


__logger.setLevel(__log_level)
__logger.addHandler(logging.StreamHandler(sys.stdout))
__logger.info(f"logging initialized with level: {__log_level}")

try:
    print("cuda version:", torch.version.cuda)
    print("pytorch version:", torch.__version__)
    print(torch.backends.cudnn.version())
except Exception as e:
    print(e)

# ----------------------------------------------------------------------------------------------------
# START: SAGEMAKER ENDPOINT FUNCTIONS
# ----------------------------------------------------------------------------------------------------


# defining model and loading weights to it.
def model_fn(model_dir): 
    __logger.info(f"[MODEL_FN] entered, model dir: {model_dir}")
    __logger.info(f"[MODEL_FN] Count of Available GPUs:{torch.cuda.device_count()}")
    __logger.info(f"[MODEL_FN] Initial State of GPU {__gpu}: {torch.cuda.mem_get_info(int(__gpu))[0]/(1024**3), torch.cuda.mem_get_info(int(__gpu))[1]/(1024**3)}")
    __logger.info(f"[MODEL_FN] Initial State of GPU {__gpu1}: {torch.cuda.mem_get_info(int(__gpu1))[0]/(1024**3), torch.cuda.mem_get_info(int(__gpu1))[1]/(1024**3)}")
        
    weights_path_dict = get_weights_paths(model_dir)
    # weights_encoder_path, weights_decoder_path = get_weight_encoder_paths(model_dir)

    __logger.info(f"[MODEL_FN] SSM weights encoder path: {weights_path_dict['weights_encoder_path']}")
    __logger.info(f"[MODEL_FN] SSM weights decoder path: {weights_path_dict['weights_decoder_path']}")
    __logger.info(f"[MODEL_FN] NN Max weights path: {weights_path_dict['NN_Max_weights_path']}")
    __logger.info(f"[MODEL_FN] NN Rest weights path: {weights_path_dict['NN_Rest_weights_path']}")
    global __ssm_cfg
    __ssm_cfg = gen_cfg(weights_path_dict['weights_encoder_path'], weights_path_dict['weights_decoder_path'])

    __logger.info(f"[MODEL_FN] model config: {__ssm_cfg}")

    segmentation_module, gpu = load_ssm_model(__ssm_cfg, gpu=__gpu)
    # segmentation_module = None

    __logger.info(f"[MODEL_FN] SSM Loaded to {__gpu}: {torch.cuda.mem_get_info(int(__gpu))[0]/(1024**3), torch.cuda.mem_get_info(int(__gpu))[1]/(1024**3)}")
    
    # severity_device = torch.device(f"cuda:{__gpu1}" if torch.cuda.is_available() else "cpu")
    severity_device = torch.cuda.set_device(__gpu1)
    severity_module = torch.jit.load(weights_path_dict['NN_Max_weights_path'], map_location=severity_device)
    severity_module = severity_module.to(severity_device)

    __logger.info(f"[MODEL_FN] Loaded Max NN model to device: {__gpu1}: {torch.cuda.mem_get_info(int(__gpu1))[0]/(1024**3), torch.cuda.mem_get_info(int(__gpu1))[1]/(1024**3)}")       

    rest_severity_module = torch.jit.load(weights_path_dict['NN_Rest_weights_path'], map_location=severity_device)
    rest_severity_module = rest_severity_module.to(severity_device)
    
    __logger.info(f"[MODEL_FN] Loaded Rest NN model to device {__gpu1}: {torch.cuda.mem_get_info(int(__gpu1))[0]/(1024**3), torch.cuda.mem_get_info(int(__gpu1))[1]/(1024**3)}")
    
    torch.cuda.empty_cache()
    return severity_module, rest_severity_module, segmentation_module, __gpu, __gpu1#, severity_device
 
# data preprocessing
def input_fn(request_body, request_content_type):
    __logger.info("[INPUT_FN] entered")
    __logger.debug(f"[INPUT_FN] content type: {request_content_type}")
    t0 = datetime.utcnow()
    # input_object = get_input_batch_object(request_body, __ssm_cfg)
    input_object = json.loads(request_body)
    #TODO: Check JSON Format if needed. Assumes Correct for now
    batch_size = len(input_object)
    __logger.info(f"[INPUT_FN] Batch JSON loaded. Preparing {batch_size} Images from S3")
    __logger.info(f"[INPUT_FN] input transform elapsed time: {datetime.utcnow() - t0}")

    return input_object

# inference
def predict_fn(input_object, model):
    __logger.info("[PREDICT_FN] entered")
    # severity_module, segmentation_module, gpu, severity_device = model
    severity_module, rest_severity_module, segmentation_module, gpu, gpu1 = model
    
    b_t0 = datetime.utcnow()
    for image_id, input_object_dict in input_object.items():
        __logger.info(f"[PREDICT_FN] Begining Inference on {image_id}")
        input_dict = get_image_from_input_json(image_id, input_object_dict, __ssm_cfg)
        __logger.info(f"[PREDICT_FN] {image_id}: IQA Check Initialized")
        if iqa_detect_face(input_dict['img_src']):
            __logger.info(f"[PREDICT_FN] {image_id}:IQA Checks Passed; Beginning severity prediction")
            t0 = datetime.utcnow()
            ssm_pred = gen_pred_mask(segmentation_module, __ssm_cfg, input_dict, gpu)
            # ssm_pred = input_dict['img_src']
            __logger.info(f"[PREDICT_FN] {image_id}:SSM inference elapsed time: {datetime.utcnow() - t0}")
            if ssm_pred.any():
                if input_object[image_id]['imageExpression'] in ['Rest', 'R', 'rest', '0', 'r']:
                    pred = pred_severity(ssm_pred, rest_severity_module, gpu1)
                    __logger.info(f"[PREDICT_FN] {image_id}: Rest Severity inference elapsed time: {datetime.utcnow() - t0}")
                else:
                    pred = pred_severity(ssm_pred, severity_module, gpu1)
                    __logger.info(f"[PREDICT_FN] {image_id}: Max Severity inference elapsed time: {datetime.utcnow() - t0}")
            else:
                __logger.info(f"[PREDICT_FN] {image_id}: SSM Zone not Detected; Manually check input")
                pred = -1
        else:
            __logger.info(f"[PREDICT_FN] {image_id}: IQA Checks Failed; Manually check input")
            pred = -1
        input_object[image_id]['Severity'] = pred
    
    __logger.info(f"[PREDICT_FN] Batch Inference Completed in {datetime.utcnow() - b_t0}") 
    return input_object
    
# postprocess
def output_fn(predictions, content_type):
    __logger.info("[OUTPUT_FN] entered")
    __logger.debug(f"[OUTPUT_FN] content type: {content_type}")
    # if predictions is None:
    #     __logger.info("No output saved")
    #     return None
    return json.dumps(predictions)


# ----------------------------------------------------------------------------------------------------
# START: NEW FUNCTIONS
# ----------------------------------------------------------------------------------------------------

def get_weights_paths(model_dir):
    weights_path_dict = {}
    dir_objs = os.listdir(model_dir)
    for obj in dir_objs:
        if ENCODER_KEY in obj:
            weights_path_dict['weights_encoder_path'] = os.path.join(model_dir, obj)
        elif DECODER_KEY in obj:
            weights_path_dict['weights_decoder_path'] = os.path.join(model_dir, obj)
        elif NN_KEY in obj:
            weights_path_dict['NN_Max_weights_path'] = os.path.join(model_dir, obj)
        elif NN_REST_KEY in obj:
            weights_path_dict['NN_Rest_weights_path'] = os.path.join(model_dir, obj)
    # for weight_path in ['weights_encoder_path', 'weights_decoder_path', 'NN_weights_path']:
    #     assert weight_path in weights_path_dict, f"{weight_path} not in weights directory"
    return weights_path_dict

def get_weight_encoder_paths(model_dir):
    weights_encoder_path = None
    weights_decoder_path = None
    dir_objs = os.listdir(model_dir)
    for obj in dir_objs:
        if ENCODER_KEY in obj:
            weights_encoder_path = os.path.join(model_dir, obj)
        elif DECODER_KEY in obj:
            weights_decoder_path = os.path.join(model_dir, obj)

    assert weights_encoder_path != None and weights_decoder_path != None, f"could not find encoder and decoder weights from directory objects: {dir_objs}"

    return weights_encoder_path, weights_decoder_path

def pil_from_s3(bucket_name,
                   img_key):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    s3_object = bucket.Object(img_key)
    response = s3_object.get()
    file_stream = response['Body']
    img = Image.open(file_stream).convert("RGB")
    try:
        img = ImageOps.exif_transpose(img)
    except:
        __logger.info(f"[pil_from_s3] Image {img_key} failed exif_transpose")
    return img

def get_image_from_input_json(image_id, json_dict, temp_cfg):
    output = dict()
    __logger.info(f"[get_image_from_input_json] Image {image_id} imageS3Location:{json_dict['imageS3Location']}")
    __logger.info(f"[get_image_from_input_json] Image {image_id} imageName:{json_dict['imageName']}")
    img = pil_from_s3(bucket_name=json_dict['imageS3Location'], img_key=json_dict['imageName'])
    __logger.info(f"[get_image_from_input_json] Image {image_id} loaded from S3")
    max_dim = max(img.size)
    min_dim = min(img.size)
    temp_cfg.scaling_perc = 0.8
    if min_dim <= 720:
        temp_cfg.scaling_perc = 1.0
        __logger.info(f"[min_dim_check] Image {image_id} size {img.size} scaling perc changed to {temp_cfg.scaling_perc}")

    if 3250/max_dim <= temp_cfg.scaling_perc:
        temp_cfg.scaling_perc = 3250/max_dim
        __logger.info(f"[max_dim_check] Image {image_id} size {img.size} scaling perc changed to {temp_cfg.scaling_perc}")

    if temp_cfg.rotate == "rotate_counterclockwise":
        img = img.rotate(90, expand = True)

    if temp_cfg.rotate == "rotate_clockwise":
        img = img.rotate(270, expand = True)

    if temp_cfg.flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    output['img_src'] = np.array(img)

    if temp_cfg.preprocess == "resize":
        scale = temp_cfg.scaling_perc
        new_w, new_h = int(img.size[0]*scale), int(img.size[1]*scale)
        img = img.resize((new_w,new_h), Image.ANTIALIAS)

    if temp_cfg.preprocess == "resize_and_padding":
        scale = temp_cfg.scaling_perc
        border_padding = temp_cfg.border_padding
        new_w, new_h = int(img.size[0]*scale), int(img.size[1]*scale)
        img = img.resize((new_w,new_h), Image.ANTIALIAS)
        img = ImageOps.expand(img, border=border_padding, fill='black')

    if temp_cfg.preprocess == "padding":
        border_padding = temp_cfg.border_padding
        img = ImageOps.expand(img, border=border_padding, fill='black')

    ori_width, ori_height = img.size

    img_resized_list = []
    for this_short_size in temp_cfg.imgSizes:
        # calculate target height and width
        scale = min(this_short_size / float(min(ori_height, ori_width)),
                    temp_cfg.imgMaxSize / float(max(ori_height, ori_width)))
        target_height, target_width = int(ori_height * scale), int(ori_width * scale)

        # to avoid rounding in network
        target_width = round2nearest_multiple(target_width, temp_cfg.padding_constant)
        target_height = round2nearest_multiple(target_height, temp_cfg.padding_constant)

        # resize images
        img_resized = imresize(img, (target_width, target_height), interp=IMAGE_RESIZE_INTERP)

        # image transform, to torch float tensor 3xHxW
        img_resized = img_transform(img_resized)
        img_resized = torch.unsqueeze(img_resized, 0)
        img_resized_list.append(img_resized)

    output['img_ori'] = np.array(img)
    output['img_data'] = [x.contiguous() for x in img_resized_list]

    return output

        

# ----------------------------------------------------------------------------------------------------
# START: UPDATED FUNCTIONS
# ----------------------------------------------------------------------------------------------------


def gen_pred_mask(segmentation_module, temp_cfg, input_object, gpu, output_label=6):
    segmentation_module.eval()

    segSize = (input_object['img_ori'].shape[0],
               input_object['img_ori'].shape[1])

    img_resized_list = input_object['img_data']

    with torch.no_grad():
        scores = torch.zeros(1, temp_cfg.num_class, segSize[0], segSize[1])
        scores = async_copy_to(scores, gpu)
        for img in img_resized_list:
            feed_dict = input_object.copy()
            feed_dict['img_data'] = img
            del feed_dict['img_src']
            del feed_dict['img_ori']
            feed_dict = async_copy_to(feed_dict, gpu)

            # forward pass
            pred_tmp = segmentation_module(feed_dict, segSize=segSize)
            scores = scores + pred_tmp / len(temp_cfg.imgSizes)

        _, pred = torch.max(scores, dim=1)
        pred = as_numpy(pred.squeeze(0).cpu())

    # process pred
    pred = np.uint8(pred)

    # post_pred removes padding and resizes to src
    post_pred = postprocess_pred(pred, input_object['img_src'], temp_cfg)
    post_pred = post_pred.astype(np.uint8)
    pred_zones = np.array(Image.fromarray(post_pred).getcolors(),dtype=object)[:,1]
    if output_label not in pred_zones:
        __logger.info(f"Zone not detected")
        return np.array([])
    post_pred[post_pred!=output_label] = 0
    post_pred[post_pred==output_label] = 1
    stacked_pred = np.stack((post_pred,)*3, axis=-1).astype(np.uint8)
    output_mask = input_object['img_src']*stacked_pred

    active_pixels = np.stack(np.where(post_pred))
    top_left = np.min(active_pixels, axis=1).astype(np.int32)
    bottom_right = np.max(active_pixels, axis=1).astype(np.int32)

    output_mask = output_mask[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

    return output_mask.astype(np.uint8)


def gen_cfg(weights_encoder_path, weights_decoder_path):
    temp_cfg = CN()
    temp_cfg.weights_encoder=weights_encoder_path
    temp_cfg.weights_decoder=weights_decoder_path
    temp_cfg.arch_encoder="hrnetv2"
    temp_cfg.arch_decoder="c1"
    temp_cfg.num_class=7
    temp_cfg.fc_dim=720
    temp_cfg.preprocess="resize_and_padding"
    temp_cfg.rotate=False
    temp_cfg.flip=False
    temp_cfg.scaling_perc=0.8
    temp_cfg.border_padding=100
    temp_cfg.imgSizes=(300, 375, 450, 525, 600)
    temp_cfg.imgMaxSize=1000
    temp_cfg.padding_constant=32

    return temp_cfg

def pred_severity(ssm_pred, severity_model, gpu):
        # decoded = Image.open(io.BytesIO(request_body))
    # __logger.info(f"Loaded Image: {decoded.size}")
    decoded = Image.fromarray(ssm_pred)
    preprocess = transforms.Compose([
                transforms.RandomGrayscale(p=1),
                transforms.Resize(512),
                transforms.CenterCrop(1000),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])
    normalized = preprocess(decoded)
    batchified = normalized.unsqueeze(0)
    
    # predict
    # device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device = torch.cuda.set_device(gpu)
    else:
        device = torch.device("cpu")
        
    batchified = batchified.to(device)
    # output = model.forward(batchified)
    severity_model.eval()
    severity_model = torch.jit.freeze(severity_model)
    with torch.no_grad():
        output = severity_model(batchified)
        _, pred = torch.max(output, 1)
        pred = pred.squeeze(0).cpu().numpy().tolist()
        # output = output.squeeze(0).cpu().numpy().tolist()
    return pred #{"Severity": pred}
# ----------------------------------------------------------------------------------------------------
# START: EXISTING HELPER FUNCTIONS
# ----------------------------------------------------------------------------------------------------


def round2nearest_multiple(x, p):
    return ((x - 1) // p + 1) * p


def img_transform(img):
    # 0-255 to 0-1
    img = np.float32(np.array(img)) / 255.
    img = img.transpose((2, 0, 1))
    img = normalize(torch.from_numpy(img.copy()))
    return img


def imresize(im, size, interp='bilinear'):
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')

    return im.resize(size, resample)


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])


# ----------------------------------------------------------------------------------------------------
# START: EXISTING INFERENCE FUNCTIONS
# ----------------------------------------------------------------------------------------------------


def load_ssm_model(temp_cfg, gpu=0):
    if not os.path.exists(temp_cfg.weights_encoder):
        temp_cfg.weights_encoder = os.path.join(
                                    os.path.dirname(__file__), 
                                    temp_cfg.weights_encoder)
    if not os.path.exists(temp_cfg.weights_decoder):
        temp_cfg.weights_decoder = os.path.join(
                                    os.path.dirname(__file__), 
                                    temp_cfg.weights_decoder)
    
    assert os.path.exists(temp_cfg.weights_encoder) and \
           os.path.exists(temp_cfg.weights_decoder), f"checkpoint does not exist at {temp_cfg.weights_encoder} and/or {temp_cfg.weights_decoder}!"

    # torch.cuda.set_device(gpu)
    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=temp_cfg.arch_encoder,
        fc_dim=temp_cfg.fc_dim,
        weights=temp_cfg.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=temp_cfg.arch_decoder,
        fc_dim=temp_cfg.fc_dim,
        num_class=temp_cfg.num_class,
        weights=temp_cfg.weights_decoder,
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=-1)
    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
    # ssm_device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    ssm_device = torch.cuda.set_device(gpu)
    segmentation_module.to(ssm_device)

    return segmentation_module, gpu


def postprocess_pred(pred, img, cfg):
    if cfg.preprocess == "resize":
        return cv2.resize(pred, (img.shape[1], img.shape[0]), interpolation = cv2.INTER_NEAREST)
    elif cfg.preprocess == "resize_and_padding":
        border_padding = cfg.border_padding
        pred = pred[border_padding:-border_padding or None,border_padding:-border_padding or None]
        return cv2.resize(pred, (img.shape[1], img.shape[0]), interpolation = cv2.INTER_NEAREST)
    elif cfg.preprocess == "padding":
        border_padding = cfg.border_padding
        pred = pred[border_padding:-border_padding or None,border_padding:-border_padding or None]
        return pred
    else:
        return pred

    
### IQA CODE
def load_mp_detector():
    mp_detector = MediaPipeFacialDetector.load_mp_detector()
    return mp_detector

def mp_pred(image_array, mp_detector):      
    mp_detect = MediaPipeFacialDetector.mp_detect(image_array, mp_detector)
    return mp_detect

def iqa_detect_face(image_array):
    mp_detector = load_mp_detector()
    mp_detect = mp_pred(image_array, mp_detector)
    if mp_detect.face_landmarks!=[]:
        return True
    else:
        return False

# def process_mp_pred(mp_detect, image_array):
#     mp_dict = MediaPipeFacialDetector.process_detection(mp_detect, image_array)
#     return mp_dict

# def detect_face(mp_detect):
#     if mp_detect.face_landmarks!=[]:
#         return {'detected_face': True}
#     else:
#         return {'detected_face': False}