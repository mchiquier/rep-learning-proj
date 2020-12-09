import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models
from PIL import Image
import pdb
import imageio
# from img_utils import *

import csv
import os
import subprocess
import sys
import numpy as np

from tqdm import tqdm

from taskonomy_network import (
    TaskonomyEncoder,
    TaskonomyDecoder,
    TaskonomyNetwork,
    TASKONOMY_PRETRAINED_URLS,
    TASKS_TO_CHANNELS,
    PIX_TO_PIX_TASKS,
    DONT_APPLY_TANH_TASKS,
)

from scipy.ndimage.interpolation import zoom
from skimage.transform import resize
import skimage
from skimage import io
from skimage import img_as_ubyte

from PIL import Image, ImageFile
import PIL

ImageFile.LOAD_TRUNCATED_IMAGES = True


class FeatureExtractor:
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs:
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model.encoder._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0), -1)
            else:
                x = module(x)
        x = torch.reshape(x, (-1, 2048))
        for name, module in self.model.decoder._modules.items():
            x = module(x)
        return target_activations, x


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input


def show_cam_on_image(img, mask):
    #pdb.set_trace()
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = 0.4 * heatmap + 0.6 * np.float32(img)
    cam = cam / np.max(cam)
    # cv2.imwrite("here.png", cam*255)
    # cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    
    return cam



class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        # cam = np.maximum(cam, 0)
        # cam = cam - np.min(cam)
        # cam = cam / np.max(cam)
        cam = rescale_image(cam, new_scale=[0, 1.0])
        cam = cv2.resize(cam, input.shape[2:])
        return cam



class GuidedBackpropReLU(Function):
    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(
            torch.zeros(input.size()).type_as(input), input, positive_mask
        )
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(
            torch.zeros(input.size()).type_as(input),
            torch.addcmul(
                torch.zeros(input.size()).type_as(input), grad_output, positive_mask_1
            ),
            positive_mask_2,
        )

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda, gpu):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.to(gpu)

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == "ReLU":
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.to(gpu))
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.to(gpu) * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        default=False,
        help="Use NVIDIA GPU acceleration",
    )
    parser.add_argument(
        "--image-path", type=str, default="./examples/both.png", help="Input image path"
    )
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


def load_raw_image( filename, color=True, use_pil=False ):
    """
    Load an image converting from grayscale or alpha as needed.
    Adapted from KChen
    Args:
        filename : string
        color : boolean
            flag for color format. True (default) loads as RGB while False
            loads as intensity (if image is already grayscale).
    Returns
        image : an image with image original dtype and image pixel range
            of size (H x W x 3) in RGB or
            of size (H x W x 1) in grayscale.
    """
    if use_pil:
        img = Image.open( filename )
    else:
        img = io.imread(filename, as_gray= (not color))

    if use_pil:
        return img

    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        if color:
            img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
 
    return img

def resize_rescale_image(
    img, new_dims, new_scale, interp_order=1, current_scale=None, no_clip=False
):
    """
    Resize an image array with interpolation, and rescale to be
      between
    Parameters
    ----------
    im : (H x W x K) ndarray
    new_dims : (height, width) tuple of new dimensions.
    new_scale : (min, max) tuple of new scale.
    interp_order : interpolation order, default is linear.
    Returns
    -------
    im : resized ndarray with shape (new_dims[0], new_dims[1], K)
    """
    img = skimage.img_as_float(img)
    img = resize_image(img, new_dims, interp_order)
    img = rescale_image(img, new_scale, current_scale=current_scale, no_clip=no_clip)

    return img


def resize_image(im, new_dims, interp_order=1):
    """
    Resize an image array with interpolation.
    Parameters
    ----------
    im : (H x W x K) ndarray
    new_dims : (height, width) tuple of new dimensions.
    interp_order : interpolation order, default is linear.
    Returns
    -------
    im : resized ndarray with shape (new_dims[0], new_dims[1], K)
    By kchen @ https://github.com/kchen92/joint-representation/blob/24b30ca6963d2ec99618af379c1e05e1f7026710/lib/data/input_pipeline_feed_dict.py
    """
    if type(im) == PIL.PngImagePlugin.PngImageFile:
        interps = [PIL.Image.NEAREST, PIL.Image.BILINEAR]
        return skimage.util.img_as_float(im.resize(new_dims, interps[interp_order]))

    if all(new_dims[i] == im.shape[i] for i in range(len(new_dims))):
        resized_im = im  # return im.astype(np.float32)
    elif im.shape[-1] == 1 or im.shape[-1] == 3:
        resized_im = resize(im, new_dims, order=interp_order, preserve_range=True)
    else:
        # ndimage interpolates anything but more slowly.
        scale = tuple(np.array(new_dims, dtype=float) / np.array(im.shape[:2]))
        resized_im = zoom(im, scale + (1,), order=interp_order)
    return resized_im


def rescale_image(im, new_scale=[-1.0, 1.0], current_scale=None, no_clip=False):
    """
    Rescales an image pixel values to target_scale

    Args:
        img: A np.float_32 array, assumed between [0,1]
        new_scale: [min,max]
        current_scale: If not supplied, it is assumed to be in:
            [0, 1]: if dtype=float
            [0, 2^16]: if dtype=uint
            [0, 255]: if dtype=ubyte
    Returns:
        rescaled_image
    """
    im = skimage.img_as_float(im).astype(np.float32)
    # im = im.astype(np.float32)
    if current_scale is not None:
        min_val, max_val = current_scale
        if not no_clip:
            im = np.clip(im, min_val, max_val)
        im = im - min_val
        im /= max_val - min_val
    min_val, max_val = new_scale
    im *= max_val - min_val
    im += min_val
    # print(im.max())
    # print(im.min())
    return im



def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)


def read_image_resize(current_img_path, resize=(0, 0)):

    im = Image.open(current_img_path)

    if resize[0] != 0:
        img = resample_lanczos(im, resize[0], resize[1])
    else:
        img = np.array(im)

    img = np.float32(img) / 255
    return img


def resample_lanczos(im, W, H):

    """Resize image to size (W, H)."""
    new_size = (W, H)
    im = im.resize(new_size, Image.LANCZOS)

    return np.array(im)


def normalize(original_img):

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    # preprocessed_img = original_img

    if len(original_img.shape) == 4:
        for i in range(3):
            original_img[:, i, :, :] = original_img[:, i, :, :] - means[i]
            original_img[:, i, :, :] = original_img[:, i, :, :] / stds[i]

    else:
        print("missing a dimension!")

    return original_img


def convert_whc_to_cwh(img):

    if torch.is_tensor(img):
        if len(img.shape) == 4:
            preprocessed_img = img.permute(0, 3, 1, 2)
        else:
            preprocessed_img = img.permute(2, 0, 1)
    else:
        if len(img.shape) == 4:
            preprocessed_img = np.transpose(img, (0, 3, 1, 2))
        else:
            preprocessed_img = np.transpose(img, (2, 0, 1))

    return preprocessed_img


saving_model_path = "./post_trained_normals.pth"
# default_device = "cuda" if torch.cuda.is_available() else "cpu"
default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TASKONOMY_LOCATION = "https://github.com/StanfordVL/taskonomy/tree/master/taskbank"

fname_task = 'class_object'
# out_channels = TASKS_TO_CHANNELS[fname_task]
# encoder_path = TASKONOMY_PRETRAINED_URLS[fname_task + "_encoder"]
# decoder_path = TASKONOMY_PRETRAINED_URLS[fname_task + "_decoder"],

is_decoder_mlp = fname_task not in PIX_TO_PIX_TASKS
apply_tanh = fname_task not in DONT_APPLY_TANH_TASKS


net_paths_to_load = []
net_paths_to_load.append(
    (
        TASKS_TO_CHANNELS[fname_task],
        TASKONOMY_PRETRAINED_URLS[fname_task + "_encoder"],
        TASKONOMY_PRETRAINED_URLS[fname_task + "_decoder"],
        is_decoder_mlp,
        apply_tanh,
    )
)


nets = []
for (
    out_channels,
    encoder_path,
    decoder_path,
    is_decoder_mlp,
    apply_tanh,
) in net_paths_to_load:
    nets.append(
        TaskonomyNetwork(
            out_channels=out_channels,
            load_encoder_path=encoder_path,
            load_decoder_path=decoder_path,
            model_dir=None,
            is_decoder_mlp=is_decoder_mlp,
            apply_tanh=apply_tanh,
            progress=True,
        )
    )

decoder_nets = nets[0].to(default_device)
for p in decoder_nets.parameters():
    p.requires_grad = True


""" python grad_cam.py <path_to_image>
1. Loads an image with opencv.
2. Preprocesses it for VGG19 and converts to a pytorch variable.
3. Makes a forward pass to find the category index with the highest score,
and computes intermediate activations.
Makes the visualization. """

# Can work with any model, but it assumes that the model has a
# feature method, and a classifier method,
# as in the VGG models in torchvision

# Use our decoder
# print(model.encoder.layer4)

model = decoder_nets
grad_cam = GradCam(
    model=model,
    feature_module=model.encoder.layer4,
    target_layer_names=["2"],
    use_cuda=True,
)

(w, h) = 256, 256

# Use Resnet
# print(model_resnet.layer4)

# model_resnet = models.resnet50(pretrained=True)
# grad_cam = GradCam(
#     model=model_resnet,
#     feature_module=model_resnet.layer4,
#     target_layer_names=["2"],
#     use_cuda=True,
# )

# (w, h) = 224, 224


#pdb.set_trace()
# current_img_path = "../rep_learning_data/rgb/aldine/rgb/point_0_view_0_domain_rgb.png"
# current_class_path = "../rep_learning_data/aldine_class_object/class_object/point_0_view_0_domain_class_object.npy"

## Load data on server + run grad_cam on all results

csv_file = "joint4--test--full.csv"
len1 = len("aldine/rgb/")
image_path = []
class_path = []
with open(csv_file) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        fname_img = "data/taskonomy_rgb_full/test/rgb/" + row[0][len1:]
        fname_class = "data/taskonomy_class_obj_full/test/class_object/" + row[0][len1:-7] + "class_object.npy"
        image_path.append(fname_img)
        class_path.append(fname_class)

results_dir = "grad_cam_results_before_posttraining"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

len2 = len("data/taskonomy_rgb_full/test/rgb/")

# print(image_path[0])
# print(class_path[0])
# for img_path, cls_path in tqdm(zip(image_path, class_path)):


img_path = "data/taskonomy_rgb_full/test/rgb/point_0_view_0_domain_rgb.png"
cls_path = "data/taskonomy_class_obj_full/test/class_object/point_0_view_0_domain_class_object.npy"

theclasses = np.load(cls_path)
target_index = np.argmax(theclasses)

original_img = load_raw_image(img_path)
original_img = resize_rescale_image(
    original_img,
    (256, 256),
    new_scale=[-1., 1.0],
    interp_order=1,
    current_scale=[0, 1.0],
    no_clip=False,
)

# original_img = read_image_resize(img_path, (w, h))
img = np.expand_dims(original_img, axis=0)
original_img = torch.tensor(convert_whc_to_cwh(img))
# the_input = normalize(original_img).requires_grad_(True)
the_input = original_img.requires_grad_(True)
img_mult = torch.cat(5 * [the_input], axis=0)

target_index = None
mask = grad_cam(img_mult, target_index)

cam = show_cam_on_image(img[0], mask)
fname_save = results_dir + '/' + img_path[len2:]
cam = resize_rescale_image(
    cam,
    (256, 256),
    new_scale=[0, 255],
    interp_order=1,
    current_scale=[-1., 1.0],
    no_clip=False,
)
cv2.imwrite(fname_save, cam)
