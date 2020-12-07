import multiprocessing.dummy as mp
import torch
import torch.utils.model_zoo
import pdb
# import imageio
import torch.optim as optim
import torch.nn as nn
from PIL import Image
import PIL

import numpy as np

from torch.utils.data.dataloader import DataLoader
import os

from scipy.ndimage.interpolation import zoom
from skimage.transform import resize
import skimage
from skimage import io
from skimage import img_as_ubyte

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from taskonomy_network import (
    TaskonomyEncoder,
    TaskonomyDecoder,
    TaskonomyNetwork,
    TASKONOMY_PRETRAINED_URLS,
    TASKS_TO_CHANNELS,
    PIX_TO_PIX_TASKS,
    DONT_APPLY_TANH_TASKS,
)


def read_image_resize(current_img_path, resize=(0, 0)):

    im = Image.open(current_img_path)

    if resize[0] != 0:
        img = resample_lanczos(im, resize[0], resize[1])
    else:
        img = np.array(im, dtype=np.uint8)
    return img

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
    # resized_im = resized_im.astype(np.float32)
    # print(resized_im.max())
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


class Dataset(torch.utils.data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, imgs_path, labels_path, task_name):
        "Initialization"

        self.labels_path = labels_path
        self.imgs_path = imgs_path
        self.task_name = task_name

        self.labels = os.listdir(labels_path)
        self.images = os.listdir(imgs_path)
        # pdb.set_trace()

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.images)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        current_img = self.images[index]

        img = load_raw_image(self.imgs_path + "/" + current_img)
        img = resize_rescale_image(
            img,
            (256, 256),
            new_scale=[-1., 1.0],
            interp_order=1,
            current_scale=[0, 1.0],
            no_clip=False,
        ) 
        img = torch.tensor(img.transpose((2, 0, 1))).type(torch.FloatTensor)
        
        if self.task_name == "class_object": 
            current_label = self.images[index].split(".")[0][:-3] + self.task_name + ".npy"
            label = np.load(self.labels_path + "/" + current_label, allow_pickle=True)
            label = torch.tensor(label).type(torch.FloatTensor)

        elif self.task_name == "normal":
            current_label = self.images[index].split(".")[0][:-3] + self.task_name + ".png"
            label = load_raw_image(self.labels_path + "/" + current_label)
            label = resize_rescale_image(
                label,
                (256, 256),
                new_scale=[-1.0, 1.0],
                interp_order=1,
                current_scale=[0, 1.0],
                no_clip=False,
            )
            label = torch.tensor(label.transpose((2, 0, 1))).type(torch.FloatTensor)

        return (
            img, label
        )

params = {"batch_size": 32, "shuffle": True, "num_workers": 6}
params_test = {"batch_size": 128, "shuffle": True, "num_workers": 6}

### To get Encoder + Decoder
fname_task = "class_object"

train_dataset = Dataset(
    "data/taskonomy_rgb_full/train/rgb",
    "data/taskonomy_class_obj_full/train/class_object",
    fname_task
)
# test_dataset = Dataset(
#     "data/rgb",
#     "data/class_object",
#     fname_task
# )

saving_model_path = "./trained_classifier.pth"

training_generator = torch.utils.data.DataLoader(train_dataset, **params)
# test_generator = torch.utils.data.DataLoader(test_dataset, **params_test)

default_device = "cuda" if torch.cuda.is_available() else "cpu"
TASKONOMY_LOCATION = "https://github.com/StanfordVL/taskonomy/tree/master/taskbank"

## Define Model
# is_decoder_mlp = feature_task not in PIX_TO_PIX_TASKS
# print("is_decoder_mlp")
# print(is_decoder_mlp)
is_decoder_mlp= True
# apply_tanh = feature_task not in DONT_APPLY_TANH_TASKS
# print("apply_tanh")
# print(apply_tanh)
apply_tanh = True

out_channels = TASKS_TO_CHANNELS[fname_task]
encoder_path = TASKONOMY_PRETRAINED_URLS[fname_task + "_encoder"]
decoder_path = TASKONOMY_PRETRAINED_URLS[fname_task + "_decoder"],

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

model_path = 'trained_classifier.pth'
decoder_nets = nets[0].to(default_device)
decoder_nets_checkpoint = torch.load(model_path)
decoder_nets.load_state_dict(decoder_nets_checkpoint)
decoder_nets.eval()

# Define loss & optimizer 

curr_dataset = Dataset(
    "data/taskonomy_rgb_full/test/rgb",
    "data/taskonomy_class_obj_full/test/class_object",
    fname_task
)
training_generator = torch.utils.data.DataLoader(curr_dataset, **params_test)

def isin2daccuracy(arr1, arr2):
    j = 0
    for i in range(arr1.shape[0]):
        if np.isin(arr1[i], arr2[i]):
            j += 1
    return j / arr1.shape[0]



batch_id = 0
for img, label in training_generator:
    img, label = img.to(default_device), label.to(default_device)
    print(batch_id)
    m = torch.nn.Softmax(dim=1)
    output_decoder = m(decoder_nets(img))
    output_numpy = output_decoder.cpu().data.numpy()
    label_numpy = label.cpu().data.numpy()
    top5class = output_numpy[0].argsort()[::-1][:5]
    print("Prediction")
    
    print(top5class)
    print(output_numpy[0][top5class])
    print("Label")
    top5labelclass = label_numpy[0].argsort()[::-1][:5]
    print(top5labelclass)
    print(label_numpy[0][top5labelclass])
    print(
        "MSE : " + str(((label_numpy - output_numpy) ** 2).sum() / params_test["batch_size"])
    )
    print(
        "Accuracy top prediction : "
        + str(
            (label_numpy.argmax(1) == output_numpy.argmax(1)).sum()
            / params_test["batch_size"]
        )
    )
    print(
        "Is in top 5 accuracy : "
        + str(
            isin2daccuracy(
                output_numpy.argmax(1), label_numpy.argsort()[:, ::-1][:, :5]
            )
        )
    )
    batch_id += 1
