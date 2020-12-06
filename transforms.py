from taskonomy_network import (
    TaskonomyEncoder,
    TaskonomyDecoder,
    TaskonomyNetwork,
    TASKONOMY_PRETRAINED_URLS,
    TASKS_TO_CHANNELS,
)
import multiprocessing.dummy as mp
import torch

default_device = "cuda" if torch.cuda.is_available() else "cpu"
from taskonomy_network import (
    TaskonomyEncoder,
    TaskonomyDecoder,
    TaskonomyNetwork,
    TASKONOMY_PRETRAINED_URLS,
    TASKS_TO_CHANNELS,
    PIX_TO_PIX_TASKS,
    DONT_APPLY_TANH_TASKS,
)
import multiprocessing.dummy as mp
import torch
import torch.utils.model_zoo
import pdb
import imageio

from PIL import Image
import PIL

import numpy as np

from torch.utils.data.dataloader import DataLoader
import os

from scipy.ndimage.interpolation import zoom
from skimage.transform import resize
import skimage

default_device = "cuda" if torch.cuda.is_available() else "cpu"
TASKONOMY_LOCATION = "https://github.com/StanfordVL/taskonomy/tree/master/taskbank"


def representation_transform(img, feature_task="normal", device=default_device):
    """
    Transforms an RGB image into a feature driven by some vision task
        Expects inputs:
            shape  (batch_size, 3, 256, 256)
            values [-1,1]
        Outputs:
            shape  (batch_size, 8, 16, 16)
    """
    return VisualPrior.to_representation(
        img, feature_tasks=[feature_task], device=device
    )


def multi_representation_transform(
    img, feature_tasks=["normal"], device=default_device
):
    """
    Transforms an RGB image into a features driven by some vision tasks
        Expects inputs:
            shape  (batch_size, 3, 256, 256)
            values [-1,1]
        Outputs:
            shape  (batch_size, 8, 16, 16)
    """
    return VisualPrior.to_representation(img, feature_tasks, device)


def max_coverage_featureset_transform(img, k=4, device=default_device):
    """
    Transforms an RGB image into a features driven by some vision tasks.
    The tasks are chosen according to the Max-Coverage Min-Distance Featureset
    From the paper:
        Mid-Level Visual Representations Improve Generalization and Sample Efficiency
            for Learning Visuomotor Policies.
        Alexander Sax, Bradley Emi, Amir R. Zamir, Silvio Savarese, Leonidas Guibas, Jitendra Malik.
        Arxiv preprint 2018.
    This function expects inputs:
            shape  (batch_size, 3, 256, 256)
            values [-1,1]
        Outputs:
            shape  (batch_size, 8*k, 16, 16)
    """
    return VisualPrior.max_coverage_transform(img, k, device)


def feature_readout(img, feature_task="normal", device=default_device):
    """
    Transforms an RGB image into a feature driven by some vision task,
    then returns the result of a readout of the feature.
        Expects inputs:
            shape  (batch_size, 3, 256, 256)
            values [-1,1]
        Outputs:
            shape  (batch_size, 8, 16, 16)
    """
    return VisualPrior.to_predicted_label(
        img, feature_tasks=[feature_task], device=device
    )


def multi_feature_readout(img, feature_tasks=["normal"], device=default_device):
    """
    Transforms an RGB image into a features driven by some vision tasks
    then returns the readouts of the features.
        Expects inputs:
            shape  (batch_size, 3, 256, 256)
            values [-1,1]
        Outputs:
            shape  (batch_size, 8, 16, 16)
    """
    return VisualPrior.to_predicted_label(img, feature_tasks, device)


class VisualPrior(object):

    max_coverate_featuresets = [
        ["autoencoding"],
        ["segment_unsup2d", "segment_unsup25d"],
        ["edge_texture", "reshading", "curvature"],
        ["normal", "keypoints2d", "segment_unsup2d", "segment_semantic"],
    ]
    model_dir = None
    viable_feature_tasks = [
        "autoencoding",
        "colorization",
        "curvature",
        "denoising",
        "edge_texture",
        "edge_occlusion",
        "egomotion",
        "fixated_pose",
        "jigsaw",
        "keypoints2d",
        "keypoints3d",
        "nonfixated_pose",
        "point_matching",
        "reshading",
        "depth_zbuffer",
        "depth_euclidean",
        "normal",
        "room_layout",
        "segment_unsup25d",
        "segment_unsup2d",
        "segment_semantic",
        "class_object",
        "class_scene",
        "inpainting",
        "vanishing_point",
    ]

    @classmethod
    def to_representation(cls, img, feature_tasks=["normal"], device=default_device):
        """
        Transforms an RGB image into a feature driven by some vision task(s)
        Expects inputs:
            shape  (batch_size, 3, 256, 256)
            values [-1,1]
        Outputs:
            shape  (batch_size, 8, 16, 16)
        This funciton is technically unsupported and there are absolutely no guarantees.
        """
        VisualPriorRepresentation._load_unloaded_nets(feature_tasks)
        for t in feature_tasks:
            VisualPriorRepresentation.feature_task_to_net[
                t
            ] = VisualPriorRepresentation.feature_task_to_net[t].to(device)
        nets = [VisualPriorRepresentation.feature_task_to_net[t] for t in feature_tasks]
        with torch.no_grad():
            return torch.cat([net(img) for net in nets], dim=1)

    @classmethod
    def to_predicted_label(cls, img, feature_tasks=["normal"], device=default_device):
        """
        Transforms an RGB image into a predicted label for some task.
        Expects inputs:
            shape  (batch_size, 3, 256, 256)
            values [-1,1]
        Outputs:
            shape  (batch_size, C, 256, 256)
            values [-1,1]
        This funciton is technically unsupported and there are absolutely no guarantees.
        """
        VisualPriorPredictedLabel._load_unloaded_nets(feature_tasks)
        for t in feature_tasks:
            VisualPriorPredictedLabel.feature_task_to_net[
                t
            ] = VisualPriorPredictedLabel.feature_task_to_net[t].to(device)
        nets = [VisualPriorPredictedLabel.feature_task_to_net[t] for t in feature_tasks]
        with torch.no_grad():
            return torch.cat([net(img) for net in nets], dim=1)

    @classmethod
    def max_coverage_transform(cls, img, k=4, device=default_device):
        assert (
            k > 0
        ), "Number of features to use for the max_coverage_transform must be > 0"
        if k > 4:
            raise NotImplementedError(
                "max_coverage_transform featureset not implemented for k > 4"
            )
        return cls.to_representation(
            img, feature_tasks=max_coverate_featuresets[k - 1], device=device
        )

    @classmethod
    def set_model_dir(model_dir):
        cls.model_dir = model_dir


class VisualPriorRepresentation(object):
    """
    Handles loading networks that transform images into encoded features.
    Expects inputs:
        shape  (batch_size, 3, 256, 256)
        values [-1,1]
    Outputs:
        shape  (batch_size, 8, 16, 16)
    """

    feature_task_to_net = {}

    @classmethod
    def _load_unloaded_nets(cls, feature_tasks, model_dir=None):
        net_paths_to_load = []
        feature_tasks_to_load = []
        for feature_task in feature_tasks:
            if feature_task not in cls.feature_task_to_net:
                net_paths_to_load.append(
                    TASKONOMY_PRETRAINED_URLS[feature_task + "_encoder"]
                )
                feature_tasks_to_load.append(feature_task)
        nets = cls._load_networks(net_paths_to_load)
        for feature_task, net in zip(feature_tasks_to_load, nets):
            cls.feature_task_to_net[feature_task] = net
        return nets

    @classmethod
    def _load_networks(cls, network_paths, model_dir=None):
        return [cls._load_encoder(url, model_dir) for url in network_paths]

    @classmethod
    def _load_encoder(cls, url, model_dir=None, progress=True):
        net = TaskonomyEncoder()  # .cuda()
        net.eval()
        checkpoint = torch.utils.model_zoo.load_url(
            url, model_dir=model_dir, progress=progress
        )
        net.load_state_dict(checkpoint["state_dict"])
        for p in net.parameters():
            p.requires_grad = False
        # net = Compose(nn.GroupNorm(32, 32, affine=False), net)
        return net


class VisualPriorPredictedLabel(object):
    """
    Handles loading networks that transform images into transformed images.
    Expects inputs:
        shape  (batch_size, 3, 256, 256)
        values [-1,1]
    Outputs:
        shape  (batch_size, C, 256, 256)
        values [-1,1]

    This class is technically unsupported and there are absolutely no guarantees.
    """

    feature_task_to_net = {}

    @classmethod
    def _load_unloaded_nets(cls, feature_tasks, model_dir=None):
        net_paths_to_load = []
        feature_tasks_to_load = []
        for feature_task in feature_tasks:
            if feature_task not in cls.feature_task_to_net:
                if feature_task not in TASKS_TO_CHANNELS:
                    raise NotImplementedError(
                        "Task {} not implemented in VisualPriorPredictedLabel. Recommended to get predictions from {}".format(
                            feature_task, TASKONOMY_LOCATION
                        )
                    )
                is_decoder_mlp = feature_task not in PIX_TO_PIX_TASKS
                apply_tanh = feature_task not in DONT_APPLY_TANH_TASKS
                net_paths_to_load.append(
                    (
                        TASKS_TO_CHANNELS[feature_task],
                        TASKONOMY_PRETRAINED_URLS[feature_task + "_encoder"],
                        TASKONOMY_PRETRAINED_URLS[feature_task + "_decoder"],
                        is_decoder_mlp,
                        apply_tanh,
                    )
                )
                feature_tasks_to_load.append(feature_task)
        nets = cls._load_networks(net_paths_to_load)
        for feature_task, net in zip(feature_tasks_to_load, nets):
            cls.feature_task_to_net[feature_task] = net

        return nets

    @classmethod
    def _load_networks(cls, network_paths, model_dir=None, progress=True):
        nets = []
        for (
            out_channels,
            encoder_path,
            decoder_path,
            is_decoder_mlp,
            apply_tanh,
        ) in network_paths:
            nets.append(
                TaskonomyNetwork(
                    out_channels=out_channels,
                    load_encoder_path=encoder_path,
                    load_decoder_path=decoder_path,
                    model_dir=model_dir,
                    is_decoder_mlp=is_decoder_mlp,
                    apply_tanh=apply_tanh,
                    progress=progress,
                )
            )
        return nets


def read_image_resize(current_img_path, resize=(0, 0)):

    im = Image.open(current_img_path)

    if resize[0] != 0:
        img = resample_lanczos(im, resize[0], resize[1])
    else:
        img = np.array(im, dtype=np.uint8)
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
        current_label = self.images[index].split(".")[0][:-3] + self.task_name + ".npy"

        img = read_image_resize(self.imgs_path + "/" + current_img)
        img = resize_rescale_image(
            img,
            (256, 256),
            new_scale=[-1.0, 1.0],
            interp_order=1,
            current_scale=[0, 1.0],
            no_clip=False,
        ) 
        img = torch.tensor(img.transpose((2, 0, 1))).type(torch.FloatTensor)
        
        if self.task_name == "class_object": 
        # Load data and get label
        
        #label = read_image_resize(self.labels_path + "/" + current_label)
            label = np.load(self.labels_path + "/" + current_label, allow_pickle=True)
            label = torch.tensor(label).type(torch.FloatTensor)

        elif self.task_name == "normal":
        
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

params = {"batch_size": 128, "shuffle": True, "num_workers": 6}

def isin2daccuracy(arr1, arr2):
    j = 0
    for i in range(arr1.shape[0]):
        if np.isin(arr1[i], arr2[i]):
            j += 1
    return j / arr1.shape[0]


fname_task = "class_object"

### To get Encoder + Decoder
newrep = VisualPriorPredictedLabel()
decoder_nets = newrep._load_unloaded_nets([fname_task])

curr_dataset = Dataset(
    "../../rep_learning_data/rgb/aldine/rgb",
    "../../rep_learning_data/aldine_class_object/class_object",
    fname_task
)

training_generator = torch.utils.data.DataLoader(curr_dataset, **params)

batch_id = 0
for img, label in training_generator:
    print(batch_id)
    m = torch.nn.Softmax(dim=1)
    output_decoder = m(decoder_nets[0](img))
    output_numpy = output_decoder.data.numpy()
    label_numpy = label.data.numpy()
    top5class = output_numpy[0].argsort()[::-1][:5]
    print("Prediction")
    
    print(top5class)
    print(output_numpy[0][top5class])
    print("Label")
    top5labelclass = label_numpy[0].argsort()[::-1][:5]
    print(top5labelclass)
    print(label_numpy[0][top5labelclass])
    print(
        "MSE : " + str(((label_numpy - output_numpy) ** 2).sum() / params["batch_size"])
    )
    print(
        "Accuracy top prediction : "
        + str(
            (label_numpy.argmax(1) == output_numpy.argmax(1)).sum()
            / params["batch_size"]
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
    

#########################################################################################

"""
fname_task = "depth_euclidian"

### To get Encoder + Decoder
newrep = VisualPriorPredictedLabel()
decoder_nets = newrep._load_unloaded_nets([fname_task])

curr_dataset = Dataset(
    "../../rep_learning_data/rgb/aldine/rgb",
    "../../rep_learning_data/normal",
    fname_task
)
training_generator = torch.utils.data.DataLoader(curr_dataset, **params)

batch_id = 0
for img, label in training_generator:
    print(batch_id)
    m = torch.nn.Softmax(dim=1)
    output_decoder = m(decoder_nets[0](img))
    output_numpy = output_decoder.data.numpy()
    
    imageio.imwrite("output_img.png", output_numpy[0].transpose((1,2,0)))
    imageio.imwrite("label_img.png", label[0].numpy().transpose((1,2,0)))
    
    batch_id += 1"""