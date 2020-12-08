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
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0), -1)
            else:
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
    cv2.imwrite("here.png", cam*255)
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    
    return cam


class OldGradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda, gpu):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        self.gpu = gpu
        if self.cuda:
            self.model = model.to(self.gpu)

        self.extractor = ModelOutputs(
            self.model, self.feature_module, target_layer_names
        )

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.to(self.gpu))
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.to(self.gpu) * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()

        weights = np.mean(grads_val, axis=(2, 3))

        cam = np.zeros(
            (target.shape[0], target.shape[2], target.shape[3]), dtype=np.float32
        )

        for i in range(weights.shape[1]):
            weight = weights[:, i][:, None, None]
            cam += weight * target[:, i, :, :]

        cam = np.mean(cam, axis=0)
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input.shape[2], input.shape[3]))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        cam_avg = np.zeros((target.shape[2], target.shape[3]), dtype=np.float32)
        target_avg = np.mean(target, axis=0)
        weight_avg = np.mean(weights, axis=0)
        for i, w in enumerate(weight_avg):
            cam_avg += w * target_avg[i, :, :]

        cam_avg = np.maximum(cam_avg, 0)
        cam_avg = cv2.resize(cam_avg, (input.shape[2], input.shape[3]))
        cam_avg = cam_avg - np.min(cam_avg)
        cam_avg = cam_avg / np.max(cam_avg)

        return cam, cam_avg

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

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
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


if __name__ == "__main__":
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision
    model = models.resnet50(pretrained=True)
    grad_cam = GradCam(
        model=model,
        feature_module=model.layer4,
        target_layer_names=["2"],
        use_cuda=True,
    )

    (w, h) = 224, 224
    #pdb.set_trace()
    current_img_path = "../rep_learning_data/rgb/aldine/rgb/point_0_view_0_domain_rgb.png"
    current_class_path = "../rep_learning_data/aldine_class_object/class_object/point_0_view_0_domain_class_object.npy"

    theclasses = np.load(current_class_path)
    target_index = np.argmax(theclasses)
    
    theclass = np.load(current_class_path)
    
    original_img = read_image_resize(current_img_path, (w, h))
    img = np.expand_dims(original_img, axis=0)
    original_img = torch.tensor(convert_whc_to_cwh(img))
    the_input = normalize(original_img).requires_grad_(True)
    img_mult = torch.cat(5 * [the_input], axis=0)

    target_index = None
    mask = grad_cam(img_mult, target_index)

    show_cam_on_image(img[0], mask)