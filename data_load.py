import torch
from torch.utils.data.dataloader import DataLoader
import os

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, imgs_path, labels_path):
        'Initialization'
        self.labels = os.listdir(labels_path)
        self.images= os.listdir(imgs_path)

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        current_img = self.images[index]
        current_label = self.labels[index]

        # Load data and get label
        img = torch.load(current_img)
        label = torch.load(current_label)

        return img, label

#print(os.listdir(''))
curr_dataset = Dataset("rep_learning_data/rgb/aldine/rgb","rep_learning_data/aldine_class_object/class_object")
training_generator = torch.utils.data.DataLoader(training_set, **params)

params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}