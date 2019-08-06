import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.init as init
from torch.autograd import Variable
import argparse
import numpy as np
from PIL import Image
from utils.dataloader import *
#from utils.auc_test import *
from utils.auc import *
from utils import new_transforms
import time
import torchvision.models as models

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', default='', help="name of experiment to test")
parser.add_argument('--model', default='', help="name of model to test")
parser.add_argument('--root_dir', type=str, default='<ROOT_PATH><CANCER_TYPE>TilesSorted/', help='Data directory .../dataTilesSorted/')
parser.add_argument('--num_class', type=int, default=2, help='number of classes ')
parser.add_argument('--tile_dict_path', type=str, default='"<ROOT_PATH><CANCER_TYPE>_FileMappingDict.p', help='Tile dictinory path')
parser.add_argument('--val', type=str, default='test', help='validation set')
parser.add_argument('--imgSize', type=int, default=299, help='the height / width of the image to network')
parser.add_argument('--model_type',type=str,  default='PathCNN', help='choose the model to train with: PathCNN, alexnet,vgg16')

opt = parser.parse_args()

root_dir = str(opt.root_dir)
num_classes = int(opt.num_class)
tile_dict_path = str(opt.tile_dict_path)
test_val = str(opt.val)
imgSize = int(opt.imgSize)

transform = transforms.Compose([new_transforms.Resize((imgSize,imgSize)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_data = TissueData(root_dir, test_val, transform = transform, metadata=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

classes = test_data.classes
class_to_idx = test_data.class_to_idx

print('Class encoding:')
print(class_to_idx)

def get_tile_probability(tile_path):

    """
    Returns an array of probabilities for each class given a tile
    @param tile_path: Filepath to the tile
    @return: A ndarray of class probabilities for that tile
    """

    # Some tiles are empty with no path, return nan
    if tile_path == '':
        return np.full(num_classes, np.nan)

    tile_path = root_dir + tile_path

    with open(tile_path, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('RGB')

    # Model expects a 4D tensor, unsqueeze first dimension
    img = transform(img).unsqueeze(0)
    img = img.cuda()

    # Turn output into probabilities with softmax
    var_img = Variable(img, volatile=True)
    output = F.softmax(model(var_img)).data.squeeze(0)

    return output.cpu().numpy()

with open(tile_dict_path, 'rb') as f:
    tile_dict = pickle.load(f)

def aggregate(file_list, method):

    """
    Given a list of files, return scores for each class according to the
    method and labels for those files.
    @param file_list: A list of file paths to do predictions on
    @param method: 'average' - returns the average probability score across
                               all tiles for that file
                   'max' - predicts each tile to be the class of the maximum
                           score, and returns the proportion of tiles for
                           each class
    @return: a ndarray of class probabilities for all files in the list
             a ndarray of the labels
    """

    model.eval()
    predictions = []
    true_labels = []

    for file in file_list:
        tile_paths, label = tile_dict[file]

        folder = classes[label]

        def add_folder(tile_path):
            if tile_path == '':
                return ''
            else:
                return folder + '/' + tile_path

        # Add the folder for the class name in front
        add_folder_v = np.vectorize(add_folder)
        tile_paths = add_folder_v(tile_paths)

        # Get the probability array for the file
        prob_v = np.vectorize(get_tile_probability, otypes=[np.ndarray])
        probabilities = prob_v(tile_paths)


        """
        imgSize = probabilities.shape()
        newShape = (imgSize[0], imgSize[1], 3)
        probabilities = np.reshape(np.stack(probabilities.flat), newShape)
        """

        if method == 'average':
            probabilities = np.stack(probabilities.flat)
            prediction = np.nanmean(probabilities, axis = 0)

        elif method == 'max':
            probabilities = np.stack(probabilities.flat)
            probabilities = probabilities[~np.isnan(probabilities).all(axis=1)]
            votes = np.nanargmax(probabilities, axis=1)
            
            out = np.array([sum(votes == i) for i in range(num_classes)])
            prediction = out / out.sum()

        else:
            raise ValueError('Method not valid')

        predictions.append(prediction)
        true_labels.append(label)

    return np.array(predictions), np.array(true_labels)

# Create model objects
if(opt.model_type == 'alexnet'):
    model = models.alexnet(num_classes=3)
elif(opt.model_type == 'vgg16'):
    model = models.vgg16(num_classes=3)

model.cuda()

model_path = opt.experiment + '/checkpoints/' + opt.model
state_dict = torch.load(model_path)
model.load_state_dict(state_dict)


predictions, labels = aggregate(test_data.filenames, method='max')

data = np.column_stack((np.asarray(predictions),np.asarray(labels)))
data.dump(open('{0}/outputs/{1}_pred_label_max_{2}.npy'.format(opt.experiment,opt.val,opt.model), 'wb'))

#This can be used if need to print the auc and save the roc curve automatically

roc_auc  = get_auc('{0}/images/{1}_AUC_max_{2}.jpg'.format(opt.experiment,opt.val,opt.model),
                   predictions, labels, classes = range(num_classes))
print('Max method:')
print(roc_auc)




print('Starting aggregate')
start = time.time()
local_time = time.ctime(start)
print(local_time)

predictions, labels = aggregate(test_data.filenames, method='average')

print('end of aggregate')
start = time.time()
local_time = time.ctime(start)
print(local_time)



data = np.column_stack((np.asarray(predictions),np.asarray(labels)))
data.dump(open('{0}/outputs/{1}_pred_label_avg_{2}.npy'.format(opt.experiment,opt.val,opt.model), 'wb'))

#This can be used if need to print the auc and save the roc curve automatically
roc_auc  = get_auc('{0}/images/{1}_AUC_avg_{2}.jpg'.format(opt.experiment,opt.val,opt.model),
                   predictions, labels, classes = range(num_classes))
print('Average method:')
print(roc_auc)
