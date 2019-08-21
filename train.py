#from comet_ml import Experiment

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.init as init
from torch.autograd import Variable

import os
import time
import numpy as np
from PIL import Image
from utils.dataloader import *
#use AUC for AUC and CI, auc2 for precision, AUC and CI, auc3 precision auc and CI
from utils.auc import *
from utils import new_transforms
import argparse
import random

"""
Options for training
"""

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--imgSize', type=int, default=299, help='the height / width of the image to network')
parser.add_argument('--nc', type=int, default=3, help='input image channels (+ concatenated info channels if metadata = True)')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
parser.add_argument('--dropout', type=float, default=0.5, help='probability of dropout, default=0.5')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam, default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--model', default='', help="path to model (to continue training)")
parser.add_argument('--experiment', default=None, help='where to store samples and models')
parser.add_argument('--augment', action='store_true', help='whether to use data augmentation or not')
parser.add_argument('--optimizer',type=str, default='Adam',  help='optimizer: Adam, SGD or RMSprop; default: Adam')
parser.add_argument('--metadata', action='store_true', help='whether to use metadata (default is not)')
parser.add_argument('--init', type=str, default='normal', help='initialization method (normal, xavier, kaiming)')
parser.add_argument('--nonlinearity', type=str, default='relu', help='nonlinearity to use (selu, prelu, leaky, relu)')
parser.add_argument('--earlystop', action='store_true', help='trigger early stopping (boolean)')
parser.add_argument('--method', type=str, default='average', help='aggregation prediction method (max, average)')
parser.add_argument('--decay_lr', action='store_true', help='activate decay learning rate function')
parser.add_argument('--root_dir', type=str, default='<ROOT_PATH><CANCER_TYPE>TilesSorted/', help='Data directory .../dataTilesSorted/')
parser.add_argument('--num_class', type=int, default=2, help='number of classes ')
parser.add_argument('--tile_dict_path', type=str, default='"<ROOT_PATH><CANCER_TYPE>_FileMappingDict.p', help='Tile dictinory path')
parser.add_argument('--step_freq', type=int, default=100000000, help='save the checkpoint at every step_freq steps')
parser.add_argument('--calc_val_auc', action='store_true', help='trigger validation auc calculatio at each epoch (boolean)')

opt = parser.parse_args()

print(opt)

ngpu = int(opt.ngpu)
nc = int(opt.nc)
imgSize = int(opt.imgSize)
root_dir = str(opt.root_dir)
num_classes = int(opt.num_class)
tile_dict_path = str(opt.tile_dict_path)
step_freq = int(opt.step_freq)

"""
Save experiment 
"""

if opt.experiment is None:
    opt.experiment = 'samples'

os.system('mkdir {0}'.format(opt.experiment))
os.system('mkdir {0}/images'.format(opt.experiment))
os.system('mkdir {0}/checkpoints'.format(opt.experiment))
os.system('mkdir {0}/outputs'.format(opt.experiment))
opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

###############################################################################

"""
Load data
"""


# Random data augmentation
augment = transforms.Compose([new_transforms.Resize((imgSize, imgSize)),
                              transforms.RandomHorizontalFlip(),
                              new_transforms.RandomRotate(),
                              new_transforms.ColorJitter(0.25, 0.25, 0.25, 0.05),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform = transforms.Compose([new_transforms.Resize((imgSize,imgSize)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

data = {}
loaders = {}

for dset_type in ['train', 'valid']:
    if dset_type == 'train' and opt.augment:
        data[dset_type] = TissueData(root_dir, dset_type, train_log='', transform = augment, metadata=opt.metadata)
    else:
        data[dset_type] = TissueData(root_dir, dset_type, train_log='', transform = transform, metadata=opt.metadata)

    loaders[dset_type] = torch.utils.data.DataLoader(data[dset_type], batch_size=opt.batchSize, shuffle=True, num_workers=8)
    print('Finished loading %s dataset: %s samples' % (dset_type, len(data[dset_type])))

class_to_idx = data['train'].class_to_idx
classes = data['train'].classes

print('Class encoding:')
print(class_to_idx)

###############################################################################

"""
Model initialization and definition
"""

# Custom weights initialization
if opt.init not in ['normal', 'xavier', 'kaiming']:
    print('Initialization method not found, defaulting to normal')

def init_model(model):
    for m in model.modules():
        if isinstance(m,nn.Conv2d):
            if opt.init == 'xavier':
                m.weight.data = init.xavier_normal(m.weight.data)
            elif opt.init == 'kaiming':
                m.weight.data = init.kaiming_normal(m.weight.data)
            else:
                m.weight.data.normal_(-0.1, 0.1)
            
        elif isinstance(m,nn.BatchNorm2d):
            m.weight.data.normal_(-0.1, 0.1)

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, pool, **kwargs):
        super(BasicConv2d, self).__init__()

        self.pool = pool
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

        if opt.nonlinearity == 'selu':
            self.relu = nn.SELU()
        elif opt.nonlinearity == 'prelu':
            self.relu = nn.PReLU()
        elif opt.nonlinearity == 'leaky':
            self.relu = nn.LeakyReLU()
        else:
            self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=opt.dropout)

    def forward(self, x):
        x = self.conv(x)

        if self.pool:
            x = F.max_pool2d(x, 2)
        
        x = self.relu(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x

# Define model
class cancer_CNN(nn.Module):
    def __init__(self, nc, imgSize, ngpu):
        super(cancer_CNN, self).__init__()
        self.nc = nc
        self.imgSize = imgSize
        self.ngpu = ngpu
        #self.data = opt.data
        self.conv1 = BasicConv2d(nc, 16, False, kernel_size=5, padding=1, stride=2, bias=True)
        self.conv2 = BasicConv2d(16, 32, False, kernel_size=3, bias=True)
        self.conv3 = BasicConv2d(32, 64, True, kernel_size=3, padding=1, bias=True)
        self.conv4 = BasicConv2d(64, 64, True, kernel_size=3, padding=1, bias=True)
        self.conv5 = BasicConv2d(64, 128, True, kernel_size=3, padding=1, bias=True)
        self.conv6 = BasicConv2d(128, 64, True, kernel_size=3, padding=1, bias=True)
        self.linear = nn.Linear(5184, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

###############################################################################

# Create model objects
model = cancer_CNN(nc, imgSize, ngpu)
init_model(model)
model.train()

criterion = nn.CrossEntropyLoss()

# Load checkpoint models if needed
if opt.model != '': 
    model.load_state_dict(torch.load(opt.model))
print(model)

if opt.cuda:
    model.cuda()

# Set up optimizer
if opt.optimizer == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
elif opt.optimizer == "RMSprop":
    optimizer = optim.RMSprop(model.parameters(), lr = opt.lr)
elif opt.optimizer == "SGD": 
    optimizer = optim.SGD(model.parameters(), lr = opt.lr)
else: 
    raise ValueError('Optimizer not found. Accepted "Adam", "SGD" or "RMSprop"')

###############################################################################

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

    if opt.cuda:
        img = img.cuda()

    # Turn output into probabilities with softmax
    var_img = Variable(img, volatile=True)
    output = F.softmax(model(var_img)).data.squeeze(0)

    return output.cpu().numpy()

# Load tile dictionary

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

###############################################################################

def early_stop(val_history, t=3, required_progress=0.0001):

    """
    Stop the training if there is no non-trivial progress in k steps
    @param val_history: a list contains all the historical validation auc
    @param required_progress: the next auc should be higher than the previous by 
        at least required_progress amount to be non-trivial
    @param t: number of training steps 
    @return: a boolean indicates if the model should early stop
    """
    
    if (len(val_history) > t+1):
        differences = []
        for x in range(1, t+1):
            differences.append(val_history[-x]-val_history[-(x+1)])
        differences = [y < required_progress for y in differences]
        if sum(differences) == t: 
            return True
        else:
            return False
    else:
        return False

if opt.earlystop:
    validation_history = []
else:
    print("No early stopping implemented")
    
stop_training = False

###############################################################################

def adjust_learning_rate(optimizer, epoch):

    """Sets the learning rate to the initial LR decayed by 10 every 3 epochs
        Function copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py"""
    
    lr = opt.lr * (0.1 ** (epoch // 3)) # Original
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

###############################################################################

"""
Training loop
"""

best_AUC = 0.0

print('Starting training')
start = time.time()
local_time = time.ctime(start)
print(local_time)

print(time.time())
for epoch in range(1,opt.niter+1):
    data_iter = iter(loaders['train'])
    i = 0
    
    if opt.decay_lr:
        adjust_learning_rate(optimizer, epoch)
        print("Epoch %d :lr = %f" % (epoch, optimizer.state_dict()['param_groups'][0]['lr']))    
    while i < len(loaders['train']):
        model.train()
        img, label = data_iter.next()
        i += 1

        # Drop the last batch if it's not the same size as the batchsize
        if img.size(0) != opt.batchSize:
            break

        if opt.cuda:
            img = img.cuda()
            label = label.cuda()


        input_img = Variable(img)
        target_label = Variable(label)

        train_loss = criterion(model(input_img), target_label)
        #print(model(input_img)[0])
        # Zero gradients then backward pass
        optimizer.zero_grad()
        train_loss.backward()
      
        correc=0
        total=0

        optimizer.step()
        
        print('[%d/%d][%d/%d] Training Loss: %f'
               % (epoch, opt.niter, i, len(loaders['train']), train_loss.item()))
        ii=i+((epoch)*len(loaders['train']))
        #get validation AUC every step_freq 
        if ii % step_freq == 0:
            val_predictions, val_labels = aggregate(data['valid'].filenames, method=opt.method)

            data_ = np.column_stack((data['valid'].filenames,np.asarray(val_predictions),np.asarray(val_labels)))
            data_.dump(open('{0}/outputs/val_pred_label_avg_step_{1}.npy'.format(opt.experiment,str(ii)), 'wb'))
            torch.save(model.state_dict(), '{0}/checkpoints/step_{1}.pth'.format(opt.experiment, str(ii)))           
            print('validation scores:')

            roc_auc = get_auc('{0}/images/val_roc_step_{1}.jpg'.format(opt.experiment,epoch), val_predictions, val_labels, classes = range(num_classes))
            for k, v in roc_auc.items(): 
                if k in range(num_classes):
                    k = classes[k] 
                #experiment.log_metric("{0} AUC".format(k), v)
                print('%s AUC: %0.4f' % (k, v))

    #save the checkpoint at every epoch
    torch.save(model.state_dict(), '{0}/checkpoints/epoch_{1}.pth'.format(opt.experiment, str(epoch)))

    #print(time.time())
    # Get validation AUC once per epoch
    if opt.calc_val_auc:
        val_predictions, val_labels = aggregate(data['valid'].filenames, method=opt.method)
        data_ = np.column_stack((data['valid'].filenames,np.asarray(val_predictions),np.asarray(val_labels)))
        data_.dump(open('{0}/outputs/val_pred_label_avg_epoch_{1}.npy'.format(opt.experiment,str(epoch)), 'wb'))

        roc_auc = get_auc('{0}/images/val_roc_epoch_{1}.jpg'.format(opt.experiment, epoch),val_predictions, val_labels, classes = range(num_classes))

        for k, v in roc_auc.items():
            if k in range(num_classes):
                k = classes[k]

            #experiment.log_metric("{0} AUC".format(k), v)
            print('%s AUC: %0.4f' % (k, v))

    # Stop training if no progress on AUC is being made
    if opt.earlystop:
        validation_history.append(roc_auc['macro'])
        stop_training = early_stop(validation_history)

        if stop_training: 
            print("Early stop triggered")
            break


    epoch_time = time.time()
    local_time = time.ctime(epoch_time)
    print(local_time)

# Final evaluation
print('Finished training, best AUC: %0.4f' % (best_AUC))
end = time.time()
print(end-start)
