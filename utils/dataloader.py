import os
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import pickle
import numpy as np
from PIL import Image

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def find_classes(dir):
    # Classes are subdirectories of the root directory
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

class TissueData(data.Dataset):
    def __init__(self, root, dset_type, data='lung', transform=None, metadata=False):

        classes, class_to_idx = find_classes(root)

        self.data = data

        #json_dict_path = ''
        #with open(json_dict_path, 'rb') as f:
        #    self.json = pickle.load(f)

        self.root = root
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.metadata = metadata
        self.datapoints, self.filenames = self.make_dataset(root, dset_type, class_to_idx)
        self.transform = transform

    def parse_json(self, fname):
        json = self.json[fname]  
 
        if self.data == 'breast':
            return []
        elif self.data == 'kidney':
            return []
        else:
            age, cigarettes, gender = json['age_at_diagnosis'], json['cigarettes_per_day'], json['gender']
            return [age, cigarettes, gender]

    def make_dataset(self, dir, dset_type, class_to_idx):
        datapoints = []
        filenames = []

        dir = os.path.expanduser(dir)
        
        for target in os.listdir(dir):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            print('Loading from:', target)
            dd=[]
            for root, _, fnames in os.walk(d):
                for fname in fnames:
                    # Parse the filename
                    dataset_type = fname.strip('.jpeg').split('_')[0]
                    y = fname.strip('.jpeg').split('_')[-1]
                    x = fname.strip('.jpeg').split('_')[-2]
                    raw_file = '_'.join(fname.strip('.jpeg').split('_')[1:-2])
	            #dataset_type, raw_file, x, y = fname.strip('.jpeg').split('_')

                    raw_file_name = dset_type + '_' + raw_file
                    original_file = raw_file + '.svs'
        
                    # Only add it if it's the correct dset_type (train, valid, test)
                    if fname.endswith(".jpeg") and dataset_type == dset_type:
                        path = os.path.join(root, fname)

                        if self.metadata:
                            item = (path, self.parse_json(original_file) + [int(x), int(y)], class_to_idx[target])
                        else:
                            item = (path, class_to_idx[target])
                        
                        datapoints.append(item)
                        dd.append(item)
                        
                        if raw_file_name not in filenames:
                            filenames.append(raw_file_name)
                print('number of samples:', len(dd))

        return datapoints, filenames

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (img + concatenated extra info, label) for the given index
        """

        if self.metadata:
            filepath, info, label = self.datapoints[index]
        else:
            filepath, label = self.datapoints[index]

        # Load image from filepath
        img = pil_loader(filepath)
        
        if self.transform is not None:
            img = self.transform(img)

        if self.metadata:
            # Reshape extra info, then concatenate to image as extra channels
            info = np.array(info)
            info_length = len(info)
            height, width = img.size(1), img.size(2)
            reshaped = torch.FloatTensor(np.repeat(info, height*width).reshape((len(info), height, width)))
            output = torch.cat((img, reshaped), 0)
        else:
            output = img

        return output, label

    def __len__(self):
        return len(self.datapoints)
