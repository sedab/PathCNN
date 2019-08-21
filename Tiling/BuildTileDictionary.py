import re 
import numpy as np
import os
import pickle
import argparse

'''
Author: @edufierro

Capstone project

Purpose: Get dictionary with files:[2Darray tiles, type of cancer]
'''

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='Lung', help='Data to train on (Lung/Breast/Kidney)')
parser.add_argument('--file_path', type=str, default='/beegfs/jmw784/Capstone/', help='Root path where the tiles are')
parser.add_argument('--train_log', type=str, default='', help='path to the training output')

opt = parser.parse_args()

root_dir = opt.file_path + opt.data + "TilesSorted/"
out_file = opt.file_path + opt.data + "_FileMappingDict.p"
train_log = opt.train_log

def find_classes(dir):
    # Classes are subdirectories of the root directory
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def get_class_coding(lf):
    auc_new = []
    phrase = "Class encoding:"

    with open(lf, 'r+') as f:
        lines = f.readlines()
        for i in range(0, len(lines)):
            line = lines[i]
            #print(line)
            if phrase in line:
                class_encoding = lines[i + 1] # you may want to check that i < len(lines)
                break
                
    class_encoding = class_encoding.strip('\n').strip('{').strip('}')
    #print(class_encoding)
            
    class_names = []
    class_codes = []

    for c in class_encoding.split(','):
        #print(c)
        class_names.append(c.split(':')[0].replace("'", "").replace(" ", ""))#.split('-')[-1])
        class_codes.append(int(c.split(':')[1]))
    

    class_coding = {}
    for i in range(len(class_names)):
        class_coding[class_codes[i]] = class_names[i]
    
    class_codes.sort()
    return class_names, class_codes, class_coding


def getCoords(tile_list): 
    
    '''
    Given a list of tiles, with format: 
    [test, valid, train]_NAME_x_y.jpeg
    Returns a two list of same size with xcoords and y coords    
    '''
    
    xcoords = [re.split("_", i)[-2] for i in tile_list]
    xcoords = list(map(int, xcoords))
    ycoords = [re.split("_", i)[-1] for i in tile_list]
    ycoords = [re.sub(".jpeg", "", i) for i in ycoords]
    ycoords = list(map(int, ycoords))
    
    return xcoords, ycoords

def fileCleaner(tile_list): 
    
    '''
    Given a list of tiles, remove coords ("_X_Y_") and ".jpeg" termination
    '''
    
    tile_list = [re.sub("_[0-9]*_[0-9]*.jpeg", "", x) for x in tile_list]
    
    return (tile_list)

def get2Darray(xcoords, ycoords, tiles_input): 
    
    '''
    Given a list of xcoords, ycoords and files, returns a 2D array where each file
       correspond to the pair of coords
    '''
    
    xmax = max(xcoords) + 1
    ymax = max(ycoords) + 1
    tiles_output = np.empty((ymax, xmax), dtype=np.dtype((str, 100)))
    for i in range(0,len(xcoords)): 
        tiles_output[ycoords[i], xcoords[i]] = tiles_input[i]
        
    return tiles_output

def fastdump(obj, file):
    p = pickle.Pickler(file)
    p.fast = True
    p.dump(obj)

def main():

    if os.path.exists(out_file):
        response = None

        while response not in ['y', 'n']:
            response = input('Tile dictionary already exists, do you want to continue (y/n)? ')

        if response == 'n':
            quit()

#if classes provided for testing are less than for training        
    if(train_log!=''):
        c_names, c_codes, c_coding = get_class_coding(train_log)
        c_coding_invert = {v: k for k, v in c_coding.items()}
        classes, _ = find_classes(root_dir)
        class_to_idx = {}
        for n in classes:
            class_to_idx[n] = c_coding_invert[n]  
    else:
        classes, class_to_idx = find_classes(root_dir)
    
    
    print(class_to_idx)
    
    tile_files = {}
    original_files = {}
    main_dict = {}

    print("Importing File Names...")

    for c in classes:
        tile_files[c] = os.listdir(root_dir + c)
        original_files[c] = fileCleaner(tile_files[c])

        print("Updating dict for %s files ..." % (c))

        for file in set(original_files[c]):
        
            index_list = [i for i, x in enumerate(original_files[c]) if x==file]
            tiles = [tile_files[c][i] for i in index_list]
            xs, ys = getCoords(tiles)
            tiles_array = get2Darray(xs, ys, tiles)
            loop_dict = {file:[tiles_array, class_to_idx[c]]}
            main_dict.update(loop_dict)

        # Prevent running out of memory
        del tiles, xs, ys, tiles_array, loop_dict, tile_files[c], original_files[c]
    
    fastdump(main_dict, open(out_file, "wb" ) ) 
    print("Dictionary Ready!!! Saved as pickle in: \n {0}".format(out_file))
    
    return main_dict

if __name__ == '__main__':
    main()
