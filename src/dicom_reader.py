import sys
from glob import glob
from utils.preprocessing.preprocess_utils import mask_extractor, mask_processing, load_CT
from multiprocessing import Pool, cpu_count
import os
import time
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
from utils.manifest import *
import sys

def produce_box(path, extendbox, spacing, true_spacing, origin):
    

    delt_z = extendbox[0, 0]
    delt_y = extendbox[1, 0]
    delt_x = extendbox[2, 0]

    # delta z, y, x, spacing z, y, x, true_spacing z, y, x
    delt_zyx = [delt_z, delt_y, delt_x,#delta z y x
                spacing[0], spacing[1], spacing[2],# spacing z y x
                true_spacing[0], true_spacing[1], true_spacing[2],# true spacing z y x
               origin[0], origin[1], origin[2]] #origiin z, y, x

    
    return delt_zyx

def full_process(path, target_path, cache):
    
    img, spacing, origin, uid = load_CT(path)
    
    if spacing[0] < 4.5:
        target_csv = 'preprocess3D.csv'
        prep_dir = os.path.join(target_path, '3D')
    else:
        target_csv = 'preprocess2D.csv'
        prep_dir = os.path.join(target_path, '2D')
    
    if uid not in cache:
        img, mask = mask_extractor(img, uid) # name is seriesuid

        slice_im, extendbox, true_spacing = mask_processing(img, mask, spacing)

        #determine which folder to save according to z spacing/slicethickness
        delt_zyx = produce_box(path, extendbox, spacing, true_spacing, origin)

        
        np.save(os.path.join(prep_dir, uid + '_clean'), slice_im)
        np.save(os.path.join(prep_dir, uid + '_delta_zyx'), delt_zyx)
        # save the data and labels


        '''
            save the delta zyx information to calculate back to the original coordinate
             #np.save(os.path.join(prep_folder, uid + '_label'), label)
            no label in test phase
        '''

    with open(os.path.join(target_path, target_csv), 'a') as f:
        df = pd.DataFrame(data = np.column_stack([np.array(path), np.array(uid)]))
        df.to_csv(f, header = False, index = False)
        


def parse_CT(source_path, advanced = False):
    paths = []
     # if a directory is given
    if os.path.isdir(source_path):
        for root, dirs, files in os.walk(source_path):
            
            # search for all dicom file and mhd file
            if advanced:
                dicom_num = 0
                mhd_nii_list = []
                for file_name in files:
                    if '.dcm' in file_name:
                        dicom_num += 1
                    elif '.mhd' in file_name or '.nii' in file_name:
                        mhd_nii_list.append(os.path.join(root, file_name))
                        print 'found ct: ', os.path.join(root, file_name)
                if dicom_num > 50:
                    paths.append(root)
                    print 'found ct: ', root
                paths += mhd_nii_list
            else:
                for file_name in files:
                    if '.dcm' in file_name:
                        if len(files) > 50:
                            paths.append(root)
                            print 'found ct: ', root
                        break
                    if '.mhd' in file_name:
                        paths.append(os.path.join(root, file_name))
                        print 'found ct: ', os.path.join(root, file_name)

    # or if a file is given
    elif os.path.isfile(source_path):
        if source_path[-4:] == '.txt':
            temp = []
            try:
                with open(source_path, 'r') as f:
                    temp = f.read().splitlines()
            except:
                raise Exception('Can not open ' + source_path + ' !!!')
            #parse each line in the file
            for t in temp:
                paths += parse_CT(t)
            
        elif source_path[-4:] == '.mhd' or source_path[-4:] == '.nii':
            paths.append(source_path)
            print 'found ct: ', source_path
        '''
        else:
            print 'Unidentified file format: ' + source_path
        '''
    
    # neither a file nor a directory
    '''
    else:
        print source_path + ' is neither a txt file nor a directory!!!'
    '''
    return paths

def read_in(source_path, target_path, is_cache=False):

    print_title('Reading CTs ...')
    
    '''
        creat preprocess directory
    '''
    if not os.path.exists(os.path.join(target_path, '2D')):
        os.makedirs(os.path.join(target_path, '2D'))
    if not os.path.exists(os.path.join(target_path, '3D')):
        os.makedirs(os.path.join(target_path, '3D'))
        
    
    with open(os.path.join(target_path, 'preprocess2D.csv'), 'w') as f:
        df = pd.DataFrame(columns=['path', 'uid'])
        df.to_csv(f, index = False)
            
    with open(os.path.join(target_path, 'preprocess3D.csv'), 'w') as f:
        df = pd.DataFrame(columns=['path', 'uid'])
        df.to_csv(f, index = False)
    '''
        parse CT paths from given dir or txt file
    '''
    paths = parse_CT(source_path)
    
    print_tip( '%d CTs found ...' % len(paths))
    
    bad_cases = []
    
    print_title('Loading Cache ...')

    cache = []
    if is_cache:
        for root, dirs, files in os.walk(target_path):
            for file_name in files:
                if '_clean.npy' in file_name:
                    cache.append(file_name.split('_clean.npy')[0])
    

    print_tip('%d cached CTs found ... '%len(cache))
    '''
        start preprocessing CTs...
    '''
    
    total = len(paths)
    print_title('preprocessing ...')

    for idx, path in enumerate(paths):
        try:
            full_process(path, target_path, cache)
        except Exception, err:
            print err
            bad_cases.append(path)
        print_percentage('preprocess', idx + 1, total)


    if bad_cases:
        with open("./bad_cases.txt", "w") as f:
            for bad_case in bad_cases:
                f.write(bad_case+"\n")
    
    print_tip('preprocessing done!')



            

