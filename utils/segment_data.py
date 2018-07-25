import os
import numpy as np
from random import shuffle


def generate_kfold_data(positive_path_file, negative_path_file, pos_label, neg_label,
                        disease_name, save_dir, seg_ratio, multi):
    '''Segment the data and write them to local. Test data is set aside in advance. 
    Training data and validation data are splitted in k ways (k-fold).
    
    Args:
        positive_path_file: the filename of which file containing positive image paths
        negative_path_file: the filename of which file containing negative image paths
        pos_label: the filename of which file containing positive labels
        neg_label: the filename of which file containing negative labels
        disease_name: disease name
        save_dir: the direction for segmented data
        seg_ratio: the ratio of segmentation, should be like [0.8, 0.1, 0.1], corresponding to 
                [train, val, test]
    Returns:
        k: equal to int(sum(seg_ratio[:2])/seg_ratio[1]) # k-fold
    '''
    # Create label and write the shuffled data
    def write_data(pos_path, neg_path, save_name, multi):
        pos_path = np.concatenate(([pos_path]*multi))
        path = np.concatenate((pos_path, neg_path))
        label = np.concatenate((pos_label * len(pos_path), neg_label * len(neg_path)))
        data = zip(path, label)
        shuffle(data)
        path, label = zip(*data) # unzip
        with open(os.path.join(save_dir,'ccyy_%s_%s_path.txt'%(disease_name,save_name)),'w') as f:
            for p in path:
                f.write(p+'\n')
        np.save(os.path.join(save_dir,'ccyy_%s_%s_label.npy'%(disease_name,save_name)),label)
    
    seg_ratio = np.array(seg_ratio)/sum(seg_ratio) # normalization
    k = int(sum(seg_ratio)/seg_ratio[2]) # k-fold
    
    with open(positive_path_file) as f:
        positive_path = f.read().splitlines()
    with open(negative_path_file) as f:
        negative_path = f.read().splitlines()
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
     
    shuffle(positive_path)
    shuffle(negative_path)
    # Generate data
    test_pos_len = len(positive_path)/k # test len
    test_neg_len = len(negative_path)/k
    val_pos_len = int(len(positive_path)*seg_ratio[1])
    val_neg_len = int(len(negative_path)*seg_ratio[1])
    for i in range(k):
        curr_test_pos_path = positive_path[i*test_pos_len:(i+1)*test_pos_len]
        curr_test_neg_path = negative_path[i*test_neg_len:(i+1)*test_neg_len]        
        if i!=k-1:
            curr_val_pos_path = positive_path[(i+1)*test_pos_len:(i+1)*test_pos_len+val_pos_len]
            curr_val_neg_path = negative_path[(i+1)*test_neg_len:(i+1)*test_neg_len+val_neg_len]
            curr_train_pos_path = np.concatenate((positive_path[:i*test_pos_len],positive_path[(i+1)*test_pos_len+val_pos_len:]))
            curr_train_neg_path = np.concatenate((negative_path[:i*test_neg_len],negative_path[(i+1)*test_neg_len+val_neg_len:]))
        else:
            curr_val_pos_path = positive_path[:val_pos_len]
            curr_val_neg_path = negative_path[:val_neg_len]
            curr_train_pos_path = positive_path[val_pos_len:-test_pos_len]
            curr_train_neg_path = negative_path[val_neg_len:-test_neg_len]        
        write_data(curr_train_pos_path, curr_train_neg_path, 'train_'+str(i),multi)
        write_data(curr_val_pos_path, curr_val_neg_path, 'val_'+str(i),1)
        write_data(curr_test_pos_path, curr_test_neg_path, 'test_'+str(i),1)
        
    return k

def seg_sanity_check(k, path_pattern, label_pattern):
    for i in range(k):
        for phase in ['train', 'val', 'test']:
            try:
                with open(path_pattern%(phase+'_'+str(i))) as f:
                    img_len = len(f.readlines())  
                label_len = len(np.load(label_pattern%(phase+'_'+str(i))))
                if img_len != label_len:
                    print i, 'Wrong segmentation in', phase, 'phase.'
            except(IOError):
                if i == 0 and phase == 'train':
                    print 'Path does not exist.'

        