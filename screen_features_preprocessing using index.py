# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 12:41:23 2019

@author: user
"""

import os
import time
import numpy as np

from pysc2.lib import actions
from pysc2.lib import features

PROJECT_DIR = os.path.abspath('D:\\1.Project\\2019.04_Game AI\\parsed_0708\\')
REPLAY_DIR = os.path.join(PROJECT_DIR, 'sample\\')
RESULT_DIR = os.path.join(REPLAY_DIR, 'Parsed_Result_0708\\')

assert all([os.path.isdir(PROJECT_DIR), os.path.isdir(REPLAY_DIR)])

def load_features(replay_dir, feature_type='screen'):
    """Load parsed features from .npz file format."""
    if feature_type == 'screen':
        filepath = os.path.join(replay_dir, 'ScreenFeatures.npz')
    elif feature_type == 'minimap':
        filepath = os.path.join(replay_dir, 'MinimapFeatures.npz')
    elif feature_type == 'flat':
        filepath = os.path.join(replay_dir, 'FlatFeatures.npz')
        raise NotImplementedError
    else:
        raise ValueError
    
    with np.load(filepath) as fp:
        name2feature = {k: v for k, v in fp.items()}
    
    return name2feature

def human_readable_size(size, precision=2):
    suffixes = ['B','KB','MB','GB','TB']
    suffix_idx = 0
    while size > 1024 and suffix_idx < 4:
        suffix_idx += 1     # increment the index of the suffix
        size = size / 1024.0  # apply the division
    return "%.*f%s" % (precision, size, suffixes[suffix_idx])

# Load screen features
screen_specs = features.SCREEN_FEATURES._asdict()
screen_features = load_features(replay_dir=REPLAY_DIR, feature_type='screen')  # dict

# start = time.time()

sc_feats_one_hot = {}
print('[Screen features]')
print('=' * 105)
for i, (sc_name, sc_feat) in enumerate(screen_features.items()):
    start = time.time()
    type_ = str(screen_specs[sc_name].type).split('.')[-1]
    one_hot = screen_features[sc_name]
    timesteps, width, height = one_hot.shape
#    if type_ != 'CATEGORICAL':
#        continue
    # One_hot_Encoding Categorical feature
    if type_ == 'CATEGORICAL':
        results = []
        for j in range(timesteps):
            one_hot_ = one_hot[j]
            one_hot_ = one_hot_.flatten()
            nonzero_indices = np.where(one_hot_ != 0)[0]
            nonzero_values = one_hot_[nonzero_indices]
            
            def get_yx(index):
                return [index // height, index % width]
            
            yx_indices = [get_yx(i) for i in nonzero_indices]
            
            cyx_indices = [[c] + yx for c, yx in zip(nonzero_values, yx_indices)]        
            scale_ = screen_specs[sc_name].scale
            result = np.zeros((scale_, height, width))
            #result[np.array(cyx_indices)] = 1.
            
            for (c, y, x) in cyx_indices:
                result[c][y][x] = 1.
                
            results.append(result)
            
        results = np.asarray(results)
        sc_feats_one_hot[sc_name] = results
        
    # break
    else:
        sc_feats_one_hot[sc_name] = np.expand_dims(one_hot, 1)
        
    print('Elapsed Time: {:.4f}s'.format(time.time() - start)) 
        
    scale_ = screen_specs[sc_name].scale
    print(
            '[{:>02}] Name: {:<15} | Type: {:<11} | Scale: {:>4} | Shape: {} | Size: {}'.format(
            i, sc_name, type_, scale_, sc_feats_one_hot[sc_name].shape, human_readable_size(sc_feat.nbytes)
        )
    )
    print('Elapsed Time: {:.4f}s'.format(time.time() - start))
    

# check = []
# for i in range(len(cyx_indices)):
#   chk_ = cyx_indices[i][0]
#   check.append(chk_)
#    
# max(check)
# check2 = list(set(check))