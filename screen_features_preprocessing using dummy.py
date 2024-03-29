# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 12:47:18 2019

@author: user
"""

import os
import time
import numpy as np
import pandas as pd

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

sc_feats_one_hot = {}
print('[Screen features]')
print('=' * 105)
for i, (sc_name, sc_feat) in enumerate(screen_features.items()):
    start = time.time()
    type_ = str(screen_specs[sc_name].type).split('.')[-1]
    one_hot = screen_features[sc_name]

    # One_hot_Encoding Categorical feature
    if type_ == 'CATEGORICAL':
        for j in range(len(screen_features[sc_name])):  # loop through time index
            # one_hot_.shape(height * width)
            one_hot_ = one_hot[j]
            #one_hot_ = one_hot_.reshape(one_hot_.shape[0]*one_hot_.shape[1])
            one_hot_ = one_hot_.flatten()
            one_hot_dum = pd.get_dummies(one_hot_)
            if one_hot_dum.columns[0] != 0:
                one_hot_dum = np.array(np.transpose(one_hot_dum))
                # final one_hot_dum.shape(Number of categorical * height * width)
                one_hot_dum = one_hot_dum.reshape(one_hot_dum.shape[0], one_hot.shape[1], one_hot.shape[2])
            else:
                # if value of categorcal = 0, delete [0] column
                one_hot_dum = one_hot_dum.drop([0], axis=1)
                one_hot_dum = np.array(np.transpose(one_hot_dum))
                # final one_hot_dum.shape(Number of categorical-1 * height * width)
                one_hot_dum = one_hot_dum.reshape(one_hot_dum.shape[0], one_hot.shape[1], one_hot.shape[2])
            # sc_feat_dum = categorical feature after one_hot_encoding
            if j == 0:
                sc_feat_dum = one_hot_dum
            else:
                sc_feat_dum = np.concatenate([sc_feat_dum, one_hot_dum], axis=0)
    
            sc_feats_one_hot[sc_name] = sc_feat_dum
            
    # Scalar feature
    else:
        sc_feats_one_hot[sc_name] = one_hot
        
    scale_ = screen_specs[sc_name].scale
    print(
            '[{:>02}] Name: {:<15} | Type: {:<11} | Scale: {:>4} | Shape: {} | Size: {}'.format(
            i, sc_name, type_, scale_, sc_feats_one_hot[sc_name].shape, human_readable_size(sc_feat.nbytes)
        )
    )
    print('Elapsed Time: {:.4f}s'.format(time.time() - start))
    # break
