"""
functions for datasets
"""

import pandas as pd
from sklearn.model_selection import train_test_split

LUCKY_SEED = len('Kontur_DS_2023')

def stratified_train_test_split(dataset):
    """
    split given dataset in huggingface datasets format in train and test part
    but make split stratified by labels and empty parts in extracted_part, 
    by generating new feature in one hot style

    Parameters:
    -----------
        dataset (datasets.dataset_dict.DatasetDict):
            dataset for splitting: has only 'train' field
    
    Return:
    -------
        dataset (datasets.dataset_dict.DatasetDict):
            splitted dataset, that has 'train' and 'test' fields
    """

    stratify_feature = []

    for sample in dataset['train']:

        one_hot_feature = ''
        if sample['label'] == 'обеспечение гарантийных обязательств':
            one_hot_feature += '1'
            if sample['extracted_part']['text'] == ['']:
                one_hot_feature += '1'
            else:
                one_hot_feature += '0'
        else:
            one_hot_feature += '0'
            if sample['extracted_part']['text'] == ['']:
                one_hot_feature += '1'
            else:
                one_hot_feature += '0'

        stratify_feature.append(one_hot_feature)

    stratify_feature = pd.Series(stratify_feature, name='stratify_by')

    train_indices, test_indices = train_test_split(stratify_feature, 
                                                test_size=0.2, 
                                                stratify=stratify_feature,
                                                shuffle=True,
                                                random_state=LUCKY_SEED)

    dataset['test'] = dataset['train'].select(test_indices.index)
    dataset['train'] = dataset['train'].select(train_indices.index)

    return dataset