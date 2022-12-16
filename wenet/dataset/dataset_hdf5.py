# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2022 veelion
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import h5py
import logging

import torch

import wenet.dataset.processor as processor
from wenet.utils.file_utils import read_lists
from wenet.dataset.dataset import (Processor, DataList)


def hdf5_file_opener(data):
    """ open hdf5 file
    Inplace operation

    Args:
        data(Iterable[str]): path of hdf5 file list

    Returns:
        Iterable[{src, h5}]
    """
    for sample in data:
        assert "src" in sample
        print('========== to open h5 file: ', sample['src'])
        uri = sample['src']
        if not h5py.is_hdf5(uri):
            logging.warning(f'Failed to open {uri}')
            continue
        sample['h5'] = h5py.File(uri)
        yield sample


def hdf5_feature(data):
    """ read feature and label from hdf5

    Args:
        data: Iterable[{src, h5}]

    Returns:
        Iterable[{key, feat, label}]
    """
    for sample in data:
        h5 = sample['h5']
        for key in h5.keys():
            example = {
                'key': key,
                'feat': torch.from_numpy(h5[key]['feat'][:]),
                'label': torch.from_numpy(h5[key]['label'][:]),
            }
            yield example


def DatasetHdf5(data_list_file,
                conf,
                partition=True):
    """ Load dataset from hdf5 file which includes features and labels

        We have two shuffle stage in the Dataset. The first is global
        shuffle at shards tar/raw file level. The second is global shuffle
        at training samples level.

        Args:
            data_list_file: list of hdf5 files
            partition(bool): whether to do data partition in terms of rank
    """
    lists = read_lists(data_list_file)
    shuffle = conf.get('shuffle', True)
    dataset = DataList(lists, shuffle=shuffle, partition=partition)
    dataset = Processor(dataset, hdf5_file_opener)
    dataset = Processor(dataset, hdf5_feature)

    spec_aug = conf.get('spec_aug', True)
    spec_sub = conf.get('spec_sub', False)
    spec_trim = conf.get('spec_trim', False)
    if spec_aug:
        spec_aug_conf = conf.get('spec_aug_conf', {})
        dataset = Processor(dataset, processor.spec_aug, **spec_aug_conf)
    if spec_sub:
        spec_sub_conf = conf.get('spec_sub_conf', {})
        dataset = Processor(dataset, processor.spec_sub, **spec_sub_conf)
    if spec_trim:
        spec_trim_conf = conf.get('spec_trim_conf', {})
        dataset = Processor(dataset, processor.spec_trim, **spec_trim_conf)

    if shuffle:
        shuffle_conf = conf.get('shuffle_conf', {})
        dataset = Processor(dataset, processor.shuffle, **shuffle_conf)

    sort = conf.get('sort', True)
    if sort:
        sort_conf = conf.get('sort_conf', {})
        dataset = Processor(dataset, processor.sort, **sort_conf)

    batch_conf = conf.get('batch_conf', {})
    dataset = Processor(dataset, processor.batch, **batch_conf)
    dataset = Processor(dataset, processor.padding)
    return dataset
