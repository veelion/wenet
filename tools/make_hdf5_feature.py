#!/usr/bin/env python
# coding:utf-8


import argparse
import os


def get_args():
    parser = argparse.ArgumentParser(description='Extract speech feature to hdf5 files')
    parser.add_argument('--config', required=True,
                        help='train config')
    parser.add_argument('--data_type',
                        default='shard',
                        choices=['raw', 'shard'],
                        help='train/cv data type')
    parser.add_argument('--data_list', required=True,
                        help='data list file')
    parser.add_argument('--h5_dir', required=True,
                        help='save h5 files in the dir')
    parser.add_argument('--h5_list', required=True,
                        help='save h5 files list')
    parser.add_argument('--num_utts_per_h5',
                        type=int,
                        default=2000,
                        help='num utts per h5')
    parser.add_argument('--symbol_table',
                        required=True,
                        help='model unit symbol table for training')
    parser.add_argument("--non_lang_syms",
                        help="non-linguistic symbol file. One symbol per line.")
    parser.add_argument('--bpe_model',
                        default=None,
                        type=str,
                        help='bpe model for english part')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--no_aug',
                        default=False,
                        action='store_true',
                        help='do not augment data')
    args = parser.parse_args()
    return args


def gen_h5name(h5_dir, h5_index):
    return os.path.join(h5_dir, f'feature_{h5_index:09d}.hdf5')


def save_h5(samples, h5name):
    import h5py
    with h5py.File(h5name, 'w') as h5:
        for sm in samples:
            group = h5.create_group(sm['key'])
            group.create_dataset('feat', data=sm['feat'])
            group.create_dataset('label', data=sm['label'])


def extract(data_loader, args):
    import time
    print(f'Extracting features to {args.h5_dir} @ {time.strftime("%Y-%m-%d %H:%M:%S")}')
    total = 0
    h5_index = 0
    if not os.path.exists(args.h5_dir):
        os.makedirs(args.h5_dir)
    h5name = gen_h5name(args.h5_dir, h5_index)
    print('start ', h5name)
    samples = []
    h5_list = []
    start = time.time()
    begin = time.time()
    for sample in data_loader:
        samples.append(sample)
        total += 1
        if total % args.num_utts_per_h5 == 0:
            print(f'done {h5name}/{total = }, time cost:', time.time() - begin)
            save_h5(samples, h5name)
            h5_index += 1
            h5_list.append(h5name)
            samples = []
            h5name = gen_h5name(args.h5_dir, h5_index)
            begin = time.time()
            print('start ', h5name)
    if samples:
        save_h5(samples, h5name)
        h5_list.append(h5name)
    print(f'done, {total = }, {len(h5_list) = }, time cost:', time.time() - start)
    with open(args.h5_list, 'w') as f:
        f.write('\n'.join(h5_list))
        f.write('\n')


def main(args):
    import yaml
    import torch
    from torch.utils.data import DataLoader

    from wenet.dataset.dataset import Dataset
    from wenet.utils.file_utils import read_symbol_table, read_non_lang_symbols
    torch.manual_seed(777)
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    conf = configs['dataset_conf']
    conf['shuffle'] = False  # just extract feature
    conf['sort'] = False  # just extract feature
    if args.no_aug:
        print('no augmentation...')
        conf['speed_perturb'] = False
        conf['spec_aug'] = False
        conf['spec_sub'] = False
    non_lang_syms = read_non_lang_symbols(args.non_lang_syms)
    symbol_table = read_symbol_table(args.symbol_table)
    dataset = Dataset(args.data_type,
                      args.data_list,
                      symbol_table,
                      conf,
                      args.bpe_model,
                      non_lang_syms,
                      partition=False,
                      only_extract=True)
    data_loader = DataLoader(dataset,
                             batch_size=None,
                             pin_memory=args.pin_memory,
                             num_workers=args.num_workers)
    extract(data_loader, args)


if __name__ == '__main__':
    args = get_args()
    main(args)
