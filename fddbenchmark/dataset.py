import os
from tqdm import tqdm
import zipfile
import requests
import pandas as pd


class FDDDataset():
    def __init__(self, name: str,):
        self.name = name
        self.df = None
        self.labels = None
        self.train_mask = None
        self.test_mask = None
        available_datasets = ['small_tep', 'reinartz_tep', 'rieth_tep', 'lessmeier_bearing']
        available_datasets_str = ', '.join(available_datasets)
        if self.name not in available_datasets:
            raise Exception(
                f'{name} is an unknown dataset. Available datasets are: {available_datasets_str}'
            )
        if self.name in ['small_tep', 'reinartz_tep', 'rieth_tep']:
            self.load_file_by_name(self.name)
        elif self.name == "lessmeier_bearing":
            self.load_file_by_name("lessmeier_bearing_4n1of20")
        
    def load_file_by_name(self, filename: str):
        ref_path = 'data/' + filename + '/'
        if not os.path.exists(ref_path):
            os.makedirs(ref_path)        
        url = "https://industrial-makarov.obs.ru-moscow-1.hc.sbercloud.ru/" + filename + ".zip"
        zfile_path = 'data/' + filename + '.zip'
        if not os.path.exists(zfile_path):
            download_pgbar(url, zfile_path, fname=filename+'.zip')
        
        extracting_files(zfile_path, ref_path)
        self.df = read_csv_pgbar(ref_path + 'dataset.csv', index_col=['run_id', 'sample'])
        self.labels = read_csv_pgbar(ref_path + 'labels.csv', index_col=['run_id', 'sample'])['labels']
        train_mask = read_csv_pgbar(ref_path + 'train_mask.csv', index_col=['run_id', 'sample'])['train_mask']
        test_mask = read_csv_pgbar(ref_path + 'test_mask.csv', index_col=['run_id', 'sample'])['test_mask']
        self.train_mask = train_mask.astype('boolean')
        self.test_mask = test_mask.astype('boolean')


def extracting_files(zfile_path, ref_path):
    with zipfile.ZipFile(zfile_path, 'r') as zfile:
        for entry_info in zfile.infolist():
            if os.path.exists(ref_path + entry_info.filename):
                continue
            input_file = zfile.open(entry_info.filename)
            target_file = open(ref_path + entry_info.filename, 'wb')
            bsize = 1024 * 10000
            block = input_file.read(bsize)
            with tqdm(
                total=entry_info.file_size, 
                desc=f'Extracting {entry_info.filename}', 
                unit='B', 
                unit_scale=True, 
                unit_divisor=1024) as pbar:
                while block:
                    target_file.write(block)
                    block = input_file.read(bsize)
                    pbar.update(bsize)
            input_file.close()
            target_file.close()


def download_pgbar(url, zfile_path, fname):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("Content-Length"))
    with open(zfile_path, 'wb') as file: 
        with tqdm(
            total=total,
            desc=f'Downloading {fname}',
            unit='B',
            unit_scale=True,
            unit_divisor=1024) as pbar:
            for data in resp.iter_content(chunk_size=1024):
                file.write(data)
                pbar.update(len(data))


def read_csv_pgbar(csv_path, index_col, chunksize=1024*100):
    rows = sum(1 for _ in open(csv_path, 'r')) - 1
    chunk_list = []
    with tqdm(total=rows, desc=f'Reading {csv_path}') as pbar:
        for chunk in pd.read_csv(csv_path, index_col=index_col, chunksize=chunksize):
            chunk_list.append(chunk)
            pbar.update(len(chunk))
    df = pd.concat((f for f in chunk_list), axis=0)
    return df
