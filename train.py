from models.basemodel import BaseModel
from torch.nn import Conv2d
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch


class superLotteryData(Dataset):
    def __init__(self, d_path, col_num, window_size, is_train=True):
        """Initialization"""
        if is_train:
            df = pd.read_excel(d_path, engine='openpyxl')[100:]
        else:
            df = pd.read_excel(d_path, engine='openpyxl')[:100]
        self.data = []
        self.window_size = window_size
        self.col_num = col_num
        for idx, row in df.iterrows():
            self.data.append(list(map(int, row['result'].split(' '))))
        self.data = np.array(self.data, dtype=np.float32)

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.data) - self.window_size

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        X = torch.tensor(self.data[index:index + self.window_size], dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.data[index + self.window_size][self.col_num]-1, dtype=torch.float32).type(torch.LongTensor)
        return X, y


def main():
    model_name = 'resnet50'
    num_classes = 35
    num_workers = 0
    window_size = 100
    col_num = 0

    resnet_classifier = BaseModel(model_name, num_classes)
    resnet_classifier.model_ft.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    sld_train = superLotteryData('./data/super_lottery/all_data.xlsx', col_num, window_size)
    sld_val = superLotteryData('./data/super_lottery/all_data.xlsx', col_num, window_size, is_train=False)
    sldl_train = DataLoader(sld_train, batch_size=5, shuffle=False, num_workers=num_workers)
    sldl_val = DataLoader(sld_val, batch_size=5, shuffle=False, num_workers=num_workers)
    dataloaders = {
        'train': sldl_train,
        'val': sldl_val
    }

    model = resnet_classifier.train_model(dataloaders=dataloaders, num_epochs=20)


if __name__ == '__main__':
    main()
