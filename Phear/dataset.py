# make a dataloader
# https://zhuanlan.zhihu.com/p/346553758
import torch

# import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import cv2


def load_spec(ttv):
    """_summary_

    Args:
        ttv (_type_): _description_

    Returns:
        _type_: _description_
    """
    # get spec
    # to be fast, we can use pickle instead of npy
    # to be fast, we can use jams instead of wav
    # wav_folder = '/home/tonypeng/Workspace1/Hearing/SemanticHearing/AP/data_folder/mix_data_1/' + ttv + '/'
    # npy_folder = '/home/tonypeng/Workspace1/Hearing/SemanticHearing/AP/data_folder/mix_data_1/npy/' + ttv + '/'
    npy_folder = "./data_folder/mix_data_1/npy_log/" + ttv + "/"
    gt_folder = "./data_folder/mix_data_1/npy_log/"

    # record is in , we can use pickle as well
    record_train = pd.read_csv("./data_folder/mix_data_1/record_train.csv")
    record_test = pd.read_csv("./data_folder/mix_data_1/record_test.csv")
    record_val = pd.read_csv("./data_folder/mix_data_1/record_val.csv")

    if ttv == "train":
        record = record_train
    elif ttv == "test":
        record = record_test
    elif ttv == "val":
        record = record_val

    # newfname,fname1,fname2,labels,gt_label,gt_index
    # return newspec, gtspec, gt_label
    newnpys = []
    gtnpys = []
    gt_labels = []
    for i in range(record["newfname"].shape[0]):
        newnpy = np.load(npy_folder + record["newfname"][i] + ".npy")
        if record["gt_index"][i] == 0:
            
            gtnpy = np.load(gt_folder + str(record["fname1"][i]) + ".npy")

        else:
            gtnpy = np.load(gt_folder + str(record["fname2"][i]) + ".npy")
        # npys are from 0 -> 1, we need to change it to 0 -> 255
        gtnpy = gtnpy * 255
        newnpy = newnpy * 255
        # change the type to uiint8
        gtnpy = gtnpy.astype(np.uint8)
        newnpy = newnpy.astype(np.uint8)
        gt_label = record["gt_label"][i]
        newnpys.append(newnpy)
        gtnpys.append(gtnpy)
        gt_labels.append(gt_label)
    return newnpys, gtnpys, gt_labels


class mix_data_1:
    # __init__ is the constructor
    def __init__(self, ttv, resize=False):
        super(mix_data_1, self).__init__()
        self.newnpys, self.gtnpys, self.gt_labels = load_spec(ttv)

        self.resize = resize  # size here is 640*480, width = 640, height = 480
        if self.resize == True:
            #
            self.newnpys = [cv2.resize(npy, (224, 224)) for npy in self.newnpys]
            self.gtnpys = [cv2.resize(npy, (224, 224)) for npy in self.gtnpys]

    def __len__(self):
        return len(self.newnpys)

    def __getitem__(self, idx):
        newnpy = self.newnpys[idx]
        gtnpy = self.gtnpys[idx]
        gt_label = self.gt_labels[idx]

        # change the type to float32
        newnpy = newnpy.astype(np.float32)
        gtnpy = gtnpy.astype(np.float32)

        # change the type to tensor
        newnpy = torch.from_numpy(newnpy)
        gtnpy = torch.from_numpy(gtnpy)

        # tensor, tensor, int
        return newnpy, gtnpy, gt_label

def load_data(ttv, batch_size):
    """_summary_

    Args:
        ttv (_type_): _description_
        batch_size (_type_): _description_

    Returns:
        _type_: _description_
    """
    dataset = mix_data_1(ttv)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader