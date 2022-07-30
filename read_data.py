import os
import scipy.io as sio
import numpy as np
import torch
import torchvision.transforms as transforms
import h5py
from torch.utils.data import Dataset
from random import randint, random
import os
import cv2
from PIL import Image
from PIL import ImageSequence


def data_read():
    data = []

    image_paths = os.listdir('data')
    k =  0
    for path in image_paths:
        gif_path = 'data\\'+ path
        gif_names = os.listdir(gif_path)

        for name in gif_names:
            gif_load = GIF(gif_path + '\\' + name)
            gif_imgs = gif_load.load()
            data.append(np.array(gif_imgs, dtype=np.float32))
            k  +=1
            print(k)

    return data

def data_gen(train_size=0.7):

    names = []
    case_name  = []
    image_paths = os.listdir('data')
    for path in image_paths:
        gif_path = 'data\\' + path
        names.extend(os.listdir(gif_path))
    for i, name in enumerate(names):
        if i < 40:
            tem = name.split("_")
            case_name.append(('5-' + tem[2] + '-' + tem[-1].split('.')[0]))
        elif i < 72:
            tem =  name.split("_")
            case_name.append(('10-' + tem[1] + '-' + tem[-1].split('.')[0]))
        else:
            tem =  name.split("_")
            case_name.append(('15-' + tem[2] + '-' + tem[-1].split('.')[0]))

    case_num = len(case_name)
    np.random.seed(2022)
    ind = np.random.permutation(len(case_name))
    train_case = np.array(case_name)[ind[:int(case_num*train_size)]]
    valid_case = np.array(case_name)[ind[int(case_num*train_size):]]

    train_names = []
    valid_names = []
    train_label = []
    valid_label = []
    for case in train_case:
        for i in range(1, 181):
            train_names.append('png\\' + case  + '-' + str(i)  + '.png')
            tmp =  case.split('-')
            train_label.append([float(tmp[0]), float(tmp[1]), float(tmp[2]), float(i)])

    for case in valid_case:
        for i in range(1, 181):
            valid_names.append('png\\' + case  + '-' + str(i)  + '.png')
            tmp =  case.split('-')
            valid_label.append([float(tmp[0]), float(tmp[1]), float(tmp[2]), float(i)])

    return (train_names, valid_names), (np.array(train_label,dtype=np.float32), np.array(valid_label,dtype=np.float32))


class GIF():
    def __init__(self, file_path):
        self.file_path = file_path
        self.material = []

    def load(self):
        cap = cv2.VideoCapture(self.file_path)
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        res = []
        while True:
            ret, frame = cap.read()
            if ret is False:
                break
            res.append(frame)
        self.material = res
        self.total_frames = len(res)
        cap.release()
        return self.material

    def get_durations(self):
        if self.fps is None:
            return 0.125
        return 1/self.fps


class data_norm():

    def __init__(self, data, method="min-max"):
        axis = tuple(range(len(data.shape) - 1))
        self.method = method
        if method == "min-max":
            self.max = np.max(data, axis=axis)
            self.min = np.min(data, axis=axis)

        elif method == "mean-std":
            self.mean = np.mean(data, axis=axis)
            self.std = np.std(data, axis=axis)

    def norm(self, x):
        if torch.is_tensor(x):
            if self.method == "min-max":
                x = 2 * (x - torch.tensor(self.min, device=x.device))\
                    / (torch.tensor(self.max, device=x.device) - torch.tensor(self.min, device=x.device) + 1e-10) - 1
            elif self.method == "mean-std":
                x = (x - torch.tensor(self.mean, device=x.device)) / (torch.tensor(self.std, device=x.device) + 1e-10)
        else:
            if self.method == "min-max":
                x = 2 * (x - self.min) / (self.max - self.min+1e-10) - 1
            elif self.method == "mean-std":
                x = (x - self.mean) / (self.std + 1e-10)

        return x

    def back(self, x):
        if torch.is_tensor(x):
            if self.method == "min-max":
                x = (x + 1) / 2 * (torch.tensor(self.max, device=x.device)
                                   - torch.tensor(self.min, device=x.device) + 1e-10) + torch.tensor(self.min, device=x.device)
            elif self.method == "mean-std":
                x = x * (torch.tensor(self.std, device=x.device) + 1e-10) + torch.tensor(self.mean, device=x.device)
        else:
            if self.method == "min-max":
                x = (x + 1) / 2 * (self.max - self.min+1e-10) + self.min
            elif self.method == "mean-std":
                x = x * (self.std + 1e-10) + self.mean
        return x



class custom_dataset(Dataset):
    def __init__(self, names, label):
        self.names = names
        self.label = label

        self.transform = transforms.Compose(
            [
                transforms.Resize([512, 512]),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
    def __getitem__(self, idx):  # 根据 idx 取出其中一个

        image = Image.open(self.names[idx])
        image = self.transform(image)
        label = self.label[idx]

        return image, label


    def __len__(self):  # 总数据的多少
        return len(self.names)

if __name__ == '__main__':

    if not os.path.exists('png'):
        os.makedirs('png')

    # duration = gif_load.get_durations()
    names = []
    image_paths = os.listdir('data')
    for path in image_paths:
        gif_path = 'data\\' + path
        names.extend(os.listdir(gif_path))

    for i, name in enumerate(names):
        k = 0
        if i < 40:
            img = Image.open('data\\' + image_paths[0] + '\\' + name)
            tem =  name.split("_")
            for frame in ImageSequence.Iterator(img):
                frame.save('png\\' + '5-' + tem[2] + '-' + tem[-1].split('.')[0] + "-%d.png" % k)
                k += 1
        elif i < 72:
            img = Image.open('data\\' + image_paths[1] + '\\' + name)
            tem =  name.split("_")
            for frame in ImageSequence.Iterator(img):
                frame.save('png\\' + '10-' + tem[1] + '-' + tem[-1].split('.')[0] + "-%d.png" % k)
                k += 1

        else:
            img = Image.open('data\\' + image_paths[2] + '\\' + name)
            tem =  name.split("_")
            for frame in ImageSequence.Iterator(img):
                frame.save('png\\' + '15-' + tem[2] + '-' + tem[-1].split('.')[0] + "-%d.png" % k)
                k += 1

    a= 0


