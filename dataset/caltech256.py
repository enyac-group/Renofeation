import torch.utils.data as data
from PIL import Image
import glob
import time
import numpy as np
import random
import os


class Caltech257Data(data.Dataset):
    def __init__(self, root, is_train=True, transform=None, shots=5, seed=0, preload=False):
        self.num_classes = 257
        self.transform = transform
        self.preload = preload
        cls = glob.glob(os.path.join(root, 'Images', '*'))

        self.labels = []
        self.image_path = []

        test_samples = 20

        #random.seed(int(time.time()))

        for idx, cls_path in enumerate(cls):
            cls_label = int(cls_path.split('/')[-1][:3])-1
            imgs = glob.glob(os.path.join(cls_path, '*.jpg'))
            imgs = np.array(imgs)
            indices = np.arange(0, len(imgs))
            random.seed(99+idx)
            random.shuffle(indices)

            if is_train:
                trainval_ind = indices[:int(shots)]
                np.concatenate((trainval_ind, indices[int(shots+test_samples):]), axis=0)
                random.seed(seed+idx)
                random.shuffle(trainval_ind)
                cur_img_paths = imgs[trainval_ind[:int(shots)]]

            else:
                cur_img_paths = imgs[indices[shots:shots+test_samples]]
            self.image_path.extend(cur_img_paths)
            self.labels.extend([cls_label for _ in range(len(cur_img_paths))])

        random.seed(int(time.time()))

        if preload:
            self.imgs = {}
            for idx, path in enumerate(self.image_path):
                if idx % 100 == 0:
                    print('Loading {}/{}...'.format(idx+1, len(self.image_path)))
                img = Image.open(path).convert('RGB')
                self.imgs[idx] = img

    def __getitem__(self, index):
        if self.preload:
            img = self.imgs[index]
        else:
            img = Image.open(self.image_path[index]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[index]

    def __len__(self):
        return len(self.labels)

if __name__ == '__main__':
    seed= int(time.time())
    data_train = Caltech257Data('/data/caltech_256', 'train', shots=30, seed=seed)
    data_test = Caltech257Data('/data/caltech_256', 'test', shots=30, seed=seed)
    for i in data_train.image_path:
        if i in data_test.image_path:
            print('Test in training...')
    print(data_train.image_path[:5])
    print(data_test.image_path[:5])
