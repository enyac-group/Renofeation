import torch.utils.data as data
from PIL import Image
import os


class ImageNetData(data.Dataset):
    def __init__(self, root, is_train=True, transform=None):
        self.transform = transform
        self.num_classes = 1000

        if is_train:
            mapfile = os.path.join(root, 'train.txt')
            imageset = 'train'
        else:
            mapfile = os.path.join(root, 'val.txt')
            imageset = 'val'
        assert os.path.exists(mapfile), 'The mapping file does not exist!'

        self.datapaths = []
        self.labels = []
        with open(mapfile) as f:
            for l in f:
                self.datapaths.append('{}/{}/{}'.format(root, imageset, l.split(' ')[0].strip()))
                self.labels.append(int(l.split(' ')[1].strip()))


    def __getitem__(self, index):
        img_label = self.labels[index]
        #img = self.data[index]
        img = Image.open(self.datapaths[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        
        return img, img_label

    def __len__(self):
        return len(self.labels)
     
