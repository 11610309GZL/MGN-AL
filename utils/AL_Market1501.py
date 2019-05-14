import collections
from torch.utils.data import dataset
from torchvision.datasets.folder import default_loader

import os
import re





class AlMarket1501(dataset.Dataset):
    def __init__(self, transform, dtype, data_path):

        self.transform = transform
        self.loader = default_loader
        self.data_path = data_path

        if dtype == 'train':
            self.data_path += '/bounding_box_train'
        elif dtype == 'test':
            self.data_path += '/bounding_box_test'
        else:
            self.data_path += '/query'

        self.allImgs = [path for path in self.list_pictures(self.data_path) if self.id(path) != -1]
        self._id2label = {_id: idx for idx, _id in enumerate(self.unique_ids)}

        self.imgs, self.unlabeled = self.initLabeled(self.unique_ids, self.allImgs)


       #字典，人的原始ID映射人的有序ID


    def __getitem__(self, index):
        path = self.imgs[index]
        target = self._id2label[self.id(path)]

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def id(file_path):
        """
        :param file_path: unix style file path
        :return: person id
        """
        return int(file_path.split('/')[-1].split('_')[0])

    @staticmethod
    def camera(file_path):
        """
        :param file_path: unix style file path
        :return: camera id
        """
        return int(file_path.split('/')[-1].split('_')[1][1])

    @property
    def ids(self):
        """
        :return: person id list corresponding to dataset image paths
        """
        # return [self.id(path) for path in self.imgs]
        return [self.id(path) for path in self.allImgs]

    @property
    def unique_ids(self):
        """
        :return: unique person ids in ascending order
        """
        return sorted(set(self.ids))

    @property
    def cameras(self):
        """
        :return: camera id list corresponding to dataset image paths
        """
        return [self.camera(path) for path in self.imgs]

    @staticmethod
    def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm|npy'):
        assert os.path.isdir(directory), 'dataset is not exists!{}'.format(directory)

        return sorted([os.path.join(root, f)
                       for root, _, files in os.walk(directory) for f in files
                       if re.match(r'([\w]+\.(?:' + ext + '))', f)])

    @staticmethod
    def initLabeled(unique_ids, all_images):

        labeled = []

        _id2indexlist = collections.defaultdict(list)
        for idx, path in enumerate(all_images):
            _id = int(path.split('/')[-1].split('_')[0])
            _id2indexlist[_id].append(path)

        for _id in unique_ids:
            labeled.append(_id2indexlist[_id][0])

        unlabled = list(set(all_images) - set(labeled))

        return labeled, unlabled


