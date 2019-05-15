import collections
import os
import re

def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm|npy'):
    assert os.path.isdir(directory), 'dataset is not exists!{}'.format(directory)

    return sorted([os.path.join(root, f)
                   for root, _, files in os.walk(directory) for f in files
                   if re.match(r'([\w]+\.(?:' + ext + '))', f)])

def get_path(dataset_path, dtype):
    """
    :param dataset_path: path to the Market1501
    :return corresponding path of the original train set, query, gallery
    get the corresponding dataset path
    """
    if dtype == 'train':
        data_path = dataset_path + '/bounding_box_train'
    elif dtype == 'test':
        data_path = dataset_path + '/bounding_box_test'
    else:
        data_path = dataset_path + '/query'

    return data_path


def id_of(img_path):
    return int(img_path.split('/')[-1].split('_')[0])


def get_all_imgs_path(train_set_path):
    return [path for path in list_pictures(train_set_path) if id_of(path) != -1]


def get_unique_ids(all_imgs_path):
    person_ids = []
    for img_path in all_imgs_path:
        person_ids.append(id_of(img_path))

    return sorted(set(person_ids))


"""
divide the original train set into labeled set (for training) and unlabeled set
"""
def divide_trainset (unique_ids, all_images_path):
    labeled_path = []

    _id2indexlist = collections.defaultdict(list)
    for idx, path in enumerate(all_images_path):
        _id = int(path.split('/')[-1].split('_')[0])
        _id2indexlist[_id].append(path)

    # choose labeled data
    for _id in unique_ids:
        labeled_path.append(_id2indexlist[_id][0])

    unlabled_path = list(set(all_images_path) - set(labeled_path))

    return labeled_path, unlabled_path

def addLabeled(train_loader, unlabeled_loader):

    # choose some valuable data
    new_labeled_data = unlabeled_loader.imgs[0:10]
    for new_labeled in new_labeled_data:
        unlabeled_loader.imgs.remove(new_labeled)

    train_loader.imgs.extend(new_labeled_data)
    return new_labeled_data