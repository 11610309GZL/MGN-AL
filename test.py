# import torch
# import collections
# from torch.utils.data import dataset, dataloader
# from torch.utils.data import sampler
# from torchvision.datasets.folder import default_loader
#
# class myTestDataset(dataset.Dataset):
#
#     def __init__(self):
#         self.data = [[11, 12, 13], [21, 22, 23], [31, 32, 33]]
#         self.label = [1, 2, 3]
#
#     def __getitem__(self, item):
#
#         return self.data[item], self.label[item]
#
# class testSampler(sampler.Sampler):
#
#     def __init__(self):
#         self.my_indexes = [2 ,1]
#
#     def __iter__(self):
#         return iter(self.my_indexes)
#
#     def __len__(self):
#         return len(self.my_indexes)
#
# if __name__ == "__main__":
#     test_dataset = myTestDataset()
#
#     testdataloader = dataloader.DataLoader(dataset=test_dataset, sampler=testSampler())
#
#     for (data, label) in testdataloader:
#         print("data: {} label: {}".format(data, label))

a = ['1', '2']
a.remove('1')
print(a)