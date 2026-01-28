

import numpy as np
import random
import math
import os
import time
import torch
import scipy.spatial.distance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Import functions to read and write ply files
from ply import write_ply, read_ply

import MinkowskiEngine as ME
import examples.resnet as resnets



class RandomRotation_z(object):
    def __call__(self, pointcloud):
        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[ math.cos(theta), -math.sin(theta),      0],
                               [ math.sin(theta),  math.cos(theta),      0],
                               [0,                               0,      1]])
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return rot_pointcloud

class RandomNoise(object):
    def __call__(self, pointcloud):
        noise = np.random.normal(0, 0.02, (pointcloud.shape))
        noisy_pointcloud = pointcloud + noise
        return noisy_pointcloud

class ShufflePoints(object):
    def __call__(self, pointcloud):
        np.random.shuffle(pointcloud)
        return pointcloud

class ToTensor(object):
    def __call__(self, pointcloud):
        return torch.from_numpy(pointcloud)


def default_transforms():
    return transforms.Compose([RandomRotation_z(),RandomNoise(),ShufflePoints()])


class PointCloudData(torch.utils.data.Dataset):
    def __init__(self, root_dir, folder="train", transform=default_transforms(), size_voxel=0.01):
        self.voxel_size = size_voxel
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir+"/"+dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform
        self.files = []
        for category in self.classes.keys():
            new_dir = root_dir+"/"+category+"/"+folder
            for file in os.listdir(new_dir):
                if file.endswith('.ply'):
                    sample = {}
                    sample['ply_path'] = new_dir+"/"+file
                    sample['category'] = category
                    self.files.append(sample)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        ply_path = self.files[idx]['ply_path']
        category = self.files[idx]['category']
        data = read_ply(ply_path)
        pointcloud = self.transforms(np.vstack((data['x'], data['y'], data['z'])).T)
        coords = pointcloud / self.voxel_size
        feats = pointcloud
        label = self.classes[category]
        return (coords, feats, label)


def collate_pointcloud_fn(list_data):
    new_list_data = []
    num_removed = 0
    for data in list_data:
        if data is not None:
            new_list_data.append(data)
        else:
            num_removed += 1

    list_data = new_list_data

    if len(list_data) == 0:
        raise ValueError("No data in the batch")

    coords, feats, labels = list(zip(*list_data))

    eff_num_batch = len(coords)
    assert len(labels) == eff_num_batch

    coords_batch = ME.utils.batched_coordinates(coords, dtype=torch.float32)
    feats_batch = torch.from_numpy(np.vstack(feats)).float()

    # Concatenate all lists
    return {
        "coords": coords_batch,
        "feats": feats_batch,
        "labels": torch.LongTensor(labels),
    }


def train(model, device, train_loader, test_loader=None, epochs=250):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    crit = torch.nn.CrossEntropyLoss()
    loss=0
    for epoch in range(epochs): 
        model.train()
        for i, data in enumerate(train_loader, 0):
            optimizer.zero_grad()
            sin = ME.TensorField(data['feats'], data['coords'], device=device)
            sout = model(sin)
            loss = crit(sout.F, data['labels'].to(device))
            loss.backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        if test_loader:
            with torch.no_grad():
                for data in test_loader:
                    sin = ME.TensorField(data['feats'], data['coords'], device=device)
                    sout = model(sin)
                    is_correct = data["labels"] == torch.argmax(sout.F, 1).cpu()
                    total += len(sout)
                    correct += is_correct.sum().item()
            val_acc = 100. * correct / total
            print('Epoch: %d, Loss: %.3f, Test accuracy: %.1f %%' %(epoch+1, loss, val_acc))

        scheduler.step()




if __name__ == "__main__":

    t0 = time.time()

    size_voxel = 0.005
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    
    train_ds = PointCloudData("ModelNet40_PLY", size_voxel=size_voxel)
    test_ds = PointCloudData("ModelNet40_PLY", folder='test', size_voxel=size_voxel)

    inv_classes = {i: cat for cat, i in train_ds.classes.items()}
    print("Classes: ", inv_classes)
    print('Train dataset size: ', len(train_ds))
    print('Test dataset size: ', len(test_ds))
    print('Number of classes: ', len(train_ds.classes))
    #print('Sample pointcloud shape: ', train_ds[0]['feats'])
    #sys.exit()

    train_loader = DataLoader(dataset=train_ds, num_workers=1, collate_fn=collate_pointcloud_fn, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_ds, num_workers=1, collate_fn=collate_pointcloud_fn, batch_size=32)

    default_model="ResFieldNet14"
    #other models: ResFieldNet14, ResFieldNet18, ResFieldNet34, ResFieldNet50, ResFieldNet101",
    model = getattr(resnets, default_model)(3, 40, D=3)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print("Number of parameters in the Neural Networks: ", sum([np.prod(p.size()) for p in model_parameters]))
    model.to(device)

    train(model, device, train_loader, test_loader, epochs = 250)
    
    print("Total time for training : ", time.time()-t0)
