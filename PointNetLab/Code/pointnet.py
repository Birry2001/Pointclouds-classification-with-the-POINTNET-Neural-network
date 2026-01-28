#!/usr/bin/env python
# PointNet for point cloud classification
#
# -- Paul CHECCHIN - 5/11/2021
#

import numpy as np
import random
import math
import os
import time
import torch
import scipy.spatial.distance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Classe pour la realisation de la rotation des pointclouds suivant l'axe z
class RandomRotation_z(object):
    def __call__(self, pointcloud):
        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[math.cos(theta), -math.sin(theta),      0],
                               [math.sin(theta),  math.cos(theta),      0],
                               [0,                              0,      1]])
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return rot_pointcloud

# Classe pour l'ajout d'un bruit gaussien de moyenne 0 et d'ecart type 0.02 au nuage de points

class RandomNoise(object):
    def __call__(self, pointcloud):
        noise = np.random.normal(0, 0.02, (pointcloud.shape))
        noisy_pointcloud = pointcloud + noise
        return noisy_pointcloud

# Classe pour la modification aleatoire de l'ordre des points, 
# peut servir de petite augmentation(le modèle voit la même forme sous différentes permutations),ce qui aide à éviter de “sur-apprendre” un ordre accidentel
# Ça évite un biais d’ordre introduit par le dataset
class ShufflePoints(object):
    def __call__(self, pointcloud):
        np.random.shuffle(pointcloud)
        return pointcloud

# Classe pour la convertion du nuage de points NumPy en tenseur PyTorch
class ToTensor(object):
    def __call__(self, pointcloud):
        return torch.from_numpy(pointcloud)




# Pipeline de transformations (data augmentation + conversion) appliqué à CHAQUE nuage de points
# dans le Dataset, avant de le donner au réseau.
#
# transforms.Compose enchaîne plusieurs transforms : 1 devient l’entrée de la transfo 2, etc.

def default_transforms():
    return transforms.Compose([RandomRotation_z(),
                               RandomNoise(),
                               ShufflePoints(),
                               ToTensor()])


class PointCloudData(Dataset):
    """
    Dataset PyTorch pour ModelNet
    Rôle :
      - indexer tous les fichiers .ply (chemins + labels)
      - fournir __len__ et __getitem__ pour que DataLoader puisse itérer et créer des batches
      - charger un .ply, construire un nuage Nx3, appliquer des transforms, renvoyer (pointcloud, label)
    """

    def __init__(self,
                 root_dir,
                 folder="train",
                 transform=default_transforms()):
        # root_dir : dossier racine du dataset (contient un dossier par classe)
        self.root_dir = root_dir

        # Liste des noms de classes : on prend uniquement les sous-dossiers de root_dir
        # sorted(...) pour garder un ordre stable (utile pour un mapping classe->id reproductible)
        folders = [dir for dir in sorted(os.listdir(root_dir))
                   if os.path.isdir(root_dir + "/" + dir)]

        # Mapping "nom_de_classe" -> index entier (ex: {"chair": 3, "table": 7, ...})
        # enumerate donne (0, folders[0]), (1, folders[1]), ...
        self.classes = {folder: i for i, folder in enumerate(folders)}

        # Pipeline de transformations appliqué à chaque nuage dans __getitem__
        # (ex: rotation, bruit, shuffle des points, conversion en Tensor)
        self.transforms = transform

        # Liste qui contiendra tous les samples du dataset
        # Chaque élément sera un dict: {'ply_path': ..., 'category': ...}
        self.files = []

        # Parcours de toutes les classes (dossiers) pour indexer tous les fichiers .ply
        for category in self.classes.keys():
            # Exemple: root_dir/chair/train ou root_dir/chair/test
            new_dir = root_dir + "/" + category + "/" + folder

            # On parcourt les fichiers du dossier correspondant à cette classe et ce split
            for file in os.listdir(new_dir):
                # On ne garde que les fichiers PLY (nuages de points)
                if file.endswith('.ply'):
                    sample = {}
                    # Chemin complet du fichier .ply
                    sample['ply_path'] = new_dir + "/" + file
                    # Nom de la classe (string) ; l'index entier sera obtenu via self.classes[category]
                    sample['category'] = category
                    # Ajout dans la liste globale de samples
                    self.files.append(sample)

    def __len__(self):
        """
        Retourne le nombre total d'exemples.
        Permet à len(dataset) de fonctionner et aide DataLoader à savoir combien itérer.
        """
        return len(self.files)

    def __getitem__(self, idx):
        """
        Retourne un exemple à l'index idx.
        dataset[idx] appelle cette méthode.

        Sortie : un dict contenant
          - 'pointcloud' : nuage de points (après transforms)
          - 'category'   : label entier de la classe
        """

        # Récupération des infos du sample indexé
        ply_path = self.files[idx]['ply_path']
        category = self.files[idx]['category']  # string (ex: "chair")

        # Lecture du fichier .ply (renvoie typiquement des champs 'x','y','z' sous forme de vecteurs)
        data = read_ply(ply_path)

        # Construction du nuage de points au format (N, 3)
        # - vstack empile en (3, N)
        # - .T transpose en (N, 3)
        # Ensuite on applique les transformations (rotation/bruit/shuffle/toTensor...)
        pointcloud = self.transforms(np.vstack((data['x'],
                                                data['y'],
                                                data['z'])).T)

        # Conversion du nom de classe (string) en label entier
        label = self.classes[category]

        # On renvoie un sample sous forme de dictionnaire (DataLoader saura faire des batches)
        return {'pointcloud': pointcloud, 'category': label}



class PointMLP(nn.Module):
    def __init__(self, classes=40):
        super().__init__()

        self.conv1 = nn.Conv1d(3, 64,1)
        self.bn1 = nn.BatchNorm1d(64)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv1d(64, 64,1)
        self.bn2 = nn.BatchNorm1d(64)
        self.act2 = nn.ReLU()
        
        self.conv3 = nn.Conv1d(64, 64,1)
        self.bn3 = nn.BatchNorm1d(64)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv1d(64, 128,1)
        self.bn4 = nn.BatchNorm1d(128)
        self.act4 = nn.ReLU()       

        self.conv5 = nn.Conv1d(128, 1024,1)
        self.bn5 = nn.BatchNorm1d(1024)
        self.act5 = nn.ReLU()

        self.maxpool5 = nn.MaxPool1d(1024)

        self.conv6 = nn.Conv1d(1024, 512,1)
        self.bn6 = nn.BatchNorm1d(512)
        self.act6 = nn.ReLU()

        self.conv7 = nn.Conv1d(512, 256,1)
        self.bn7 = nn.BatchNorm1d(512)
        self.act7 = nn.ReLU()
        self.drop7 = nn.Dropout(0.3)

        self.conv8 = nn.Conv1d(256, classes,1)
        self.logsoftmax = nn.LogSoftmax(dim=1)


    def forward(self, input):

        x = self.flatten(input)

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.drop(x)

        x = self.fc3(x)
        x = self.logsoftmax(x)

        return x

        


class PointNetBasic(nn.Module):
    def __init__(self, classes=40):
        super().__init__()
        # input = data[pointcloud] et pointcould = ( B,N, 3)
        self.flatten = nn.Flatten(start_dim=1)

        self.fc1 = nn.Linear(3072, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.act1 = nn.ReLU()

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.act2 = nn.ReLU()
        self.drop = nn.Dropout(0.3)

        self.fc3 = nn.Linear(256, classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        # YOUR CODE
        pass


class Tnet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        # YOUR CODE
        pass

    def forward(self, input):
        # YOUR CODE
        pass


class PointNetFull(nn.Module):
    def __init__(self, classes=40):
        super().__init__()
        # YOUR CODE
        pass

    def forward(self, input):
        # YOUR CODE
        pass


def basic_loss(outputs, labels):
    # NLLLoss = Negative Log-Likelihood Loss :
    # compare des log-probabilités par classe (sortie de LogSoftmax) au label vrai (classe entière)
    criterion = torch.nn.NLLLoss()

    # Taille du batch (nombre d'exemples) 
    bsize = outputs.size(0)

    # Calcule la perte moyenne sur le batch
    return criterion(outputs, labels)



def pointnet_full_loss(outputs, labels, m3x3, alpha=0.001):
    # NLLLoss: attend des log-probabilités (LogSoftmax) et des labels entiers
    criterion = torch.nn.NLLLoss()
    bsize = outputs.size(0)

    # Crée une matrice identité 3x3 I, puis la duplique bsize fois pour obtenir un tenseur (bsize, 3, 3).
    # On veut une identité par élément du batch car m3x3 est aussi une matrice (3x3) par exemple.
    # require_grad=True permet à PyTorch de suivre les opérations pour le calcul des gradients (ici I est une "référence" constante).
    # Si le modèle tourne sur GPU (outputs.is_cuda), on déplace id3x3 sur le GPU aussi, sinon CPU/GPU mélangés => erreur de device.
    id3x3 = torch.eye(3, requires_grad=True).repeat(bsize, 1, 1)
    if outputs.is_cuda:
        id3x3 = id3x3.cuda()


    # Régularisation T-Net : forcer m3x3 à être "presque orthogonale" (m*m^T ≈ I)
    diff3x3 = id3x3 - torch.bmm(m3x3, m3x3.transpose(1, 2))

    # Perte classification + pénalité d'orthogonalité
    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3)) / float(bsize)



def train(model, device, train_loader, test_loader=None, epochs=250):
    # Optimiseur Adam : met à jour les paramètres du modèle via le gradient (lr = pas d'apprentissage)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Scheduler : réduit le lr tous les 20 epochs (lr *= 0.5) pour stabiliser/affiner l'apprentissage
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    loss = 0
    for epoch in range(epochs):

        # Mode entraînement : active dropout, batchnorm en mode "train", etc.
        model.train()

        # Boucle sur les mini-batchs fournis par le DataLoader
        for i, data in enumerate(train_loader, 0):

            # Récupère le batch et l'envoie sur le bon device (CPU/GPU)
            # .float() : s'assure que les points sont en float32 pour le réseau
            inputs = data['pointcloud'].to(device).float()
            labels = data['category'].to(device)

            # Remet à zéro les gradients accumulés (sinon ils s'additionnent d'une itération à l'autre)
            optimizer.zero_grad()

            # Le modèle attend (B, 3, N) : on transpose depuis (B, N, 3)
            outputs = model(inputs.transpose(1, 2))


            # Calcule la loss de classification (ici NLLLoss sur les log-probas)
            loss = basic_loss(outputs, labels)


            # Backprop : calcule dloss/dparams
            loss.backward()

            # Mise à jour des paramètres avec l'optimiseur
            optimizer.step()

        # Mode évaluation : dropout désactivé, batchnorm en mode "eval"
        model.eval()
        correct = total = 0

        # Si on a un loader de test/validation, on calcule l'accuracy
        if test_loader:
            with torch.no_grad():  # pas de gradient en eval -> plus rapide et moins de mémoire
                for data in test_loader:
                    inputs = data['pointcloud'].to(device).float()
                    labels = data['category'].to(device)

                    outputs = model(inputs.transpose(1, 2))
                    # outputs, __ = model(inputs.transpose(1,2))

                    # predicted = argmax sur la dimension "classes"
                    _, predicted = torch.max(outputs.data, 1)

                    # total = nombre total d'exemples vus jusque-là (on ajoute la taille du batch courant)    
                    total += labels.size(0)

                    # (predicted == labels) donne un tenseur de booléens (True si bonne prédiction)
                    # .sum() compte combien de True dans le batch (donc nb de prédictions correctes)
                    # .item() convertit le résultat (tensor scalaire) en nombre Python
                    correct += (predicted == labels).sum().item()


            # Calcule l'accuracy en pourcentage :
            # - correct / total = proportion de prédictions correctes
            # - 100. * ...      = conversion en %
            val_acc = 100. * correct / total


            # Affiche un résumé de l'epoch :
            # - epoch+1 : numéro d'epoch (on commence à 1 pour l'affichage)
            # - loss    : perte (formatée avec 3 décimales)
            # - val_acc : accuracy test (formatée avec 1 décimale) ; "%%" affiche le caractère % dans la string
            print('Epoch: %d, Loss: %.3f, Test accuracy: %.1f %%' % (epoch+1, loss, val_acc))

        # Applique la mise à jour du learning rate selon le scheduler (en fin d'epoch)
        scheduler.step()


if __name__ == '__main__':
    t0 = time.time()
    train_ds = PointCloudData("/home/nochi/NOCHI/M2_PAR/Apprenyissage_Pointcloud/PointNetLab/data/ModelNet10_PLY")
    test_ds = PointCloudData("/home/nochi/NOCHI/M2_PAR/Apprenyissage_Pointcloud/PointNetLab/data/ModelNet10_PLY", folder='test')

    inv_classes = {i: cat for cat, i in train_ds.classes.items()}
    print("Classes: ", inv_classes)
    print('Train dataset size: ', len(train_ds))
    print('Test dataset size: ', len(test_ds))
    print('Number of classes: ', len(train_ds.classes))
    print('Sample pointcloud shape: ', train_ds[0]['pointcloud'].size())

    train_loader = DataLoader(dataset=train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_ds, batch_size=32)

    model = PointMLP()
    # model = PointNetBasic()
    # model = PointNetFull()

    # Récupère uniquement les paramètres "entraînables" du modèle :
    # model.parameters() parcourt tous les poids/biais ; requires_grad=True => ils seront mis à jour par backprop
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())


    # Affiche le nombre total de paramètres entraînables :
    # p.size() = dimensions du tenseur (ex: (512, 3072)), np.prod(...) = produit des dims => nb d'éléments
    # sum(...) = addition sur tous les paramètres (poids + biais)
    print("Number of parameters in the Neural Networks: ",
          sum([np.prod(p.size()) for p in model_parameters]))


    # Choisit le device d'exécution :
    # - si un GPU CUDA est dispo -> "cuda:0"
    # - sinon -> CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    # Déplace le modèle sur ce device (tous ses paramètres seront sur GPU ou CPU)
    model.to(device)


    train(model, device, train_loader, test_loader, epochs=250)
    print("Total time for training : ", time.time() - t0)
