import os
from torchvision import datasets
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from PIL import Image

mtcnn = torch.load("data1.pt")
resnet = torch.load("data3.pt")




dataset=datasets.ImageFolder('train') # photos folder path 
idx_to_class = {i:c for c,i in dataset.class_to_idx.items()} # accessing names of peoples from folder names

def collate_fn(x):
    return x[0]

loader = DataLoader(dataset, collate_fn=collate_fn)

face_list = [] # list of cropped faces from photos folder
name_list = [] # list of names corrospoing to cropped photos
embedding_list = [] # list of embeding matrix after conversion from cropped faces to embedding matrix using resnet

for img, idx in loader:
    face, prob = mtcnn(img, return_prob=True) 
    if face is not None and prob>0.90: # if face detected and porbability > 90%
        emb = resnet(face.unsqueeze(0)) # passing cropped face into resnet model to get embedding matrix
        embedding_list.append(emb.detach()) # resulten embedding matrix is stored in a list
        name_list.append(idx_to_class[idx]) # names are stored in a list

data = [embedding_list, name_list]
torch.save(data, 'data2.pt')