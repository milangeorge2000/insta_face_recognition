from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import os
from torchvision import datasets
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from PIL import Image
from flask_ngrok import run_with_ngrok
from flask import jsonify
import direct
# print("hello")
app = Flask(__name__)
run_with_ngrok(app)
  
@app.route("/")
def hello():
    return "Hello Geeks!! from Google Colab"

mtcnn = torch.load("data1.pt")
resnet = torch.load("data3.pt")
# saved_data = torch.load('data2.pt') # loading data.pt file



@app.route('/', methods=['GET'])
def index():
    # Main page
    return "hello"


@app.route('/save', methods=['GET', 'POST'])
def upload():
    saved_data = torch.load('data2.pt')

    def face_match(img_path, data_path): # img_path= location of photo, data_path= location of data.pt 
        # getting embedding matrix of the given img

        img = Image.open(img_path)
        face, prob = mtcnn(img, return_prob=True) # returns cropped face and probability
        if(face is not None):

            emb = resnet(face.unsqueeze(0)).detach() # detech is to make required gradient false
            
            
            embedding_list = saved_data[0] # getting embedding data
            name_list = saved_data[1] # getting list of names
            dist_list = [] # list of matched distances, minimum distance is used to identify the person
            
            for idx, emb_db in enumerate(embedding_list):
                dist = torch.dist(emb, emb_db).item()
                dist_list.append(dist)
                
            idx_min = dist_list.index(min(dist_list))
            return (name_list[idx_min],min(dist_list))
        else:
            return "noface"
    

    if request.method == 'POST':
        f1 = request.files['img1']
        f2 = request.files['img2']
        f3 = request.files['img3']
        f4 = request.files['img4']
        f5 = request.files['img5']
        name = request.form['username']
        
        print(name)
        f1.save("img1.jpg")
               
        result = face_match("img1.jpg", 'data2.pt')
        print(result[0])
        print(result[1])
        if(result[0] == "nomatch" or result[1] > 0.8):
  
        
        # mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20) # initializing mtcnn for face detection
        # resnet = InceptionResnetV1(pretrained='vggface2').eval() # initializing resnet for face img to embeding conversion
            direct.create(name)
            # f1.save(f"train/{name}/img1.jpg")
            f2.save(f"train/{name}/img2.jpg")
            f3.save(f"train/{name}/img3.jpg")
            f4.save(f"train/{name}/img4.jpg")
            f5.save(f"train/{name}/img5.jpg") 
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
            torch.save(data, 'data2.pt') # saving data.pt file
            return jsonify({"status": "success"})
        
        else:
            return jsonify(result[0])
            



@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # mtcnn = torch.load("data1.pt")
    # resnet = torch.load("data3.pt")
    saved_data = torch.load('data2.pt')
    if request.method == 'POST':
        f = request.files['img1']
        f.save("img.jpg")

 
        def face_match(img_path, data_path): # img_path= location of photo, data_path= location of data.pt 
            # getting embedding matrix of the given img
    
            img = Image.open(img_path)
            face, prob = mtcnn(img, return_prob=True) # returns cropped face and probability
            if(face is not None):

                emb = resnet(face.unsqueeze(0)).detach() # detech is to make required gradient false
                
                
                embedding_list = saved_data[0] # getting embedding data
                name_list = saved_data[1] # getting list of names
                dist_list = [] # list of matched distances, minimum distance is used to identify the person
                
                for idx, emb_db in enumerate(embedding_list):
                    dist = torch.dist(emb, emb_db).item()
                    dist_list.append(dist)
                    
                idx_min = dist_list.index(min(dist_list))
                return (name_list[idx_min])
            else:
                return "noface"
        result = face_match("img.jpg", 'data2.pt')
        return jsonify(result)
  
if __name__ == "__main__":
  app.run()