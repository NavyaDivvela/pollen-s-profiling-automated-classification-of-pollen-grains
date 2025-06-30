import re
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from flask import Flask,app,request,render_template
from keras.models import Model
from keras.preprocessing import image
from tensorflow.python.ops.gen_array_ops import Concat
from keras.models import load_model
model=load_model(r"my_model.keras",compile=False)
app=Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/prediction.html')
def prediction():
    return render_template('prediction.html')
@app.route('/index.html')
def home():
    return render_template('index.html')
@app.route('/logout.html')
def logout():
    return render_template('logout.html')
@app.route('/result',methods=["GET","POST"])
def res():
    if request.method=="POST":
        f=request.files['image']
        basepath=os.path.dirname(__file__)
        filepath=os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)
        img=tf.keras.utils.load_img(filepath,target_size=(128,128))
        x=tf.keras.utils.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        pred=np.argmax(model.predict(x))
        op=['syagrus','tridax','arecaceae','eucalipto','schinus','serjania','matayba','faramea','anadenanthera','mimosa','chromolaena','mabea',
'arrabidaea','qualea','myrcia','dipteryx', 'protium','croton','combretum','hyptis', 'urochloa','cecropia','senegalia']
        op[pred]
        result=op[pred]
        return render_template('prediction.html',pred=result)
if __name__=="__main__":
    app.run(debug=True)







