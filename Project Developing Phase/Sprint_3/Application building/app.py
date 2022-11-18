import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask,render_template,request

app=Flask(__name__)

model=load_model('C:/Users/dharshini/Desktop/Application building/ECG.h5')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict',methods=['GET','POST'])
def upload():
    text=""
    if request.method=='POST':
        f=request.files['image']
        basepath=os.path.dirname(__file__)
        filepath=os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)
        img=image.load_img(filepath,target_size=(64,64))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        pred=np.argmax(model.predict(x),axis=1)
        if pred==0:
            text="left Bundle Branch block"
            print(text)
            
        elif pred==1:
            text="Normal"
            print(text)
            
        elif pred==2:
            text="Premature Atrial Contraction"
            print(text)
            
        elif pred==3:
            text="Premature Ventricular Contraction"
            print(text)
            
        elif pred==4:
            text="Right Bundle Branch Block"
            print(text)
        else:
            text="Ventricular Fibrillation"
            print(text)
     
    return text
if __name__=='__main__':
    app.run(debug=False)