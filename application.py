from flask import Flask,request,render_template,jsonify
import os
import sys
from src.pipelines.prediction_pipeline import Predict_Pipeline,CustomData


application=Flask(__name__)

app=application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
            Company=request.form.get('Company'),
            TypeName = request.form.get('TypeName'),
            Ram = int(request.form.get('Ram')),
            Weight = float(request.form.get('Weight')),
            Touchscreen = int(request.form.get('Touchscreen')),
            IPS=int(request.form.get('IPS')),
            ppi = float(request.form.get('ppi')),
            CPU_BRAND = request.form.get('CPU_BRAND'),
            HDD= request.form.get('HDD'),
            SSD = request.form.get('SSD'),
            GPU_BRAND = request.form.get('GPU_BRAND'),
            OS = request.form.get('OS')
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=Predict_Pipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred[0],2)

        return render_template('form.html',final_result=results)
    

if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)