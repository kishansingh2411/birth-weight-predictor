from flask import Flask,request,jsonify,render_template,Blueprint
import pandas as pd 
import pickle


miniflask=Blueprint("predict",__name__)

def get_clean_data(form_data):
    gestation = float(form_data['gestation'])
    parity = int(form_data['parity'])
    age = float(form_data['age'])
    height = float(form_data['height'])
    weight = float(form_data['weight'])
    smoke = float(form_data['smoke'])
    cleaned_data={
        "gestation":[gestation],
        "parity":[parity],
        "age":[age],
        "height":[height],
        "weight":[weight],
        "smoke":[smoke]
    }

    return cleaned_data

@miniflask.route('/predict',methods=['POST'])
def get_prediction():
    #baby_data=request.get_json
    baby_data_form=request.form

    baby_data_cleaned = get_clean_data(baby_data_form)

    baby_df = pd.DataFrame(baby_data_cleaned)

    
# load machine learnig model
    with open('model.pkl','rb') as f :
        model= pickle.load(f)

# make prediction on user data
    prediction=model.predict(baby_df)
    prediction = round(float(prediction),2)

    response={"prediction":prediction}

    #return jsonify(response)
    return render_template('index.html',prediction=prediction)


@miniflask.route('/',methods=['GET'])
def form():
    return render_template('index.html')
