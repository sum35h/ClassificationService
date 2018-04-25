from flask import Flask,request, jsonify
from flask_restful import Resource,Api
import pandas as pd
from sklearn.externals import joblib
app=Flask(__name__)
api=Api(app)

global model_dia
global model_liver
global model_bre

model_directory = 'model'
pkl_model_dia = '%s/model_dia.pkl' % model_directory
pkl_model_liver= '%s/model_liver.pkl' % model_directory
pkl_model_bre = '%s/model_bre.pkl' % model_directory

@app.route('/home', methods=['GET'])
def home():
     return ("Prediction Service is Running!")

@app.route('/predict_dia', methods=['POST'])
def perdict_dia():
	if model_dia:
		user_data=request.get_json()
		df=pd.DataFrame(user_data);
		prediction=model_dia.predict(df).tolist();
		return jsonify({'prediction': prediction})
	

@app.route('/predict_liver', methods=['POST'])
def perdict_liver():
	if model_liver:
		user_data=request.get_json()
		df=pd.DataFrame(user_data);
		prediction=model_liver.predict(df).tolist();
		return jsonify({'prediction': prediction})

@app.route('/predict_breast', methods=['POST'])
def predict_breast():
	if model_liver:
		user_data=request.get_json()
		df=pd.DataFrame(user_data);
		prediction=model_bre.predict(df).tolist();
		return jsonify({'prediction': prediction})

if __name__ == '__main__':
	try:
		model_dia= joblib.load(pkl_model_dia)
		model_liver= joblib.load(pkl_model_liver)
		model_bre= joblib.load(pkl_model_bre)
		print( 'All models loaded==============================================================')
	except :
		print ('Error: Model not found')
  
	app.run()
