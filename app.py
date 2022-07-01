from flask import Flask, request, jsonify, render_template
import pandas as pd
import os
from model import Predictor

app = Flask(__name__)

app.config['DEBUG'] = True

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

predictor = Predictor()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['uploaded-file']
    
    data_filename = file.filename
    if data_filename == '':
        return render_template('index.html', file_uploaded='No File Selected')
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)
    app.config['UPLOADED_FILE'] = filepath
    file.save(filepath)
    return render_template('index.html', file_uploaded='File Uploaded at '+filepath)
    

@app.route('/predict', methods=['POST'])
def predict():

    filepath = app.config.get('UPLOADED_FILE')
    if filepath == None:
        return render_template('index.html', file_uploaded="FILE NOT FOUND!")
        
    test_df = pd.read_csv(filepath)
    test_df = test_df.reset_index().drop('index', axis=1)

    predictor.load_data(test_df)
    gender_preds = predictor.predict_gender()
    age_preds = predictor.predict_age()
    
    test_df.loc[:,'gender'] = gender_preds
    test_df.loc[:,'age_group'] = age_preds
    
    test_df = test_df.head()
    
    test_df.loc[:,'gender'] = test_df.gender.apply(lambda x: 'Female' if x == 0 else 'Male')
    test_df.loc[:,'age_group'] = test_df.age_group.apply(lambda x: '0-24' if x == 0 else '24-32' if x == 1 else '32+')

    return render_template('index.html', columns=test_df.columns, rows=test_df.values)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
