from flask import Flask, request, send_file, render_template
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
    test_df = test_df.reset_index().drop('index', axis=1).drop_duplicates()

    predictor.load_data(test_df.drop('device_id',axis=1))
    gender_preds = predictor.predict_gender()
    male_probs = predictor.predict_gender_prob()
    age_preds = predictor.predict_age()

    test_df = test_df[['device_id']]
    test_df.loc[:,'gender'] = gender_preds
    test_df.loc[:,'age_group'] = age_preds
    test_df.loc[:,'male_probs'] = male_probs

    test_df.loc[:,'gender_campaign'] = test_df.male_probs.apply(predictor.gender_campaign_selector)
    test_df.loc[:,'age_campaign'] = test_df.age_group.apply(predictor.age_campaign_selector)

    test_df.loc[:,'gender'] = test_df.gender.apply(lambda x: 'Female' if x == 0 else 'Male')
    test_df.loc[:,'age_group'] = test_df.age_group.apply(lambda x: '0-24' if x == 0 else '24-32' if x == 1 else '32+')

    test_df.to_csv('result.csv', index=False)
    
    return render_template('index.html', columns=test_df.columns, rows=test_df.values, is_download=True) 


@app.route('/download', methods=['POST'])
def download():
    path = "result.csv"
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
