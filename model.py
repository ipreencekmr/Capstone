import pandas as pd
import pickle
import joblib

class Predictor:
    def __init__(self):
        self.gender_model = pickle.load(open('gender_model.pkl', 'rb'))
        self.age_model = pickle.load(open('age_model.pkl', 'rb'))
        self.scaler = joblib.load('scaler.gz')
        self.columns = self.gender_model.get_booster().feature_names

    def load_data(self, test_df):
        self.X_test = test_df

    def transform_df(self):
        values = self.scaler.transform(self.X_test)
        tdf = pd.DataFrame(values)
        tdf.columns = self.columns
        return tdf

    def predict_age(self):
        scaled_df = self.transform_df()
        age_predictions = self.age_model.predict(scaled_df)
        return age_predictions
    
    def predict_gender(self):
        scaled_df = self.transform_df()
        gender_predictions = self.gender_model.predict(scaled_df)
        return gender_predictions
