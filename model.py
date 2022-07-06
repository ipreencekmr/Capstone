import pandas as pd
import pickle
import joblib

class DecileRange:
    def __init__(self):
        self.minValue = 0
        self.maxValue = 0
        self.ks_df = pd.read_csv('KS_Stats.csv')
        self.top_deciles = self.ks_df.iloc[:3]
        self.bottom_deciles = self.ks_df.iloc[7:]
        
    def male_campaign_range(self):
        dRange = DecileRange()
        dRange.minValue = min(self.top_deciles['MIN_PROB'])
        dRange.maxValue = max(self.top_deciles['MAX_PROB']) 
        return dRange

    def female_campaign_range(self):
        dRange = DecileRange()
        dRange.minValue = min(self.bottom_deciles['MIN_PROB'])
        dRange.maxValue = max(self.bottom_deciles['MAX_PROB']) 
        return dRange
        
        
class Predictor:
    def __init__(self):
        self.gender_model = pickle.load(open('gender_model.pkl', 'rb'))
        self.age_model = pickle.load(open('age_model.pkl', 'rb'))
        self.scaler = joblib.load('scaler.gz')
        self.columns = self.gender_model.get_booster().feature_names
        
        dr = DecileRange()
        self.mcr = dr.male_campaign_range()
        self.fcr = dr.female_campaign_range()

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
    
    def predict_gender_prob(self):
        scaled_df = self.transform_df()
        gender_predictions = self.gender_model.predict_proba(scaled_df)
        return gender_predictions[:,1]
    
    def gender_campaign_selector(self, x):
        if x >= self.mcr.minValue and x <= self.mcr.maxValue:
            return 'Campaign 3'
        if x >= self.fcr.minValue and x <= self.fcr.maxValue:
            return 'Campaign 1 & 2'
        return 'NA'
    
    def age_campaign_selector(self, x):
        if x == 0:
            return 'Campaign 4'
        elif x == 1:
            return 'Campaign 5'
        else:
            return 'Campaign 6'