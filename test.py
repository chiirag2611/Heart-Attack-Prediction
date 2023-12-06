import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier


input_df = pd.read_csv("./heart_failure_clinical_records_dataset.csv")


heart_raw = pd.read_csv('heart_failure_clinical_records_dataset.csv')

heart_raw['sex'] = np.where(heart_raw['sex'] == 1, 'Male','Female')

heart = heart_raw.drop(columns = ['DEATH_EVENT'])
data = pd.concat([input_df,heart], axis = 0)
df = data.copy()
df1 = data.copy()



def set_cpk(creatinine_phosphokinase):
    if creatinine_phosphokinase >=10 and creatinine_phosphokinase <= 120:
        return "Normal"
    else:
        return "High"
    


def set_eject_fract(ejection_fraction):
    if ejection_fraction <= 35:
        return "Low"
    elif ejection_fraction > 35 and ejection_fraction <= 49:
        return "Below_Normal"
    elif ejection_fraction > 50 and ejection_fraction <= 75:
        return "Normal"
    else:
        return "High"



def set_platelets(platelets):    
    if platelets < 157000:
        return "Low"
    elif platelets >=157000 and platelets <= 371000:
        return "Normal"
    else:
        return "High"
        
    if platelets < 135000:
        return "Low"
    if platelets >= 135000 and platelets <= 317000:
        return "Normal"
    else:
        return "High"


def set_sodium(ss):
    serum_sodium = int(ss)
    if serum_sodium < 135:
        return "Low"
    elif serum_sodium >=135 and serum_sodium <= 145:
        return "Normal"
    else:
        return "High"

def set_creatinine(sc):
    serum_creatinine = float(sc)
    if serum_creatinine >=0.5 and serum_creatinine <= 1.1:
        return "Normal"
    else:
        return "High"
    if serum_creatinine >=0.6 and serum_creatinine <= 1.2:
        return "Normal"



df2 = df1.copy()
df1 = pd.get_dummies(df1,columns = ['sex'], drop_first = True)

df2 = pd.get_dummies(df2,columns = ['sex'], drop_first = True)



col = ['age','creatinine_phosphokinase','ejection_fraction',
       'platelets','serum_creatinine','serum_sodium','time',
       'anaemia','diabetes','high_blood_pressure','smoking',
       'sex_Male']

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler

col_trans = ColumnTransformer(remainder='passthrough',
                              transformers = [('scaler',MinMaxScaler(),
                              [0,2,4,6,7,8,10])])
trans = col_trans.fit_transform(df1)
trans = col_trans.transform(df2)


df_ = trans[:len(input_df)]


load_clf = pickle.load(open('model.pkl', 'rb'))
   

def test_set_creatinine():
    assert  str(set_creatinine(0.1)) == "High"
    assert  str(set_creatinine(0.7)) == "Normal"

def test_set_sodium():
    assert str(set_sodium(146)) == "High"
    assert str(set_sodium(138)) == "Normal"
    assert str(set_sodium(1)) == "Low"

def test_set_cpk():
    assert str(set_cpk(9)) == "Low"
    assert str(set_cpk(20)) == "Normal"


