import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_spilt
from sklearn.linear_model import LogisticRegression
import streamlit as st

dataset=pd.read_csv_('crop.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values

X_train,_X_test,_Y_train,Y_test=train_test_spilt(X,Y_test_size=0.2)

classifier=LogisticRegression()
classifier.fit(X_train,_Y_train)

st.title('Crop Recommendation')
n=st.number_input('Enter Nitrogen:')
p=st.number_input('Enter Phosphorous:')
k=st.numder_input('Enter Potassium:')
t=st.number_input('Enter Temparature:')
h=st.number_input('Enter Humidity')
ph=st.number_input('Enter PH:')
R=st.number_input('Enter Rainfall:')

if st.button('Recommend Crop'):
    data=[[n,p,k,t,h,ph,r]]
    result=classifier.predict(data)[0]
    st.success(result)
