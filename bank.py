# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 17:35:42 2020

@author: Sahil Bagwe
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler,MinMaxScaler
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,matthews_corrcoef,precision_score,recall_score
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.utils import class_weight

df = pd.read_csv('C:/Users/Sahil Bagwe/Desktop/Python/dataset/Bank/BankChurners.csv')
df = df.drop(df.columns[21:23],axis=1)
df=df.drop('CLIENTNUM',axis=1)  

df['Gender'].replace('M',1,inplace = True)
df['Gender'].replace('F',0,inplace = True)
 
df['Education_Level'].replace('Unknown',0,inplace = True)
df['Education_Level'].replace('Uneducated',1,inplace = True)
df['Education_Level'].replace('High School',2,inplace = True)
df['Education_Level'].replace('College',3,inplace = True)
df['Education_Level'].replace('Graduate',4,inplace = True)
df['Education_Level'].replace('Post-Graduate',5,inplace = True)
df['Education_Level'].replace('Doctorate',6,inplace = True)

df['Marital_Status'].replace('Unknown',0,inplace = True)
df['Marital_Status'].replace('Single',1,inplace = True)
df['Marital_Status'].replace('Married',2,inplace = True)
df['Marital_Status'].replace('Divorced',3,inplace = True)

df['Card_Category'].replace('Blue',0,inplace = True)
df['Card_Category'].replace('Gold',1,inplace = True)
df['Card_Category'].replace('Silver',2,inplace = True)
df['Card_Category'].replace('Platinum',3,inplace = True)


df['Income_Category'].replace('Unknown',0,inplace = True)
df['Income_Category'].replace('Less than $40K',1,inplace = True)
df['Income_Category'].replace('$40K - $60K',2,inplace = True)
df['Income_Category'].replace('$60K - $80K',3,inplace = True)
df['Income_Category'].replace('$80K - $120K',4,inplace = True)
df['Income_Category'].replace('$120K +',5,inplace = True)

df['Attrition_Flag'].replace('Existing Customer',0,inplace = True)
df['Attrition_Flag'].replace('Attrited Customer',1,inplace = True)


#%%

x = df[df.columns[1:20]]
y = df[df.columns[0]]
#%%

rs = RobustScaler()
x =rs.fit_transform(x)

#%%
a = y.value_counts()
ratio = a[1]/(a[1]+a[0])
weights = [ratio, 1-ratio]

   
#%%
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.3, random_state=103)

cw = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)

model = Sequential()
model.add(Dense(19,activation="sigmoid"))
model.add(Dense(10,activation="sigmoid"))
model.add(Dense(1))
model.compile(optimizer='rmsprop',loss = "binary_crossentropy",metrics=["BinaryAccuracy"],loss_weights=weights)

history = model.fit(x=X_train,y=Y_train,epochs=100, class_weight = {0:cw[0], 1:cw[1]})

predictions = model.predict_classes(X_test)    

#%%

cm = confusion_matrix(Y_test, predictions)
mcc = matthews_corrcoef(Y_test,predictions)      
print('\n')
print('Neural Network Accuracy: ', accuracy_score(Y_test,predictions))
print('Neural Network Recall score: ', recall_score(Y_test,predictions))
#%%

ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g')
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Neural Network Confusion Matrix')