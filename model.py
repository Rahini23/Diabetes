import pandas as pd
import numpy as np

df = pd.read_csv('diabetes.csv')
X=df.drop(['Outcome'],axis=1)
y=df['Outcome']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state = 42 )


from sklearn.linear_model import LogisticRegression
LogReg = LogisticRegression()
LogReg.fit(X,y)

import pickle
pickle.dump(LogReg, open('model_log.pkl','wb'))
model_log = pickle.load(open('model_log.pkl','rb'))
model_log= pickle.load(open('model_log.pkl','rb'))

