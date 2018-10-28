from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer
import numpy as np
import pandas as pd

seed = 0
n = 8500
data = pd.read_csv('sign_mnist_train.csv')
ndata = data.sample(n=int(n), random_state=seed)

feature_columns = ndata.columns[1:]
label_column = 'label'

X = ndata.loc[:, feature_columns]
y = ndata.loc[:, label_column]
scaler = MinMaxScaler(feature_range=(0,1))
X_scaled = scaler.fit_transform(X)
X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, y, stratify=y, 
                      test_size=1./3., random_state=seed)

test = pd.read_csv('sign_mnist_test.csv')

feature_columns = test.columns[1:]

X_test = test.loc[:, feature_columns]
y_test = test.loc[:, label_column]
scaler = MinMaxScaler(feature_range=(0,1))
X_test_scaled = scaler.fit_transform(X_test)

for i in ['train', 'valid', 'test']:
    if i == 'train':
      Xs = X_train
      ys = y_train
    if i == 'valid':
      Xs = X_valid
      ys = y_valid
    if i == 'test':
      Xs = X_test_scaled
      ys = y_test

    Xfname  = 'slX'+ i + '.csv'
    yfname  = 'sly'+ i + '.csv'
    with open(Xfname, 'wb') as xf:
        xf.write(','.join(feature_columns)+'\n')
        np.savetxt(xf, Xs, fmt='%.10f', delimiter=',')
    with open(yfname, 'wb') as yf:
        yf.write(label_column+'\n')
        np.savetxt(yf, ys, fmt='%d', delimiter=',')

print("Then do this...")
print(
  """
  $ for i in test train valid; 
  do 
  sed -i -e 's/^# //' slX$i.csv; 
  sed -i -e 's/^# //' sly$i.csv; 
  paste -d"," slX$i.csv sly$i.csv > sl$i.csv; 
  done
  """
)