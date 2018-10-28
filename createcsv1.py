import numpy as np
import pandas as pd

print
print('Reading file...')
ccdata = pd.read_csv('creditcard.csv')

fraudtr = ccdata.loc[ccdata['Class'] == 1,:]
normaltr = ccdata.loc[ccdata['Class'] == 0,:]
fraudcnt = fraudtr.shape[0]
normalcnt = normaltr.shape[0]
print('fraudlent transactions count = {}'.format(fraudcnt))
print('normal transactions count = {}'.format(normalcnt))
print('percentage of fraudulent transactions {:.3f}%'.format(fraudcnt*100./(normalcnt+fraudcnt)))

seed=0

print
print('Upsampling data...')
upsample_total = int(fraudcnt/0.05)
normal_count = upsample_total-fraudcnt
normal_data = normaltr.sample(n=normal_count, random_state=seed)
upsample_data = pd.concat([fraudtr, normal_data])
print('upsample size = {}'.format(upsample_total))

seed = 0
labelcolumn = ['Class']
print("Label column: %s" % (labelcolumn[0]))
feature_columns = ccdata.columns[:-1]
print(feature_columns)
print('feature columns count: {}'.format(len(feature_columns)))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
print
print('Splitting data...')
splitdata = []
Xy = upsample_data
print('dataset size {}: {}, %fraud: {:.3f}'.format(len(splitdata)+1, Xy.shape[0],Xy['Class'].value_counts(sort=True).map(lambda x: x*100./Xy.shape[0])[1]))
X = Xy.loc[:, feature_columns]
y = Xy.loc[:, labelcolumn]
scaler = MinMaxScaler(feature_range=(0,1))
X_scaled = scaler.fit_transform(X)
X_valtr, X_test, y_valtr, y_test = train_test_split(X_scaled, y, stratify=y, random_state=seed)
X_train, X_valid, y_train, y_valid = train_test_split(X_valtr, y_valtr, stratify=y_valtr, random_state=seed)
splitdata.append({
    'X_train': X_train,
    'y_train': y_train,
    'X_valid': X_valid,
    'y_valid': y_valid,
    'X_test': X_test,
    'y_test': y_test
})

print("Printing shapes...")
for key in splitdata[0].keys():
    print key, splitdata[0][key].shape

for i in ['test', 'train', 'valid']:
    Xfname  = 'ccX'+ i + '.csv'
    yfname  = 'ccy'+ i + '.csv'
    with open(Xfname, 'wb') as xf:
        xf.write(','.join(feature_columns)+'\n')
        np.savetxt(xf, splitdata[0]['X_'+i], fmt='%.10f', delimiter=',')
    with open(yfname, 'wb') as yf:
        yf.write(labelcolumn[0]+'\n')
        np.savetxt(yf, splitdata[0]['y_'+i], fmt='%d', delimiter=',')

print(
  """
  Then do this...
  $ for i in test train valid
  do  
  paste -d"," ccX$i.csv ccy$i.csv > cc$i.csv
  done
  """
)