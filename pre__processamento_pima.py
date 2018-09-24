import pandas as pd
base = pd.read_csv('pima.csv')
base.describe()

#Localiza linhas de acordo com valores das colunas
#base.loc[base["age"] < 22 ]

#apaga uma coluna
#base.drop('age', 1, inplace=True)

#apagar somente algumas linhas
#base.drop(base[base.age < 22].index, inplace=True)
#base.mean()
#base['age'].mean()
#base['age'][base.age < 30].mean()
#alterar valores de uma linha
#base.loc[base.age < 22, 'age'] = 1

#localizar registros nao preenchidos
#pd.isnull(base['age'])
#base.loc[pd.isnull(base['age'])]

previsores = base.iloc[:, 0:8].values
classe = base.iloc[:, 8].values

from sklearn.preprocessing import Imputer
imputer = Imputer()
imputer.fit(previsores[:, 0:8])
previsores[:, 0:8] = imputer.transform(previsores[:,0:8])

# Pradoonizacao -> x = x - media(x) / desvio_padrao(x)
# nomalização -> x = x - minimo(x) / maximo(x) - minimo(x)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores[:,0:8])