import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve


# Importation de la base de données

data = pd.read_csv('database.csv')
data_c = data.dropna(axis=0)
data_stat = data_c.describe()


# Prétraitement des données
# Remplacer les valeurs des parametres qualitatives 
# par des valeurs quantitatives

data['Sex'].replace(['M','F'],[0,1],inplace=True)
# M : Masculin 0 / F : Féminin 1
data['ChestPainType'].replace(['TA','ATA','NAP','ASY'],[0,1,2,3],inplace=True)
# TA : Angine Typique 0 / ATA : Angine ATypique 1 
# NAP : Douleur Non Angineuse 2 / ASY : ASYmpomatique 3 
data['RestingECG'].replace(['Normal','ST','HVG'],[0,1,2],inplace=True)
# Normal : normal 0 / ST : anomalie de l'onde ST-T 1
# HVG : Hypertrophie Ventriculaire Gauche 2
data['ExerciseAngina'].replace(['N','Y'],[0,1],inplace=True)
# N : Non 0 / Y : Oui 1
data['ST_Slope'].replace(['Up','Flat','Down'],[0,1,2],inplace=True)
# Up : ascendant 0 / Flat : plat 1 / Down : descendant 2


parameters = data[['Age','ExerciseAngina','ST_Slope','Sex','ChestPainType','RestingECG','Oldpeak','RestingBP','Cholesterol','MaxHR','FastingBS']]
predict_value = np.array(data['HeartDisease']).reshape((len(data),1))


# Diviser la base de données en données d'entrainement et en données de test

parameters_train,parameters_test,predict_value_train,predict_value_test = train_test_split(parameters,predict_value,test_size = 0.2,random_state = 5)


# Création et entrainement du modèle

model1 = KNeighborsClassifier(n_neighbors = 1,metric ='euclidean')
model1.fit(parameters_train,predict_value_train)
R1_test = model1.score(parameters_test,predict_value_test)


# Optimisation du modèle en cherchant les valeurs optimales des paramètres
# n_neighbors,metric de la fonction KNeighborsClassifier()

param_grid = {'n_neighbors' : np.arange(1,30), 'metric' : ['euclidean','manhattan','minkowski']}
grid = GridSearchCV(KNeighborsClassifier(),param_grid,cv=5)
grid.fit(parameters_train,predict_value_train)
MS = grid.best_score_
M_param = grid.best_params_

model2 = grid.best_estimator_
R2_test = model2.score(parameters_test,predict_value_test)


# Evaluation de la qualité de classification

A = confusion_matrix (predict_value_test,model2.predict(parameters_test)) 



def predire(model2,Age,ExerciseAngina,ST_Slope,Sex,ChestPainType,RestingECG,Oldpeak,RestingBP,Cholesterol,MaxHR,FastingBS):
    x = np.array([Age,ExerciseAngina,ST_Slope,Sex,ChestPainType,RestingECG,Oldpeak,RestingBP,Cholesterol,MaxHR,FastingBS]).reshape(1,11)
    pred_val = model2.predict(x)
    proba = model2.predict_proba(x)
    p = round(proba[0][1]*100,2)
    pn = round(proba[0][0]*100,2)
    if pred_val == 1 :
        print('Cette personne a une maladie cardiaque avec une probabilité de : ',p,'%')
    else :
        print('Cette personne est normale avec une probabilité de : ',pn,'%')

# Exemples


# atteinte d une maladie

# predire(model2,76,0,2,1,1,2,1.5,180,300,120,0)
# predire(model2,21,1,1,0,3,0,0,120,100,120,0)
# predire(model2,30,1,1,0,3,0,1.5,200,60,80,0)
# predire(model2,30,1,1,0,3,0,1.5,200,60,180,0)

# En etat normale

# predire(model2,30,1,1,0,3,0,1,100,170,200,1)
# predire(model2,60,1,1,0,3,0,0.5,120,150,170,1)
# predire(model2,60,1,1,0,3,0,0.5,120,150,140,1)
