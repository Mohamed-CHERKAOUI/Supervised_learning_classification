import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix


def pre_traitement(database,colonne,val_qualitative,val_quantitative):
    db = database[colonne].replace(val_qualitative,val_quantitative,inplace=True)
    return db


def creer_modele(n_nei,m):
    model = KNeighborsClassifier(n_neighbors = n_nei,metric = m)
    return model

def entrainer_modele(modele,x,y):
    modele.fit(x,y)
    
def score_modele(modele,x,y):
    R = modele.score(x,y)
    return R

def cross_validation(x,y):
    param_grid = {'n_neighbors' : np.arange(1,30), 'metric' : ['euclidean','manhattan','minkowski']}
    grid = GridSearchCV(KNeighborsClassifier(),param_grid,cv=5)
    grid.fit(x,y)
    M_estimateur = grid.best_estimator_    
    return M_estimateur

def predire(modele,param):
    x = np.array(param).reshape(1,11)
    pred_val = modele.predict(x)
    proba = modele.predict_proba(x)
    p = round(proba[0][1]*100,2)
    pn = round(proba[0][0]*100,2)
    if pred_val == 1 :
        print('Cette personne a une maladie cardiaque avec une probabilité de : ',p,'%')
    else :
        print('Cette personne est normale avec une probabilité de : ',pn,'%')
        

def main(param):
    
    """
    Importation de la base de données
    """
    
    data = pd.read_csv('database.csv')
    data = data.dropna(axis=0)
    
    """
    Prétraitement des données :
    Remplacer les valeurs des parametres qualitatives par des valeurs quantitatives
    """
    
    pre_traitement(data,'Sex',['M','F'],[0,1])
    # M : Masculin 0 / F : Féminin 1
    pre_traitement(data,'ChestPainType',['TA','ATA','NAP','ASY'],[0,1,2,3])
    # TA : Angine Typique 0 / ATA : Angine ATypique 1 
    # NAP : Douleur Non Angineuse 2 / ASY : ASYmpomatique 3
    pre_traitement(data,'RestingECG',['Normal','ST','HVG'],[0,1,2])
    # Normal : normal 0 / ST : anomalie de l'onde ST-T 1
    # HVG : Hypertrophie Ventriculaire Gauche 2    
    pre_traitement(data,'ExerciseAngina',['N','Y'],[0,1])
    # N : Non 0 / Y : Oui 1
    pre_traitement(data,'ST_Slope',['Up','Flat','Down'],[0,1,2])
    # Up : ascendant 0 / Flat : plat 1 / Down : descendant 2
    
    """
    Création des paramètres et des valeurs cibles
    """
    
    parameters = data[['Age','ExerciseAngina','ST_Slope','Sex','ChestPainType','RestingECG','Oldpeak','RestingBP','Cholesterol','MaxHR','FastingBS']]
    predict_value = data['HeartDisease']
    
    """
    Diviser la base de données en données d'entrainement et en données de test
    """
    
    parameters_train,parameters_test,predict_value_train,predict_value_test = train_test_split(parameters,predict_value,test_size = 0.2,random_state = 5)
    
    """
    Création et entrainement du modèle
    """
    
    modele1 = creer_modele(1,'euclidean')
    entrainer_modele(modele1,parameters_train,predict_value_train)
    R1 = score_modele(modele1,parameters_test,predict_value_test)
    print('Le coefficient de detarmination du modele avant optimisation est : ',round(R1,3))
    
    """
    Optimisation du modèle en cherchant les valeurs optimales des paramètres
    n_neighbors,metric de la fonction KNeighborsClassifier()
    """
    modele2 = cross_validation(parameters_train,predict_value_train)
    R2 = score_modele(modele2,parameters_test,predict_value_test)
    print('Le coefficient de detarmination du modele apres optimisation est : ',round(R2,3))
    
    """
    Evaluation de la qualité de classification
    """
    
    M_confusion = confusion_matrix(predict_value_test,modele2.predict(parameters_test)) 
    print("la matrice de confusion : ")
    print(M_confusion)
    
    """
    Faire des prédictions
    """

    predire(modele2,param)    



if __name__ == '__main__':
    main([60,1,1,0,3,0,0.5,120,150,170,1])  


"""
Exemples


~~ Atteints d'une maladie

main([76,0,2,1,1,2,1.5,180,300,120,0])
main([21,1,1,0,3,0,0,120,100,120,0])
main([30,1,1,0,3,0,1.5,200,60,80,0])
main([30,1,1,0,3,0,1.5,200,60,180,0])

~~ En etat normale

main([30,1,1,0,3,0,1,100,170,200,1])
main([60,1,1,0,3,0,0.5,120,150,170,1])
main([60,1,1,0,3,0,0.5,120,150,140,1])
"""








