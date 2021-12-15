#Reading the Titinic survival Data

#importer les données – utilisation de la librairie pandas
import pandas as pd
import numpy as np

train = pd.read_csv("train.csv")

test = pd.read_csv("test.csv")
#train = np.array (train[1:])

#vérification de la version de scikit-learn
import sklearn
print(sklearn.__version__)
#####################

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml

from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.decomposition import PCA


from sklearn.preprocessing import RobustScaler, OneHotEncoder

from sklearn.ensemble import RandomForestClassifier


#subdiviser les données en échantillons d'apprentissage et de test
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
np.random.seed(42)

train, test = train_test_split(train,test_size=300,random_state=1,stratify=train.Pclass)


#dimension du data frame
print(train.shape)
print(test.shape)

#affichage de 1er ligne
print(train.head())
print(test.head())

#information sur les variables train
print(train.info())

#information sur les variables test
print(test.info())



#vérifier la distribution absolue des classes
print(train.Pclass.value_counts())
print(test.Pclass.value_counts())

#la distribution relative
print(train.Pclass.value_counts(normalize=True))

#vérifier la distribution absolue des classes
print(test.Pclass.value_counts())

##Verification les distribution en apprentissage
print(train.Pclass.value_counts(normalize=True))
#Verification des distributions en test
print(test.Pclass.value_counts(normalize=True))

print(train.isnull().sum())
print(test.isnull().sum())

import matplotlib.pyplot as plt


import seaborn as sns  #setting seaborn default for plots
sns.set ()

corr_matrix = train.corr()
corr_matrix["Survived"].sort_values(ascending=False)
#Information sur la corrélation 
train.info()
train["Embarked"] = train ["Embarked"].fillna("S")
train.info()
sns.barplot(
    data= train,
    x='Sex',
    y='Survived')
plt.show()
sns.barplot(
    data= train,
    x='Embarked',
    y='Survived')
plt.show()
sns.barplot(
    data= train,
    x='Pclass',
    y='Survived')
plt.show()


#instanciation de l'arbre
from sklearn.datasets import load_iris

from sklearn.tree import DecisionTreeClassifier
arbreFirst = DecisionTreeClassifier(min_samples_split=20,min_samples_leaf=12)
#construction de l'arbre
arbreFirst.fit(X = train.iloc[:,:1], y = train.Pclass)



#affichage graphique de l'arbre - depuis sklearn 0.21

from sklearn.tree import plot_tree

#affichage plus grand pour une meilleure lisibilité
import matplotlib.pyplot as plt
plt.figure(figsize=(100,100))
plot_tree(arbreFirst,feature_names = list(train.columns[:-1]),filled=True)
plt.show()


#La méthode "predict" permet de tester l'entrainement de notre algorithme
#prédiction sur l'échantillon test
predFirst = arbreFirst.predict(X=test.iloc[:,:1])
#distribution des predictions
#numpy traitement les tableaux
import numpy
print(numpy.unique(predFirst,return_counts=True))

#matrice de confusion
#Pour évaluer l'entrainement de l'algorithme, il suffira de comparer y_pred et y_test
#On pourra obtenir le pourcentage de réponses correctes en utilisant la méthode "metrics.accuracy_score" 
from sklearn import metrics
print(metrics.confusion_matrix(test.Pclass,predFirst))
#taux de reconnaissance ((15+11+103)/300)
print(metrics.accuracy_score(test.Pclass,predFirst))

#taux d'erreur – (10+8)/300
print(1.0 - metrics.accuracy_score(test.Pclass,predFirst))


def bar_chart (feature):
    survived= train [train ['survived']==1][feature].value_counts()
    dead= train [train ['survived']==0][feature].value_counts()
    df = pd .DataFrame([survived,dead])
    df.index= ['Survived,Dead']
    df.plot(kind='bar', stecked=True, figsize=(10))    
    bar_chart('Sex')

#Préparation des données "train"

q= ['PassengerId', 'Name','Ticket','Cabin','SibSp','Parch','Age']
train_set = train.drop(q,axis=1)
train_set.head()
#"test"
z= [ 'Name','Ticket','Cabin','SibSp','Parch','Age']
test_set = test.drop(z,axis=1)
test_set.head()

mean=test_set["Fare"].mean()
test_set["Fare"] = test_set["Fare"].fillna(mean)
test_set.info()
#les propriétés non numériques en propriétés numériques.
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
#train
train_set.iloc[:,2] = labelencoder.fit_transform(train_set.iloc[:,2].values)
train_set.iloc[:,4] = labelencoder.fit_transform(train_set.iloc[:,2].values)
#test
test_set.iloc[:,2] = labelencoder.fit_transform(test_set.iloc[:,2].values)
test_set.iloc[:,4] = labelencoder.fit_transform(test_set.iloc[:,2].values)
train_set.info()
#Train data split
#Split train data into X_train, X_test, y_train, y_test.
X = train_set.iloc[:, 1:5].values
Y = train_set.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.4, random_state=4)
#affichage x,y train et test
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
######

#train_test_split permet d'effectuer cette séparation très simplement
######
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2, random_state=1)
######### KNN ########
#construire notre modèle KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


acc=[]
for i in range(1,20):
    knn= KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    yhat= knn.predict(X_test)
    acc.append(metrics.accuracy_score(y_test,yhat))
    print("For k = ",i," : ",accuracy_score(y_test,yhat))
 
plt.figure(figsize=(8,6))
plt.plot(range(1,20),acc, marker = "o")
plt.xlabel("Value of k")
plt.ylabel("Accuracy Score")
plt.title("Finding the right k")
plt.xticks(range(1,20))
plt.show()
    
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

#Faire des prédictions sur des données hors échantillon
test_set.info()


######### arbre de desision########
#méthode "fit" permet de procéder à l'entrainement de l'algorithme 

from sklearn.tree import DecisionTreeClassifier


depth = [];

for i in range(1,8):
    clf_tree = DecisionTreeClassifier(criterion="entropy", random_state = 100, max_depth = i)
    clf_tree.fit(X_train,y_train)
    yhat = clf_tree.predict(X_test)
    depth.append(accuracy_score(y_test,yhat))
    print("For max depth = ",i, " : ",accuracy_score(y_test,yhat))

plt.figure(figsize=(8,6))
plt.plot(range(1,8),depth,color="red", marker = "o")
plt.xlabel("Depth of Tree")
plt.ylabel("Accuracy Score")
plt.title("Finding the right depth with highest accuracy")
plt.xticks(range(1,8))
plt.show() 


########SVM##########
#support vector classifier (svc)
#Import svm model
from sklearn import svm

#Créer un classificateur svc
clf = svm.SVC(kernel='linear') # Linear Kernel

#Entraîner le modèle à l'aide des ensembles d'entraînement
#méthode "fit" permet de procéder à l'entrainement de l'algorithme 
clf.fit(X_train, y_train)

#Prédire la réponse pour l'ensemble de données de test
y_pred = clf.predict(X_test)

# Model Accuracy: à quelle fréquence le classificateur est-il correct ?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Model Precision: quel pourcentage de tuples positifs sont étiquetés comme tels ?
print("Precision:",metrics.precision_score(y_test, y_pred))

# Model Recall: quel pourcentage de tuples positifs sont étiquetés comme tels ?
print("Recall:",metrics.recall_score(y_test, y_pred))
