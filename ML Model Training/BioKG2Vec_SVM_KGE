# import libraries
import pandas as pd
import pickle
import ast
import os
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Set the argument values
args = {
    'model': 'SVM',
    'parameters': "{'C': 10, 'gamma': 0.1, 'kernel': 'rbf'}",
    'features': 'Concatenation',
    'embeddings': 'pickle file',
    'output': 'pickle file'
}

def LoadId2Vec(path):
    with open(path, 'rb') as f:
        Id2Vec = pickle.load(f)
    return Id2Vec

def CreateFeatures(Id2Vec):
    gda = pd.read_csv('gold_standard.csv')
    # ADD EMBEDDINGS (FEATURES) TO THE TRAINING SET
    gda['indication'] = gda['indication'].astype(str)
    gda['drug'] = gda['drug'].astype(str)
    gda['ind_emb'] = list(map(Id2Vec.get, gda['indication'].tolist()))
    gda['drug_emb'] = list(map(Id2Vec.get, gda['drug'].tolist()))
    gda.dropna(inplace=True)
    if args['features'] == 'Concatenation':
        gda['Concatenation'] = gda.apply(lambda x: np.concatenate((x['drug_emb'], x['ind_emb'])), axis=1)

    return gda

def Main():
    id2vec = LoadId2Vec(args['embeddings'])
    TrainSet = CreateFeatures(id2vec)
    TrainSet.label = TrainSet.label.astype(int)

    # DEFINE FEATURES AND LABELS
    X = np.array(TrainSet['Concatenation'].tolist())  # Use the 'Concatenation' method
    y = np.array(TrainSet['label'].tolist())

    if args['model'] == 'SVM':
        if args['parameters']:
            parameters = ast.literal_eval(args['parameters'])
            clf = SVC(**parameters, probability=True)
        else:
            clf = SVC(probability=True)

    print(f'SELECTED MODEL = {str(clf)}')

    # SPLIT THE DATASET INTO TRAINING AND TESTING SETS
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training the model...")
    # FIT THE MODEL ON THE TRAINING SET
    clfit = clf.fit(X_train, y_train)
    print("Model training complete.")
