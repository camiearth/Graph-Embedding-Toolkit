# Import libraries
import pandas as pd
import pickle
import os
import numpy as np
import biomapy as bp
import argparse
from tqdm import tqdm

def Loaddata(path):
    with open(path, 'rb') as f:
        Id2Vec = pickle.load(f)
    return Id2Vec

def Main():
    # Load the embeddings
    Id2Vec = Loaddata(embeddings_path)

    # Load the model
    with open(model_path, 'rb') as handle:
        clf = pickle.load(handle)

    print('BUILDING THE DATASET')

    # Load the CUI codes from the file
    cui_data = pd.read_csv(cui_file_path, sep='\t', usecols=['V1'])
    cui_codes = cui_data['V1'].tolist()

    print('MAKING PREDICTIONS')


    all_predictions = []

    for idx, cui_code in enumerate(cui_codes, start=1):
        print(f"Processing CUI {idx}/{len(cui_codes)} - CUI Code: {cui_code}")

        disease_embedding = Id2Vec[cui_code]

        drugs_with_embeddings = [drug for drug in Id2Vec.keys() if isinstance(drug, str) and drug.startswith("DB") and drug[2:].isnumeric()]

        data = pd.DataFrame(zip(drugs_with_embeddings,
                                [Id2Vec[drug] for drug in drugs_with_embeddings],
                                [disease_embedding for _ in range(len(drugs_with_embeddings))]),
                            columns=['drug', 'drug_emb', 'ind_emb'])

        if features == 'Concatenation':
            data[features] = data.apply(lambda x: np.concatenate((x['drug_emb'], x['ind_emb'])), axis=1)

        data = data.dropna()

        pred = clf.predict(data[features].tolist())
        predprob = clf.predict_proba(data[features].tolist())

        data['CUI'] = [cui_code for _ in range(data.shape[0])]
        data['Predicted'] = pred
        data['Proba'] = predprob[:, 1]

        all_predictions.append(data)

    all_predictions_df = pd.concat(all_predictions, ignore_index=True)

    all_predictions_df.to_csv(output_path, index=False)


    print('Predictions saved to', output_path)


if __name__ == '__main__':
    Main()
