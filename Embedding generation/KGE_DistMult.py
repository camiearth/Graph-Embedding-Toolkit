# Import libraries 
import pandas as pd
import pickle
import os
import numpy as np
import argparse
import random
import networkx as nx
from tqdm import tqdm
import torch

# TRANSE DISTMULT IMPORTS
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline

parser=argparse.ArgumentParser(description='Create embeddings given a knowledge graph')
parser.add_argument('-k','--KnowledgeGraph',help='path of the KG in pickle format (networkx Digraph Object)')
parser.add_argument('-m','--model',help='model to use to produce embeddings possible values are: DistMult', type=str)
parser.add_argument('-e','--epochs',help='number of epochs to train the model for', type=int)
parser.add_argument('-o','--output',help='path in which to save the embeddings')

def LoadData():
    with open(args.KnowledgeGraph,'rb') as f:
        kg = pickle.load(f)

    return kg

# DISTMULT
def RunDistmult(kg):
    # CREATE MAPPING DICTONARIES
    Id2Node = dict(enumerate(list(kg.nodes)))
    Node2Id = {v: k for k, v in Id2Node.items()}
    Id2Int = dict(enumerate(list(set([edge[2]['rel_type'] for edge in kg.edges(data=True)]))))
    Int2Id = {v: k for k, v in Id2Int.items()}

    # CREATE TRIPLES FACTORY
    edgelist = list(nx.to_edgelist(kg))
    set_of_triples = [[Node2Id[node1], Int2Id[interaction['rel_type']], Node2Id[node2]] for (node1, node2, interaction) in edgelist]
    set_of_triples_Long = torch.LongTensor(set_of_triples)
    tf = TriplesFactory(set_of_triples_Long, entity_to_id=Node2Id, relation_to_id=Int2Id)
    training, testing, validation = tf.split([.8, .1, .1])

    result = pipeline(
        training=training,
        testing=testing,
        validation=validation,
        model='DistMult',
        model_kwargs=dict(embedding_dim=100),
        epochs=args.epochs)
    model = result.model
    entity_tensor = model.entity_representations[0]().detach().cpu().numpy()
    Id2Vec = dict(zip(Id2Node.values(), entity_tensor))
    return Id2Vec

def Main():
    kg = LoadData()

    if args.model == "DistMult":
        Id2Vec = RunDistmult(kg)
        with open(args.output, 'wb') as f:
            pickle.dump(Id2Vec, f)

if __name__ == '__main__':
    Main()
