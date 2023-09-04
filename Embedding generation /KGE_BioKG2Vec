
# Import libraries 

import pandas as pd
import pickle
import os
import numpy as np
import random
import networkx as nx
from tqdm import tqdm
import torch

#BIOKG2VEC IMPORTS
from gensim.models import Word2Vec


import random
from tqdm import tqdm
from collections import Counter

def KRW(Node, Graph, NodeAttributeName, EdgeAttributeName, DictOfProb, Iterations=5, Depth=70, restart='False', EdgeType='True', verbose='False', directed='False'):
    """This RW simulates the biological dogma in which drug - protein - function - disease path is prioritized
    - the dictionary of probabilities is composed of series of tuples: probabilities in which the tuple is the transition between node types and
      the number is the probability increase that I want to get to go to that specific node type to the other
    - NodeAttributeName name is a string with the name in which is stored the node type
    - EdgeAttributeName is a string with the name in which is stored the edge type"""
    RandomWalks = []
    for iteration in range(Iterations):
        Pointer = Node
        if verbose == 'True':
            print(f'\n Starting at {iteration} starting node is: {Pointer}')

        RandomWalk = []
        RandomWalk.append(Pointer)
        n = 0
        while n < Depth:
            PossiblePaths = list(Graph.neighbors(Pointer))  # Using neighbors method for regular graph

            PossiblePathsNodes = PossiblePaths

            if (len(PossiblePaths) == 0) & (n == 0):
                if verbose == 'True':
                    print(f"First node doesn't have adjacent edges, can't start the walk here: {Pointer}")
                n = 10

            elif len(PossiblePaths) > 0:
                TypeOfNode = Graph.nodes[Pointer][NodeAttributeName]
                if verbose == 'True':
                    print('Currently on node:', Pointer, TypeOfNode)
                if TypeOfNode in [e[0] for e in DictOfProb.keys()]:
                    TypeOfNodeToPrioritize = list(DictOfProb)[[e[0] for e in DictOfProb.keys()].index(TypeOfNode)][1]
                    if verbose == 'True':
                        print(f'There are nodes to prioritize: {TypeOfNodeToPrioritize}')
                    TypesOfNeighbors = [Graph.nodes[n][NodeAttributeName] for n in PossiblePathsNodes]
                    NumberOfNeighbors = len(set(TypesOfNeighbors))
                    Neighbors = Counter(TypesOfNeighbors)
                    if verbose == 'True':
                        print('I have these types of Neighbors so see if weights are calculated in an appropriated way', Neighbors)
                    Weight = list(DictOfProb.values())[[e[0] for e in DictOfProb.keys()].index(TypeOfNode)]
                    NumberOfEdgesOfInterest = Neighbors[TypeOfNodeToPrioritize]
                    Weights = [1 + Weight / NumberOfEdgesOfInterest if Type == TypeOfNodeToPrioritize else 1 for Type in TypesOfNeighbors]
                    if verbose == 'True':
                        print(f'These are the Number of edges that lead to {TypeOfNodeToPrioritize}: {NumberOfEdgesOfInterest} so the calculated Weights are {Weights}')
                else:
                    if verbose == 'True':
                        print(f'No prioritization since the current node is {TypeOfNode}')
                    NOfNeighbors = len(PossiblePaths)
                    Weights = [1 for _ in range(NOfNeighbors)]

                path = random.choices(PossiblePaths, weights=Weights)
                Edge = None  # Regular graphs do not have explicit edge labels
                Pointer = path[0]
                if EdgeType == 'True':
                    RandomWalk.append(Edge)
                RandomWalk.append(Pointer)
                if verbose == 'True':
                    print('Node and edge stored, new node:', Pointer)
                n += 1

            else:
                if verbose == 'True':
                    print('NO NEIGHBORS!!!')
                if restart == 'True':
                    if verbose == 'True':
                        print('Restarting')
                    Pointer = Node
                    n += 1
                    if verbose == 'True':
                        print(f'I restarted so now the starting point is {Pointer}')
                else:
                    if verbose == 'True':
                        print(f'I didn\'t restart and I am just continuing, so the node I am on is {Pointer}')
                    n += 1
            if verbose == 'True':
                print(f'After {n} number of steps, I am finishing an iteration and I built a random walk that is {len(RandomWalk)} ')
        RandomWalks.append(RandomWalk)
    return RandomWalks

#BIOKG2VEC
def LoadData(kg_path):
    with open(kg_path, 'rb') as f:
        kg = pickle.load(f)
    return kg

def RunBioKG2Vec(kg):
    probabilities={
        ('drug','protein'):100,
        ('protein','biological_function'):0,
        ('biological_function','indication'):1000
    }

    all_nodes=list(kg.nodes)
    random_walks = []
    print(" |================>   WALKING")
    for n in tqdm(all_nodes):
        biorw=KRW(n, kg, Iterations=5, Depth=50,
                NodeAttributeName='node_type',
                EdgeAttributeName='rel_type',
                DictOfProb=probabilities,
                directed='True')
        random_walks.extend(biorw)
    model = Word2Vec(window = 5, sg = 1, hs = 0,
                 negative = 5, # for negative sampling
                 alpha=0.03, min_alpha=0.0007,
                 seed = 14,workers=2)

    model.build_vocab(random_walks, progress_per=2)

    print(" |================>   TRAINING THE MODEL")

    model.train(random_walks, total_examples = model.corpus_count, epochs=100, report_delay=1)
    Id2Vec=dict(zip(model.wv.index_to_key,model.wv.vectors))
    return Id2Vec

kg = LoadData(kg_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

Id2Vec = RunBioKG2Vec(kg)

with open(output_path, 'wb') as f:
    pickle.dump(Id2Vec, f)
    print("BioKG2Vec embeddings have been generated and saved.")
