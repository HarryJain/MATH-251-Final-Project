#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 14:29:29 2022

@author: harry
"""


# Module imports
import pathlib
from os import path
import pandas as pd
import numpy as np
    

# Markov chain class to model transitions
class MarkovChain(object):
    # Initialize the object with a transition probability matrix encoded as a nested dictionary
    def __init__(self, states, state_indices, transition_matrix):
        self.states = states
        self.state_indices = state_indices
        self.transition_matrix = transition_matrix
        
    # Get a random next state starting from src
    def next_state(self, src):
        index = np.random.choice(len(self.states), p = [ self.transition_matrix[self.state_indices[src]][dst_ind] for dst_ind in range(len(self.states)) ])
        return self.states[index]
    
    # Create a chain of length n starting from src
    def generate_states(self, src, n):
        future_states = []
        for i in range(n):
            dst = self.next_state(src)
            future_states.append(dst)
            src = dst
        return future_states
    
    # Simulate a chain from src to dst
    def simulate_chain(self, src, dst):
        state = src
        chain = []
        while state != dst:
            state = self.next_state(state)
            chain.append(state)
        return chain


# Store the project directory and data directory as constant strings
#PROJECT_DIR = pathlib.Path(__file__).parent.parent.resolve()
PROJECT_DIR = path.join('/Users', 'Harry', 'Documents', 'LocalDevelopment', 'Math-251-Final-Project')
DATA_DIR = path.join(PROJECT_DIR, 'data')

pbp_data = pd.read_csv(path.join(DATA_DIR, '2021.csv'))
pbp_data.columns = ['Index'] + list(pbp_data.columns[1:])
pbp_data = pbp_data.set_index(['Game ID', 'Inn'])
pbp_data_filtered = pbp_data
#pbp_data_filtered = pbp_data[['RoB', 'Out']]
# for ind in pbp_data.index.unique():
#     inning_end = pd.Series(['---', 3], index = ['RoB', 'Out'], name = ind)
#     pbp_data_filtered = pbp_data_filtered.append(inning_end)
pbp_data_filtered = pbp_data_filtered[['Index', 'RoB', 'Out']]


# Filter into training and testing data
mask = np.random.rand(len(pbp_data_filtered.index.unique())) <= 0.8
training_index = pbp_data_filtered.index.unique()[mask]
testing_index = pbp_data_filtered.index.unique()[~mask]

training_pbp_data = pbp_data_filtered.loc[training_index]
testing_pbp_data = pbp_data_filtered.loc[testing_index]


# Set up transition probability matrix
robs = ['---', '1--', '-2-', '--3', '12-', '1-3', '-23', '123']
outs = [0, 1, 2]
states = []
state_indices = {}

i = 0
for out in outs:
    for rob in robs:
        states.append((rob, out))
        state_indices[(rob, out)] = i
        i += 1
states.append(('---', 3))
state_indices[('---', 3)] = i
states
state_indices

transition_counts = { src:{ dst:0 for dst in states } for src in states }


# Fill transition counts from data
for ind in training_pbp_data.index.unique():
    inning_data = training_pbp_data.loc[ind].set_index('Index')
    for play_ind in inning_data.index:
        src = tuple(inning_data.loc[play_ind, ['RoB', 'Out']])
        try:
            dst = tuple(inning_data.loc[play_ind + 1, ['RoB', 'Out']])
        except:
            dst = ('---', 3)
        transition_counts[src][dst] += 1
        
transition_counts


# Create transition probability matrix from transition counts
transition_matrix = [ [ transition_counts[src][dst] / max(sum(transition_counts[src].values()), 1) for dst in states ] for src in states ]
transition_dict = { src:{ dst:(transition_counts[src][dst] / max(sum(transition_counts[src].values()), 1)) for dst in states } for src in states }



# Check transition probability matrix sums
for row in transition_matrix:
    print(sum(row))
    
    

# Create Markov chain using these transitions
mlb_chain = MarkovChain(states, state_indices, transition_matrix)
mlb_chain.simulate_chain(('---', 0), ('---', 3))
















# Markov chain class to model transitions
class OldMarkovChain(object):
    # Initialize the object with a transition probability matrix encoded as a nested dictionary
    def __init__(self, transition_dict):
        self.states = list(transition_dict.keys())
        self.transition_dict = transition_dict
        
    # Get a random next state starting from src
    def next_state(self, src):
        index = np.random.choice(len(self.states), p = [ self.transition_dict[src][dst] for dst in self.states ])
        return states[index]
    
    # Create a chain of length n starting from src
    def generate_states(self, src, n):
        future_states = []
        for i in range(n):
            dst = self.next_state(src)
            future_states.append(dst)
            src = dst
        return future_states
    
    # Simulate a chain from src to dst
    def simulate_chain(self, src, dst):
        state = src
        chain = []
        while state != dst:
            state = self.next_state(state)
            chain.append(state)
        return chain
