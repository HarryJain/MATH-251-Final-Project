#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 23:36:42 2022

@author: harry
"""

# Module imports
import pathlib
from os import path
import pandas as pd
import numpy as np


# Indexing constants
STATE = 0
OUT = 1

CHAIN = 0
RUNS = 1

DESCENDING = 0
ASCENDING = 1

FULL_NAME = 0
TLA = 1
    

# Markov chain class to model transitions
class MarkovChain:
    # Initialize the object with the given states, state_indices, and a sequence of transition probability matrices
    def __init__(self, states, state_indices, transition_matrices):
        self.states = states
        self.state_indices = state_indices
        self.transition_matrices = transition_matrices
        self.transition_index = 0
        
    # Get a random next state starting from src, iterating the transition matrix
    def next_state(self, src):
        index = np.random.choice(len(self.states), p = [ self.transition_matrices[self.transition_index][self.state_indices[src]][dst_ind] for dst_ind in range(len(self.states)) ])
        self.transition_index = (self.transition_index + 1) % len(self.transition_matrices)
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
        chain = [src]
        while state != dst:
            state = self.next_state(state)
            chain.append(state)
        return chain
    
    

# Child class of MarkovChain with some additional functions to track runs
class BaseballChain(MarkovChain):
    # Initialize the object with the given transition probability matrices and baseball states
    def __init__(self, transition_matrices):
        # Set up the basseball innning states
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
        
        # Call the parent __init__ with the baseball inning states
        super().__init__(states, state_indices, transition_matrices)
                
    # Get a random next state starting from src, keeping track of runs scored
    def next_state(self, src):
        dst = super().next_state(src)
        if dst[OUT] == src[OUT]:
            runs = 1 + len(src[STATE].replace('-', '')) - len(dst[STATE].replace('-', ''))
            self.total_runs += runs
            self.total_ab += 1
        return dst

    # Simulate a single inning, resetting total runs and keeping track of the states and runs scored
    def simulate_inning(self):
        self.total_runs = 0
        self.total_ab = 0
        chain = super().simulate_chain(('---', 0), ('---', 3))
        return chain, self.total_runs
    
    
    # Simulate a single inning, resetting total runs and keeping track of the states and runs scored
    def simulate_game(self):
        self.total_runs = 0
        self.total_ab = 0
        inning_chains = []
        for i in range(9):
            inning_chains.append(super().simulate_chain(('---', 0), ('---', 3)))
        return inning_chains, self.total_runs


# Store the project directory and data directory as constant strings
#PROJECT_DIR = pathlib.Path(__file__).parent.parent.resolve()
PROJECT_DIR = path.join('/Users', 'Harry', 'Documents', 'LocalDevelopment', 'Math-251-Final-Project')
DATA_DIR = path.join(PROJECT_DIR, 'data')
PLOT_DIR = path.join(PROJECT_DIR, 'plots')

pbp_data = pd.read_csv(path.join(DATA_DIR, '2021.csv'))
pbp_data.columns = ['Index'] + list(pbp_data.columns[1:])
pbp_data = pbp_data.set_index(['Game ID', 'Inn'])
pbp_data_filtered = pbp_data
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
    
    
# Create a heatmap of the transition matrix
plt.figure()
plt.title(f'Transition Matrix for 2021 Season')
sns.heatmap(transition_matrix, cmap = 'icefire')

# Create Markov chain using these transitions
mlb_chain = BaseballChain([transition_matrix])

# Simulate 10000 innings according to our chain
runs = []
for i in range(9000):
    runs.append(mlb_chain.simulate_inning()[1])

# Get the five number summary of the chain run data
five_sum = pd.DataFrame(runs).describe().transpose()
five_sum.style.to_latex(hrules = True, clines = 'all;data')

# Create a DataFrame comparing the 
runs_df = pd.DataFrame(runs, columns = ['Runs'])
counts = runs_df.value_counts()
percents = runs_df.value_counts(normalize = True).mul(100)
stoll = pd.Series([72.65, 14.90, 6.86, 3.15, 1.41, 0.61, 0.25, 0.10, 0.04, 0.02, 0.007, 0.003])
stoll.index = counts.index
run_count_df = pd.concat([counts, percents, stoll], axis = 1,\
                         keys=('Markov Counts', 'Markov Percentage', 'Stoll Percentage'))
run_count_df.style.to_latex()


# Plot the run distribution
sns.kdeplot(runs)
plt.title('Run Density Plot for a 2021 MLB Half Inning')
plt.xlabel('Runs Scored')
plt.ylabel('Density')

