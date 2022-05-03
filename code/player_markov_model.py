#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 17:55:46 2022

@author: harry
"""


# Module imports
import pathlib
from os import path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
    

# Markov chain class to model transitions
class MarkovChain:
    # Initialize the object with the given states, state_indices, and transition probability matrix
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
        chain = [src]
        while state != dst:
            state = self.next_state(state)
            chain.append(state)
        return chain
    

# Child class of MarkovChain with some additional functions to track runs
class BaseballChain(MarkovChain):
    # Initialize the object with the given transition probability matrix and baseball states
    def __init__(self, transition_matrix):
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
        super().__init__(states, state_indices, transition_matrix)
                
    # Get a random next state starting from src, keeping track of runs scored
    def next_state(self, src):
        dst = super().next_state(src)
        if dst[1] == src[1]:
            runs = 1 + len(src[0].replace('-', '')) - len(dst[0].replace('-', ''))
            self.total_runs += runs
            self.total_pa += 1
        return dst

    # Simulate a single inning, resetting total runs and keeping track of the states and runs scored
    def simulate_inning(self):
        self.total_runs = 0
        self.total_pa = 0
        chain = super().simulate_chain(('---', 0), ('---', 3))
        return chain, self.total_runs, self.total_pa
    
    
    # Simulate a single inning, resetting total runs and keeping track of the states and runs scored
    def simulate_game(self):
        self.total_runs = 0
        self.total_pa = 0
        inning_chains = []
        for i in range(9):
            inning_chains.append(super().simulate_chain(('---', 0), ('---', 3)))
        return inning_chains, self.total_runs, self.total_pa
        


# Store the project directory and data directory as constant strings
#PROJECT_DIR = pathlib.Path(__file__).parent.parent.resolve()
PROJECT_DIR = path.join('/Users', 'Harry', 'Documents', 'LocalDevelopment', 'Math-251-Final-Project')
DATA_DIR = path.join(PROJECT_DIR, 'data')


# Load team play-by-play data
player_pbp_data = pd.read_csv(path.join(DATA_DIR, 'players_batting_2021.csv')).sort_values('PA', ascending = False)
player_pbp_data = player_pbp_data.drop_duplicates(subset = ['Name'])
player_pbp_data['Name'] = player_pbp_data['Name'].apply(lambda x: x.replace('*', '').replace('#', '').replace('+', ''))
player_pbp_data = player_pbp_data.groupby('Tm').head(9)
player_pbp_data['1B'] = player_pbp_data['H'] - player_pbp_data['2B'] - player_pbp_data['3B'] - player_pbp_data['HR']
player_pbp_data[['Name', 'Tm', 'PA', 'BB', '1B', '2B', '3B', 'HR']]


# Loop through the teams and construct their transition matrices, adding them to the dictionary of team transition matrices
player_transition_matrices = {}
player_rows = {}

for ind in player_pbp_data.index:
    # Calculate probabilities of relevant plays for each team
    p_BB = player_pbp_data.loc[ind, 'BB'] / player_pbp_data.loc[ind, 'PA']
    p_1B = player_pbp_data.loc[ind, '1B'] / player_pbp_data.loc[ind, 'PA']
    p_2B = player_pbp_data.loc[ind, '2B'] / player_pbp_data.loc[ind, 'PA']
    p_3B = player_pbp_data.loc[ind, '3B'] / player_pbp_data.loc[ind, 'PA']
    p_HR = player_pbp_data.loc[ind, 'HR'] / player_pbp_data.loc[ind, 'PA']
    p_out = 1 - p_BB - p_1B - p_2B - p_3B - p_HR
    
    # Create the block matrices
    a = [[p_HR, p_1B + p_BB, p_2B, p_3B, 0, 0, 0, 0],\
         [p_HR, 0, 0, p_3B, p_1B + p_BB, 0, p_2B, 0],\
         [p_HR, p_1B, p_2B, p_3B, p_BB, 0, 0, 0],\
         [p_HR, p_1B, p_2B, p_3B, 0, p_BB, 0, 0],\
         [p_HR, 0, 0, p_3B, p_1B, 0, p_2B, p_BB],\
         [p_HR, 0, 0, p_3B, p_1B, 0, p_2B, p_BB],\
         [p_HR, p_1B, p_2B, p_3B, 0, 0, 0, p_BB],\
         [p_HR, 0, 0, p_3B, p_1B, 0, p_2B, p_BB]]
    b = [ [ 0 if j != i else p_out for j in range(8) ] for i in range(8) ]
    c = [ [ 0 for j in range(8) ] for i in range(8) ]
    d = [ [0] for i in range(8) ]
    e = [ [0] for i in range(8) ]
    f = [ [p_out] for i in range(8) ]
    
    # Combine the block matrices into the 24 x 25 transition matrix
    player_transition_matrix = np.vstack((\
        np.concatenate([a, b, c, d], axis = 1),\
        np.concatenate([np.zeros((8, 8)), a, b, e], axis = 1),\
        np.concatenate([np.zeros((8, 8)), np.zeros((8, 8)), a, f], axis = 1),\
        np.concatenate([np.zeros((1, 8)), np.zeros((1, 8)), np.zeros((1, 8)), np.ones((1, 1))], axis = 1)))
    pd.DataFrame(player_transition_matrix)
    
    # Add the transition matrix to our dictionary
    player_transition_matrices[player_pbp_data.loc[ind, 'Name']] = player_transition_matrix
    player_rows


results = {}
players = list(player_pbp_data['Name'])

for player in players:
    player_results = []
    for i in range(10):
        player_chain = BaseballChain(player_transition_matrices[player])
        chain, runs, pa = player_chain.simulate_game()
        player_results.append((chain, runs, pa))
    results[player] = player_results

# run_avg = sum(result[1] for result in results['LAD']) / 100


# Iterate through the players, plot them, and trakc their runs created
rows = []
for player in players[:5]:
    # Draw the density plot
    sns.kdeplot([result[1] for result in results[player]], label = player)
    
    # Calculate the runs created according to the Markov chain and Bill James's metric
    row = player_pbp_data.loc[player_pbp_data['Name'] == player]
    markov_rc = np.mean([result[1] / (sum([len(inning) for inning in result[0]]) - 9) for result in results[player]]) * int(row['PA'])
    runs_created = float(((row['H'] + row['BB']) * row['TB']) / (row['PA']))
    row = {'Name': player, 'Markov RC': markov_rc, 'Runs Created': runs_created}
    rows.append(row)
    
# Plot formatting
plt.legend(title = 'Team')
plt.title('Run Density Plot for Players')
plt.xlabel('Runs Scored')
plt.ylabel('Density')

# Create the runs created DataFrame for the sampled players
rc_df = pd.DataFrame(rows).set_index('Name')
rc_df['Percent Diff'] = (rc_df['Markov RC'] - rc_df['Runs Created']) / rc_df['Runs Created'] * 100
print(rc_df)


# Create a runs created DataFrame for all players
all_rows = []
for player in players:
    # Calculate the runs created according to the Markov chain and Bill James's metric
    row = player_pbp_data.loc[player_pbp_data['Name'] == player]
    markov_rc = np.mean([result[1] / (sum([len(inning) for inning in result[0]]) - 9) for result in results[player]]) * int(row['PA'])
    runs_created = float(((row['H'] + row['BB']) * row['TB']) / (row['PA']))
    row = {'Name': player, 'Markov RC': markov_rc, 'Runs Created': runs_created}
    all_rows.append(row)
    
big_rc_df = pd.DataFrame(all_rows).set_index('Name')
big_rc_df['Percent Diff'] = (big_rc_df['Markov RC'] - big_rc_df['Runs Created']) / big_rc_df['Runs Created'] * 100
print(big_rc_df)

# Print out summary statistics for the percent difference
big_rc_df['Percent Diff'].describe()
