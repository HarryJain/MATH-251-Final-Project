#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 22:38:31 2022

@author: harry
"""


# Module imports
import pathlib
from os import path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
    

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
        return dst

    # Simulate a single inning, resetting total runs and keeping track of the states and runs scored
    def simulate_inning(self):
        self.total_runs = 0
        chain = super().simulate_chain(('---', 0), ('---', 3))
        return chain, self.total_runs
    
    
    # Simulate a single inning, resetting total runs and keeping track of the states and runs scored
    def simulate_game(self):
        self.total_runs = 0
        inning_chains = []
        for i in range(9):
            inning_chains.append(super().simulate_chain(('---', 0), ('---', 3)))
        return inning_chains, self.total_runs
        


# Store the project directory and data directory as constant strings
#PROJECT_DIR = pathlib.Path(__file__).parent.parent.resolve()
PROJECT_DIR = path.join('/Users', 'Harry', 'Documents', 'LocalDevelopment', 'Math-251-Final-Project')
DATA_DIR = path.join(PROJECT_DIR, 'data')


# Load team play-by-play data
team_pbp_data = pd.read_csv(path.join(DATA_DIR, 'team_batting_2021.csv'))
team_pbp_data['1B'] = team_pbp_data['H'] - team_pbp_data['2B'] - team_pbp_data['3B'] - team_pbp_data['HR']
team_pbp_data[['Tm', 'PA', 'BB', '1B', '2B', '3B', 'HR']]


# Loop through the teams and construct their transition matrices, adding them to the dictionary of team transition matrices
team_transition_matrices = {}
team_dict = {'San Francisco Giants': 'SFG', 'Los Angeles Dodgers': 'LAD', 'Chicago White Sox': 'CHW', 'Houston Astros': 'HOU', 'Boston Red Sox': 'BOS', 'Tampa Bay Rays': 'TBR', 'Milwaukee Brewers': 'MIL', 'Oakland Athletics': 'OAK', 'San Diego Padres': 'SDP', 'Seattle Mariners': 'SEA', 'New York Mets': 'NYM', 'Toronto Blue Jays': 'TOR', 'New York Yankees': 'NYY', 'Cincinnati Reds': 'CIN', 'Cleveland Indians': 'CLE', 'Philadelphia Phillies': 'PHI', 'St. Louis Cardinals': 'STL', 'Chicago Cubs': 'CHI', 'Atlanta Braves': 'ATL', 'Los Angeles Angels': 'LAA', 'Washington Nationals': 'WAS', 'Detroit Tigers': 'DET', 'Colorado Rockies': 'COL', 'Minnesota Twins': 'MIN', 'Miami Marlins': 'MIA', 'Kansas City Royals': 'KCR', 'Pittsburgh Pirates': 'PIT', 'Texas Rangers': 'TEX', 'Baltimore Orioles': 'BAL', 'Arizona Diamondbacks': 'ARI', 'League Average': 'AVG'}

for ind in team_pbp_data.index:
    # Calculate probabilities of relevant plays for each team
    p_BB = team_pbp_data.loc[ind, 'BB'] / team_pbp_data.loc[ind, 'PA']
    p_1B = team_pbp_data.loc[ind, '1B'] / team_pbp_data.loc[ind, 'PA']
    p_2B = team_pbp_data.loc[ind, '2B'] / team_pbp_data.loc[ind, 'PA']
    p_3B = team_pbp_data.loc[ind, '3B'] / team_pbp_data.loc[ind, 'PA']
    p_HR = team_pbp_data.loc[ind, 'HR'] / team_pbp_data.loc[ind, 'PA']
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
    team_transition_matrix = np.vstack((\
        np.concatenate([a, b, c, d], axis = 1),\
        np.concatenate([np.zeros((8, 8)), a, b, e], axis = 1),\
        np.concatenate([np.zeros((8, 8)), np.zeros((8, 8)), a, f], axis = 1),\
        np.concatenate([np.zeros((1, 8)), np.zeros((1, 8)), np.zeros((1, 8)), np.ones((1, 1))], axis = 1)))
    pd.DataFrame(team_transition_matrix)
    
    # Add the transition matrix to our dictionary
    team_transition_matrices[team_dict[team_pbp_data.loc[ind, 'Tm']]] = team_transition_matrix


results = {}
teams = list(team_dict.values())

for team in teams:
    team_results = []
    for i in range(100):
        team_chain = BaseballChain(team_transition_matrices[team])
        chain, runs = team_chain.simulate_game()
        team_results.append((chain, runs))
    results[team] = team_results

run_avg = sum(result[1] for result in results['LAD']) / 100


# Iterate through the teams and plot them
for team in teams[:5]:
    # Draw the density plot
    sns.kdeplot([result[1] for result in results[team]], label = team)
    
    
# Plot formatting
plt.legend(title = 'Team')
plt.title('Density Plot for Teams')
plt.xlabel('Runs Scored')
plt.ylabel('Density')


# Plot a heatmap of the probability transition matrices
sns.heatmap(team_transition_matrices['SFG'], cmap = 'icefire')
