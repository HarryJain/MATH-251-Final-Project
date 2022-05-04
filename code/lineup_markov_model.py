#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 21:14:46 2022

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



# Filter the player Series to create a player transition matrix for the given player
def player_series_to_transition_matrix(player_series):
    # Calculate probabilities of relevant plays for each team
    p_BB = player_series['BB'] / player_series['PA']
    p_1B = player_series['1B'] / player_series['PA']
    p_2B = player_series['2B'] / player_series['PA']
    p_3B = player_series['3B'] / player_series['PA']
    p_HR = player_series['HR'] / player_series['PA']
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
    
    # Combine the block matrices into the 25 x 25 transition matrix
    player_transition_matrix = np.vstack((\
        np.concatenate([a, b, c, d], axis = 1),\
        np.concatenate([np.zeros((8, 8)), a, b, e], axis = 1),\
        np.concatenate([np.zeros((8, 8)), np.zeros((8, 8)), a, f], axis = 1),\
        np.concatenate([np.zeros((1, 8)), np.zeros((1, 8)), np.zeros((1, 8)), np.ones((1, 1))], axis = 1)))
    pd.DataFrame(player_transition_matrix)
    
    # Return the transition matrix
    return player_transition_matrix



# Simulate descending and ascending batting orders for the given team
def simulate_batting_orders(team_name, team_df):
    # Create player transition matrices for each player
    team_transition_matrices = []
    for _, row in team_player_data.iterrows():
        team_transition_matrices.append(player_series_to_transition_matrix(row))


    # Create a Markov chain for the team with ascending batting average order
    ascending_results = []
    for i in range(1000):
        player_chain = BaseballChain(team_transition_matrices)
        chain, runs = player_chain.simulate_game()
        ascending_results.append((chain, runs))


    # Create a Markov chain for the team with descending batting average order
    team_transition_matrices.reverse()
    descending_results = []
    for i in range(1000):
        player_chain = BaseballChain(team_transition_matrices)
        chain, runs = player_chain.simulate_game()
        descending_results.append((chain, runs))
        
    
    # Create a Markov chain for the team with ascending batting average order
    random.shuffle(team_transition_matrices)
    random_results = []
    for i in range(1000):
        player_chain = BaseballChain(team_transition_matrices)
        chain, runs = player_chain.simulate_game()
        random_results.append((chain, runs))
    
    
    # Create a new plot
    plt.figure()
        
    # Plot the descending batting average run distribution
    sns.kdeplot([result[1] for result in ascending_results], label = 'Ascending')
    sns.kdeplot([result[1] for result in descending_results], label = 'Descending')
    sns.kdeplot([result[1] for result in random_results], label = 'Random')

    # Plot formatting
    plt.legend(title = 'Team')
    plt.title(f'Run Density Plot for {team[FULL_NAME]}')
    plt.xlabel('Runs Scored')
    plt.ylabel('Density')
    

    ascending_dist = pd.DataFrame([result[1] for result in ascending_results])
    descending_dist = pd.DataFrame([result[1] for result in descending_results])    
    random_dist = pd.DataFrame([result[1] for result in random_results])
    df =  pd.concat([ascending_dist.describe().transpose(), descending_dist.describe().transpose(), random_dist.describe().transpose()])
    df.index = ['Ascending', 'Descending', 'Random']
    return df.sort_values('mean', ascending = False)



# Store the project directory and data directory as constant strings
#PROJECT_DIR = pathlib.Path(__file__).parent.parent.resolve()
PROJECT_DIR = path.join('/Users', 'Harry', 'Documents', 'LocalDevelopment', 'Math-251-Final-Project')
DATA_DIR = path.join(PROJECT_DIR, 'data')

# Load team play-by-play data
team_data = pd.read_csv(path.join(DATA_DIR, 'team_batting_2021.csv'))
team_data['1B'] = team_data['H'] - team_data['2B'] - team_data['3B'] - team_data['HR']

# Load player play-by-play data
player_data = pd.read_csv(path.join(DATA_DIR, 'players_batting_2021.csv')).sort_values('PA', ascending = False)
player_data = player_data.groupby('Tm').head(9)
player_data['1B'] = player_data['H'] - player_data['2B'] - player_data['3B'] - player_data['HR']



# Loop through the teams and simulate descending and ascending batting orders
team_dict = {'San Francisco Giants': 'SFG', 'Los Angeles Dodgers': 'LAD', 'Chicago White Sox': 'CHW', 'Houston Astros': 'HOU', 'Boston Red Sox': 'BOS', 'Tampa Bay Rays': 'TBR', 'Milwaukee Brewers': 'MIL', 'Oakland Athletics': 'OAK', 'San Diego Padres': 'SDP', 'Seattle Mariners': 'SEA', 'New York Mets': 'NYM', 'Toronto Blue Jays': 'TOR', 'New York Yankees': 'NYY', 'Cincinnati Reds': 'CIN', 'Cleveland Indians': 'CLE', 'Philadelphia Phillies': 'PHI', 'St. Louis Cardinals': 'STL', 'Chicago Cubs': 'CHC', 'Atlanta Braves': 'ATL', 'Los Angeles Angels': 'LAA', 'Washington Nationals': 'WSN', 'Detroit Tigers': 'DET', 'Colorado Rockies': 'COL', 'Minnesota Twins': 'MIN', 'Miami Marlins': 'MIA', 'Kansas City Royals': 'KCR', 'Pittsburgh Pirates': 'PIT', 'Texas Rangers': 'TEX', 'Baltimore Orioles': 'BAL', 'Arizona Diamondbacks': 'ARI'}

player_data['Tm'].unique()


# Simulate the distribution of runs for each order for each team
dists = []
for team in list(team_dict.items()):
    print(team)
    team_player_data = player_data[player_data['Tm'] == team[TLA]].sort_values('BA')
    dist = simulate_batting_orders(team, team_player_data)
    print(dist)
    dist = dist.loc[dist['mean'] == max(dist['mean'])].head(1)
    dist['type'] = dist.index[0]
    dist.index = [team[TLA]]
    dists.append(dist.squeeze())
    
# Create a table with the best batting order foe each team
dist_df = pd.DataFrame(dists)
dist_df = dist_df.sort_values('mean', ascending = False)
str(dist_df.style.to_latex()).replace('\\\\\n', '\\ \hline ')
dist_types_df = dist_df.groupby('type').size().to_frame()
str(dist_types_df.transpose().style.to_latex()).replace('\\\\\n', '\\\hline ')
