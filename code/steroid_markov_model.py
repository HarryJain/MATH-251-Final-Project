#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  1 04:06:01 2022

@author: harry
"""


# Module imports
import pathlib
from os import path
import requests
from bs4 import BeautifulSoup as Soup
from bs4 import Comment
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Indexing constants
NAME = 0
HREF = 1
YEARS = 2


# Store the project directory and data directory as constant strings
#PROJECT_DIR = pathlib.Path(__file__).parent.parent.resolve()
PROJECT_DIR = path.join('/Users', 'Harry', 'Documents', 'LocalDevelopment', 'Math-251-Final-Project')
DATA_DIR = path.join(PROJECT_DIR, 'data')


# steroid_players = [('Babe Ruth', 'r/ruthba01.shtml', [1918, 1919, 1920, 1921]),\
#                    ('Barry Bonds', 'b/bondsba01.shtml', [1998, 1999, 2000, 2001]),\
#                    ('Mark McGwire', 'm/mcgwima01.shtml', [1995, 1996, 1998, 1999]),\
#                    ('Sammy Sosa', 's/sosasa01.shtml', [1995, 1996, 1998, 1999, 2001]),\
#                    ('Mike Trout', 't/troutmi01.shtml', [2013, 2014, 2015, 2016]),\
#                    ('Shohei Ohtani', 'o/ohtansh01.shtml', [2018, 2019, 2021])]
steroid_players = [('Barry Bonds', 'b/bondsba01.shtml', [1998, 1999, 2000, 2001]),\
                   ('Mark McGwire', 'm/mcgwima01.shtml', [1995, 1996, 1998, 1999]),\
                   ('Sammy Sosa', 's/sosasa01.shtml', [1995, 1996, 1998, 1999, 2001])]


# Return the batting statistics url for a given year from Baseball Reference
def get_url_from_player(player, prefix = 'https://www.baseball-reference.com/players'):
    return f'{prefix}/{player[HREF]}'


# Return the soup for the game page for a given game_id from Baseball Reference
def get_soup(url):
    print(url)
    response = requests.get(url)
    if not 200 <= response.status_code < 300:
        exit('Invalid Game ID')
    return Soup(response.content, 'html.parser')


# Return a list of elements in a table row
def parse_row(row):
    elements = [ elem.text for elem in row.find_all('td') ]
    return elements


# Convert a HTML table to a DataFrame, with an option to deal with an overheader
def table_to_df(table, overheader = 0, header = 'th'):
    cols = table.find('thead').find_all('tr')[overheader].find_all('th')
    cols = [ col.string if col.string != None else '' for col in cols ]
    
    stat_table = table.find('tbody')
        
    rows = stat_table.find_all('tr')
    
    rows = [row for row in rows if row.find(header) != None ]
    headers = [ row.find(header).text for row in rows if row.find(header) != None ]
    parsed_rows = [ parse_row(row) for row in rows ]
    parsed_headers = [ headers[i] for i in range(len(parsed_rows)) if parsed_rows[i] != [] and parsed_rows[i][0] != None]
    parsed_rows = [ row for row in parsed_rows if row != [] and row[0] != None ]
    df = pd.DataFrame(parsed_rows)
    if len(headers) != 0:
        df.insert(0, '', parsed_headers)
    df.columns = cols
    
    return df
    

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


def get_player_distributions(player):
    url = get_url_from_player(player)
    soup = get_soup(url)
    
    
    # Get the team batting statistics
    player_table = soup.find('table', {'id': 'batting_standard'})
    player_df = table_to_df(player_table)
    player_df = player_df[player_df['H'].str.strip().astype(bool)]
    print(player_df)
    player_df[['H', '2B', '3B', 'HR', 'Year', 'BB', 'PA']] = player_df[['H', '2B', '3B', 'HR', 'Year', 'BB', 'PA']].astype(int)
    player_df['1B'] = player_df['H'] - player_df['2B'] - player_df['3B'] - player_df['HR']
    print(player_df)
    
    
    # Calculate the transition matrices for each year for the given player
    player_transition_matrices = {}
    for year in player[YEARS]:
        matrix = player_series_to_transition_matrix(player_df.loc[player_df['Year'] == year].squeeze())
        player_transition_matrices[year] = matrix
    
    results = {}
    
    # Simulate a run distribution for each year for a given playeer
    for year in player[YEARS]:
        year_results = []
        for i in range(1000):
            player_chain = BaseballChain(matrix)
            chain, runs = player_chain.simulate_game()
            year_results.append((chain, runs))
        results[year] = year_results
    
    # Create a new plot
    plt.figure()
        
    # Plot the run distributions for each year
    for year in player[YEARS]:
        sns.kdeplot([result[1] for result in results[year]], label = year, clip = (0, None))
    
    # Plot formatting
    plt.legend(title = 'Team')
    plt.title(f'Run Density Plot for {player[NAME]}')
    plt.xlabel('Runs Scored')
    plt.ylabel('Density')
    
    return results
    

# Plot the results for each player
for player in steroid_players:
    results = get_player_distributions(player)
    year = list(results.keys())[0]