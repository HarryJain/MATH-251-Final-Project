#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 18:08:51 2022

@author: harry
"""


# Module imports
import pathlib
from os import path
import requests
from bs4 import BeautifulSoup as Soup
from bs4 import Comment
import pandas as pd


# Store the project directory and data directory as constant strings
#PROJECT_DIR = pathlib.Path(__file__).parent.parent.resolve()
PROJECT_DIR = path.join('/Users', 'Harry', 'Documents', 'LocalDevelopment', 'Math-251-Final-Project')
DATA_DIR = path.join(PROJECT_DIR, 'data')


# Return the batting statistics url for a given year from Baseball Reference
def get_url_from_year(year = 2021, prefix = 'https://www.baseball-reference.com/leagues/majors'):
    return f'{prefix}/{year}-standard-batting.shtml'


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


# Get the batting statistics page data
url = get_url_from_year()
soup = get_soup(url)


# Get the team batting statistics
team_table = soup.find('table', {'id': 'teams_standard_batting'})
team_df = table_to_df(team_table)
print(team_df)


# Get the player batting statistics
players_comment = soup.find('div', {'id': 'all_players_standard_batting'}).find(string = lambda text: isinstance(text, Comment))
players_soup = Soup(players_comment, 'html.parser')
players_table = players_soup.find('table')
players_df = table_to_df(players_table)
print(players_df)


# Write the DataFrames to csv files
team_df.to_csv(path.join(DATA_DIR, 'team_batting_2021.csv'))
players_df.to_csv(path.join(DATA_DIR, 'players_batting_2021.csv'))
