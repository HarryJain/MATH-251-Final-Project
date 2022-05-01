#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 00:28:43 2022

@author: harry
"""


# Module imports
import pathlib
from os import path
import requests
from bs4 import BeautifulSoup as Soup
from bs4 import Comment
import pandas as pd
from time import sleep


# Store the project directory and data directory as constant strings
#PROJECT_DIR = pathlib.Path(__file__).parent.parent.resolve()
PROJECT_DIR = path.join('/Users', 'Harry', 'Documents', 'LocalDevelopment', 'Math-251-Final-Project')
DATA_DIR = path.join(PROJECT_DIR, 'data')


# Return the game url for a given game_id from Baseball Reference
def get_url_from_game_id(game_id, prefix = 'https://www.baseball-reference.com/boxes'):
    return f'{prefix}/{game_id[0:3]}/{game_id}.shtml'


# Return the soup for the game page for a given game_id from Baseball Reference
def get_soup(url):
    print(url)
    response = requests.get(url)
    if not 200 <= response.status_code < 300:
        exit('Invalid Game ID')
    return Soup(response.content, 'html.parser')


# Return a list of elements in a table row
def parse_row(row):
    elements = [ elem.string for elem in row.find_all('td') ]
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


# Get the play-by-play DataFrame for a given href link
def get_game_pbp(href, prefix = 'https://www.baseball-reference.com'):
    url = prefix + href
    soup = get_soup(url)
    soup = soup.find('div', {'id': 'all_play_by_play'})
    pbp_comment = soup.find_all(string = lambda text: isinstance(text, Comment))[0]
    pbp_soup = Soup(pbp_comment, 'html.parser')
    pbp_table = pbp_soup.find('table')
    pbp_df = table_to_df(pbp_table)
    pbp_df.insert(0, 'Game ID', href.split('/')[-1].split('.')[0])
    return pbp_df

    
# Return the boxscore url for a given month, day, and year
def get_url_from_date(year, month, day, prefix = 'https://www.baseball-reference.com/boxes'):
    return f'{prefix}/?year={year}&month={month}&day={day}'


# Get the combined play-by-play DataFrame for a given date
def get_pbp_combined_from_date(year, month, day):
    url = get_url_from_date(year, month, day)
    soup = get_soup(url)
    game_href_tds = soup.find_all('td', {'class', 'right gamelink'})
    game_hrefs = [ td.find('a')['href'] for td in game_href_tds ]
    pbp_dfs = []
    for href in game_hrefs:
        pbp_df = get_game_pbp(href)
        print(pbp_df)
        pbp_dfs.append(pbp_df)
    return pd.concat(pbp_dfs)
    

# Global variables for determining relevant dates to scrape
season_dates = {
    '2021': {
        'start': {
            'year': 2021,
            'month': 4,
            'day': 1,
        },
        'end': {
            'year': 2021,
            'month': 11,
            'day': 2,
        },
    },
}                   

days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

year = season_dates['2021']['start']['year']
month = season_dates['2021']['start']['month']
day = season_dates['2021']['start']['day']

end_year = season_dates['2021']['end']['year']
end_month = season_dates['2021']['end']['month']
end_day = season_dates['2021']['end']['day']


# Store a list of the DataFrames for each date
all_dfs = []
# Loop through all the relevant days and get the games for each
#for i in range(1):
while not (year == int(end_year) and month == int(end_month) and day == int(end_day)):
    df = get_pbp_combined_from_date(year, month, day)
    all_dfs.append(df)
    
    if day == days[month - 1]:
        day = 1
        if month == 12:
            month = 1
            year += 1
        else:
            month += 1
    else:
        day += 1        

    sleep(1)
    

# Combine the daily DataFrames and write it to a CSV
combined_df = pd.concat(all_dfs)
print(combined_df)
combined_df.to_csv(path.join(DATA_DIR, '2021.csv'))
