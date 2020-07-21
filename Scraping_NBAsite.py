import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
from selenium import webdriver
import time
import re


all_games_list = pd.read_fwf('all_games_nbasite.txt', header=None).values.tolist()

data_game_home = []
data_game_away = []
data_game = []
data_erro = []
driver = webdriver.Edge('C:\Program Files (x86)\Microsoft\Edge\Application\msedgedriver.exe')

for item in range(0, len(all_games_list)):
    try:
        driver.get(all_games_list[item][0])
        time.sleep(5)
        r = requests.get(all_games_list[item][0])
        soup = BeautifulSoup(r.text, 'html.parser')
        data = str(soup.find_all('script')[15].contents[0]).strip().replace("\n", " ")
        m = re.search('window.nbaStatsLineScore =(.+?);', data).group(1)
        dict_data = json.loads(m)

        # STATS HOME TEAM
        GAME_ID = all_games_list[item][0][29:]
        DATE = dict_data[0]['GAME_DATE_EST'][:10]
        TEAM_ABBREVIATION = dict_data[1]['TEAM_ABBREVIATION']
        WINS_LOSSES = dict_data[1]['TEAM_WINS_LOSSES']
        PTS = dict_data[1]['PTS']
        PTS_QTR1 = dict_data[1]['PTS_QTR1']
        PTS_QTR2 = dict_data[1]['PTS_QTR2']
        PTS_QTR3 = dict_data[1]['PTS_QTR3']
        PTS_QTR4 = dict_data[1]['PTS_QTR4']
        FG_PCT = driver.find_element_by_xpath('/html/body/main/div[2]/div/div/div[4]/div/div[2]/div/nba-stat-table[2]/div[2]/div[1]/table/tfoot/tr').text[15:19]
        FT_PCT = driver.find_element_by_xpath('/html/body/main/div[2]/div/div/div[4]/div/div[2]/div/nba-stat-table[2]/div[2]/div[1]/table/tfoot/tr').text[37:41]
        FG3_PCT = driver.find_element_by_xpath('/html/body/main/div[2]/div/div/div[4]/div/div[2]/div/nba-stat-table[2]/div[2]/div[1]/table/tfoot/tr').text[26:30]
        AST = driver.find_element_by_xpath('/html/body/main/div[2]/div/div/div[4]/div/div[2]/div/nba-stat-table[2]/div[2]/div[1]/table/tfoot/tr').text[51:53]
        REB = driver.find_element_by_xpath('/html/body/main/div[2]/div/div/div[4]/div/div[2]/div/nba-stat-table[2]/div[2]/div[1]/table/tfoot/tr').text[48:50]
        TOV = driver.find_element_by_xpath('/html/body/main/div[2]/div/div/div[4]/div/div[2]/div/nba-stat-table[2]/div[2]/div[1]/table/tfoot/tr').text[54:56]
        #STATS AWAY TEAM
        time.sleep(2)
        GAME_ID = all_games_list[item][0][29:]
        DATE = dict_data[0]['GAME_DATE_EST'][:10]
        TEAM_ABBREVIATION = dict_data[0]['TEAM_ABBREVIATION']
        WINS_LOSSES = dict_data[0]['TEAM_WINS_LOSSES']
        PTS = dict_data[0]['PTS']
        PTS_QTR1 = dict_data[0]['PTS_QTR1']
        PTS_QTR2 = dict_data[0]['PTS_QTR2']
        PTS_QTR3 = dict_data[0]['PTS_QTR3']
        PTS_QTR4 = dict_data[0]['PTS_QTR4']
        time.sleep(2)
        FG_PCT = driver.find_element_by_xpath('/html/body/main/div[2]/div/div/div[4]/div/div[2]/div/nba-stat-table[1]/div[2]/div[1]/table/tfoot/tr').text[15:19]
        FT_PCT = driver.find_element_by_xpath('/html/body/main/div[2]/div/div/div[4]/div/div[2]/div/nba-stat-table[1]/div[2]/div[1]/table/tfoot/tr').text[37:41]
        FG3_PCT = driver.find_element_by_xpath('/html/body/main/div[2]/div/div/div[4]/div/div[2]/div/nba-stat-table[1]/div[2]/div[1]/table/tfoot/tr').text[26:30]
        AST = driver.find_element_by_xpath('/html/body/main/div[2]/div/div/div[4]/div/div[2]/div/nba-stat-table[1]/div[2]/div[1]/table/tfoot/tr').text[51:53]
        REB = driver.find_element_by_xpath('/html/body/main/div[2]/div/div/div[4]/div/div[2]/div/nba-stat-table[1]/div[2]/div[1]/table/tfoot/tr').text[48:50]
        TOV = driver.find_element_by_xpath('/html/body/main/div[2]/div/div/div[4]/div/div[2]/div/nba-stat-table[1]/div[2]/div[1]/table/tfoot/tr').text[54:56]
        data_game_away.append(DATE, GAME_ID, TEAM_ABBREVIATION, WINS_LOSSES, PTS, PTS_QTR1, PTS_QTR2, PTS_QTR3, PTS_QTR4,
                              FG_PCT, FT_PCT, FG3_PCT, AST, REB, TOV)
        data_game.append(DATE, GAME_ID, TEAM_ABBREVIATION, WINS_LOSSES, PTS, PTS_QTR1, PTS_QTR2, PTS_QTR3, PTS_QTR4,
                              FG_PCT, FT_PCT, FG3_PCT, AST, REB, TOV)
        print(all_games_list[item][0])
        time.sleep(1)

    except:
        print(f'ERRO: {all_games_list[item][0]}')
        time.sleep(1)

df_total = pd.DataFrame(data_game, columns=['GAME_ID', 'TEAM_ABBREVIATION', 'WINS_LOSSES', 'PTS', 'PTS_QTR1', 'PTS_QTR2',
                                            'PTS_QTR3', 'PTS_QTR4', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 'AST', 'REB', 'TOV'])
df_total.to_csv('dataset1920_total.csv')

df_home = pd.DataFrame(data_game_home, columns=['GAME_ID', 'TEAM_ABBREVIATION', 'WINS_LOSSES', 'PTS', 'PTS_QTR1', 'PTS_QTR2',
                                            'PTS_QTR3', 'PTS_QTR4', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 'AST', 'REB', 'TOV'])
df_home.to_csv('dataset1920_home.csv')

df_away = pd.DataFrame(data_game_away, columns=['GAME_ID', 'TEAM_ABBREVIATION', 'WINS_LOSSES', 'PTS', 'PTS_QTR1', 'PTS_QTR2',
                                            'PTS_QTR3', 'PTS_QTR4', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 'AST', 'REB', 'TOV'])
df_away.to_csv('dataset1920_home.csv')


