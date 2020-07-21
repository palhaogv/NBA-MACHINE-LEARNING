import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier


def frames():
    temps = ['1213', '1314', '1415', '1516', '1617', '1718', '1819']
    frames = []
    for item in temps:
        globals()['raw_' + str(item)] = pd.read_csv('datasets//' + str(item) + 'raw_scores.txt', sep=',')
        globals()['raw_' + str(item)].rename(columns={'Unnamed: 0': 'DATE'}, inplace=True)
        globals()['raw_' + str(item)]['WINS'], globals()['raw_' + str(item)]['LOSSES'] = globals()['raw_' + str(item)]['TEAM_WINS_LOSSES'].str.extract(r'(\d?\d)-(\d?\d)').iloc[:, 0], globals()['raw_' + str(item)]['TEAM_WINS_LOSSES'].str.extract(r'(\d?\d)-(\d?\d)').iloc[:, 1]
        globals()['raw_' + str(item)]['WINS'] = pd.to_numeric(globals()['raw_' + str(item)]['WINS'])
        globals()['raw_' + str(item)]['LOSSES'] = pd.to_numeric(globals()['raw_' + str(item)]['LOSSES'])
        globals()['raw_' + str(item)].columns = map(str.upper, globals()['raw_' + str(item)].columns)

        globals()['vegas_' + str(item)] = pd.read_csv('datasets//' + str(item) + 'vegas.txt', sep=',')
        globals()['vegas_' + str(item)].columns = ['DATE', 'LOCATION', 'TEAM_CITY_NAME', 'OPP_TEAM', 'TEAM_ID', 'GAME_ID', 'PERCENTBET_ML', 'OPEN_LINE_ML', 'PINNACLE_ML', '5DIMES_ML', 'HERITAGE_ML', 'BOVADA_ML', 'BETONLINE_ML', 'AVERAGE_LINE_ML', 'BEST_LINE_ML', 'WORST_LINE_ML', 'PERCENTBET_SPREAD', 'OPEN_LINE_SPREAD', 'OPEN_ODDS_SPREAD', 'PINNACLE_LINE_SPREAD', 'PINNACLE_ODDS_SPREAD', '5DIMES_LINE_SPREAD', 
        '5DIMES_ODDS_SPREAD', 'HERITAGE_LINE_SPREAD', 'HERITAGE_ODDS_SPREAD', 'BOVADA_LINE_SPREAD', 'BOVADA_ODDS_SPREAD', 'BETONLINE_LINE_SPREAD', 'BETONLINE_ODDS_SPREAD', 'AVERAGE_LINE_SPREAD', 'AVERAGE_ODDS_SPREAD', 'BEST_LINE_SPREAD', 'WORST_LINE_SPREAD', 'BEST_ODDS_SPREAD', 'WORST_ODDS_SPREAD', 'PERCENTBET_OU', 'OPEN_LINE_OU', 'OPEN_ODDS_OU', 'PINNACLE_LINE_OU', 'PINNACLE_ODDS_OU', '5DIMES_LINE_OU', '5DIMES_ODDS_OU', 'HERITAGE_LINE_OU', 'HERITAGE_ODDS_OU', 'BOVADA_LINE_OU', 'BOVADA_ODDS_OU', 'BETONLINE_LINE_OU', 'BETONLINE_ODDS_OU', 'AVERAGE_LINE_OU', 'AVERAGE_ODDS_OU', 'BEST_LINE_OU', 'WORST_LINE_OU', 'BEST_ODDS_OU', 'WORST_ODDS_OU', 'PTS', 'SPREAD', 'RESULT', 'TOTAL']
        globals()['vegas_' + str(item)].columns = map(str.upper, globals()['vegas_' + str(item)].columns)

        globals()['dataset' + str(item)] = pd.merge(globals()['raw_' + str(item)], globals()['vegas_' + str(item)], how='inner')
        globals()['dataset' + str(item)]['WINS'] = np.where(globals()['dataset' + str(item)]['RESULT'] == 'W', globals()['dataset' + str(item)]['WINS'] - 1, globals()['dataset' + str(item)]['WINS'])
        globals()['dataset' + str(item)]['LOSSES'] = np.where(globals()['dataset' + str(item)]['RESULT'] == 'L', globals()['dataset' + str(item)]['LOSSES'] - 1, globals()['dataset' + str(item)]['LOSSES'])

        frames.append(globals()['dataset' + str(item)])

    for frame in frames:
        frame['IDEAL_ODDS_ML'] = np.where(frame['AVERAGE_LINE_ML'] > 0, frame['AVERAGE_LINE_ML'] / 100 + 1, (100 / frame['AVERAGE_LINE_ML'] - 1) * -1)
        frame['W/L 1/0'] = np.where(frame['RESULT'] == 'L', 0, 1)  # binaring results
        frame['LOCATION'] = np.where(frame['LOCATION'] == 'away', 0, 1)  # away = 0, home = 1

        #POINTS STATS
        frame['MEAN_PTS'] = frame.groupby('TEAM_CITY_NAME')['PTS'].transform(lambda x: x.expanding().mean().shift())
        frame['MEAN_PTS_H'] = frame[frame['LOCATION'] == 1].groupby('TEAM_CITY_NAME')['PTS'].transform(lambda x: x.expanding().mean().shift())
        frame['MEAN_PTS_A'] = frame[frame['LOCATION'] == 0].groupby('TEAM_CITY_NAME')['PTS'].transform(lambda x: x.expanding().mean().shift())
        frame['LAST_3_PTS'] = frame.groupby('TEAM_CITY_NAME')['PTS'].transform(lambda x: x.rolling(window=3).mean().shift())
        frame['LAST_5_PTS'] = frame.groupby('TEAM_CITY_NAME')['PTS'].transform(lambda x: x.rolling(window=5).mean().shift())

        #POINTS PQTR STATS
        frame['MEAN_PTS_QTR1'] = frame.groupby('TEAM_CITY_NAME')['PTS_QTR1'].transform(lambda x: x.expanding().mean().shift())
        frame['MEAN_PTS_QTR2'] = frame.groupby('TEAM_CITY_NAME')['PTS_QTR2'].transform(lambda x: x.expanding().mean().shift())
        frame['MEAN_PTS_QTR3'] = frame.groupby('TEAM_CITY_NAME')['PTS_QTR3'].transform(lambda x: x.expanding().mean().shift())
        frame['MEAN_PTS_QTR4'] = frame.groupby('TEAM_CITY_NAME')['PTS_QTR4'].transform(lambda x: x.expanding().mean().shift())


        #FG STATS
        frame['MEAN_FG_PCT'] = frame.groupby('TEAM_CITY_NAME')['FG_PCT'].transform(lambda x: x.expanding().mean().shift())
        frame['MEAN_FG_PCT_H'] = frame[frame['LOCATION'] == 1].groupby('TEAM_CITY_NAME')['FG_PCT'].transform(lambda x: x.expanding().mean().shift())
        frame['MEAN_FG_PCT_A'] = frame[frame['LOCATION'] == 0].groupby('TEAM_CITY_NAME')['FG_PCT'].transform(lambda x: x.expanding().mean().shift())
        frame['LAST_3_FG_PCT'] = frame.groupby('TEAM_CITY_NAME')['FG_PCT'].transform(lambda x: x.rolling(window=3).mean().shift())
        frame['LAST_5_FG_PCT'] = frame.groupby('TEAM_CITY_NAME')['FG_PCT'].transform(lambda x: x.rolling(window=5).mean().shift())

        #FT STATS
        frame['MEAN_FT_PCT'] = frame.groupby('TEAM_CITY_NAME')['FT_PCT'].transform(lambda x: x.expanding().mean().shift())
        frame['MEAN_FT_PCT_H'] = frame[frame['LOCATION'] == 1].groupby('TEAM_CITY_NAME')['FT_PCT'].transform(lambda x: x.expanding().mean().shift())
        frame['MEAN_FT_PCT_A'] = frame[frame['LOCATION'] == 0].groupby('TEAM_CITY_NAME')['FT_PCT'].transform(lambda x: x.expanding().mean().shift())
        frame['LAST_3_FT_PCT'] = frame.groupby('TEAM_CITY_NAME')['FT_PCT'].transform(lambda x: x.rolling(window=3).mean().shift())
        frame['LAST_5_FT_PCT'] = frame.groupby('TEAM_CITY_NAME')['FT_PCT'].transform(lambda x: x.rolling(window=5).mean().shift())

        #FG3 STATS
        frame['MEAN_FG3_PCT'] = frame.groupby('TEAM_CITY_NAME')['FG3_PCT'].transform(lambda x: x.expanding().mean().shift())
        frame['MEAN_FG3_PCT_H'] = frame[frame['LOCATION'] == 1].groupby('TEAM_CITY_NAME')['FG3_PCT'].transform(lambda x: x.expanding().mean().shift())
        frame['MEAN_FG3_PCT_A'] = frame[frame['LOCATION'] == 0].groupby('TEAM_CITY_NAME')['FG3_PCT'].transform(lambda x: x.expanding().mean().shift())
        frame['LAST_3_FG3_PCT'] = frame.groupby('TEAM_CITY_NAME')['FG3_PCT'].transform(lambda x: x.rolling(window=3).mean().shift())
        frame['LAST_5_FG3_PCT'] = frame.groupby('TEAM_CITY_NAME')['FG3_PCT'].transform(lambda x: x.rolling(window=5).mean().shift())

        #AST STATS
        frame['MEAN_AST'] = frame.groupby('TEAM_CITY_NAME')['AST'].transform(lambda x: x.expanding().mean().shift())
        frame['MEAN_AST_H'] = frame[frame['LOCATION'] == 1].groupby('TEAM_CITY_NAME')['AST'].transform(lambda x: x.expanding().mean().shift())
        frame['MEAN_AST_A'] = frame[frame['LOCATION'] == 0].groupby('TEAM_CITY_NAME')['AST'].transform(lambda x: x.expanding().mean().shift())
        frame['LAST_3_AST'] = frame.groupby('TEAM_CITY_NAME')['AST'].transform(lambda x: x.rolling(window=3).mean().shift())
        frame['LAST_5_AST'] = frame.groupby('TEAM_CITY_NAME')['AST'].transform(lambda x: x.rolling(window=5).mean().shift())

        #REB STATS
        frame['MEAN_REB'] = frame.groupby('TEAM_CITY_NAME')['REB'].transform(lambda x: x.expanding().mean().shift())
        frame['MEAN_REB_H'] = frame[frame['LOCATION'] == 1].groupby('TEAM_CITY_NAME')['REB'].transform(lambda x: x.expanding().mean().shift())
        frame['MEAN_REB_A'] = frame[frame['LOCATION'] == 0].groupby('TEAM_CITY_NAME')['REB'].transform(lambda x: x.expanding().mean().shift())
        frame['LAST_3_REB'] = frame.groupby('TEAM_CITY_NAME')['REB'].transform(lambda x: x.rolling(window=3).mean().shift())
        frame['LAST_5_REB'] = frame.groupby('TEAM_CITY_NAME')['REB'].transform(lambda x: x.rolling(window=5).mean().shift())

        #TOV STATS
        frame['MEAN_TOV'] = frame.groupby('TEAM_CITY_NAME')['TOV'].transform(lambda x: x.expanding().mean().shift())
        frame['MEAN_TOV_H'] = frame[frame['LOCATION'] == 1].groupby('TEAM_CITY_NAME')['TOV'].transform(lambda x: x.expanding().mean().shift())
        frame['MEAN_TOV_A'] = frame[frame['LOCATION'] == 0].groupby('TEAM_CITY_NAME')['TOV'].transform(lambda x: x.expanding().mean().shift())
        frame['LAST_3_TOV'] = frame.groupby('TEAM_CITY_NAME')['TOV'].transform(lambda x: x.rolling(window=3).mean().shift())
        frame['LAST_5_TOV'] = frame.groupby('TEAM_CITY_NAME')['TOV'].transform(lambda x: x.rolling(window=5).mean().shift())

        frame['OPP_W/L_1/0'] = np.where(frame['W/L 1/0'] == 1, 0, 1)
        frame['LAST_2_MH'] = frame.groupby('TEAM_ABBREVIATION')['W/L 1/0'].transform(lambda x: x.rolling(window=2).sum().shift())
        frame['LAST_2_MA'] = frame.groupby('TEAM_ABBREVIATION')['OPP_W/L_1/0'].transform(lambda x: x.rolling(window=2).sum().shift())
        frame['LAST_3_MH'] = frame.groupby('TEAM_ABBREVIATION')['W/L 1/0'].transform(lambda x: x.rolling(window=3).sum().shift())
        frame['LAST_3_MA'] = frame.groupby('TEAM_ABBREVIATION')['OPP_W/L_1/0'].transform(lambda x: x.rolling(window=3).sum().shift())
        frame['LAST_5_MH'] = frame.groupby('TEAM_ABBREVIATION')['W/L 1/0'].transform(lambda x: x.rolling(window=5).sum().shift())
        frame['LAST_5_MA'] = frame.groupby('TEAM_ABBREVIATION')['OPP_W/L_1/0'].transform(lambda x: x.rolling(window=5).sum().shift())

    return frames


def data_set():
    season_frames = frames()
    NBA_all_odds = pd.concat(season_frames)
    NBA_all_odds = NBA_all_odds.reset_index(drop=True)

    # Merging WAY and HOME in a sigle game
    NBA_all_odds_home = NBA_all_odds[NBA_all_odds['LOCATION'] == 1]
    NBA_all_odds_away = NBA_all_odds[NBA_all_odds['LOCATION'] == 0]
    NBA_all_odds_away = NBA_all_odds_away[['DATE', 'GAME_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'PTS', 'IDEAL_ODDS_ML', 'WINS', 'LOSSES', 'MEAN_PTS', 'MEAN_PTS_QTR1', 'MEAN_PTS_QTR2', 'MEAN_PTS_QTR3', 'MEAN_PTS_QTR4', 'MEAN_FG_PCT', 'MEAN_FT_PCT', 'MEAN_FG3_PCT', 'MEAN_AST', 'MEAN_REB', 'MEAN_TOV', 'LAST_2_MA', 'LAST_3_MA', 'LAST_5_MA', 'MEAN_PTS_A', 'MEAN_FG_PCT_A', 'MEAN_FT_PCT_A', 'MEAN_FG3_PCT_A', 'MEAN_AST_A', 'MEAN_REB_A', 'MEAN_TOV_A', 'LAST_3_PTS', 'LAST_5_PTS', 'LAST_3_FG_PCT', 'LAST_5_FG_PCT', 'LAST_3_FT_PCT', 'LAST_5_FT_PCT', 'LAST_3_FG3_PCT', 'LAST_5_FG3_PCT', 'LAST_3_AST', 'LAST_5_AST', 'LAST_3_REB', 'LAST_5_REB', 'LAST_3_TOV', 'LAST_5_TOV']]
    NBA_all_odds_away.columns = ['DATE', 'GAME_ID', 'OPP_TEAM_ID', 'OPP_TEAM_ABBREVIATION', 'OPP_PTS', 'OPP_IDEAL_ODDS_ML', 'OPP_TEAM_WINS', 'OPP_TEAM_LOSSES', 'OPP_TEAM_MEAN_PTS', 'OPP_TEAM_MEAN_PTS_QTR1', 'OPP_TEAM_MEAN_PTS_QTR2', 'OPP_TEAM_MEAN_PTS_QTR3', 'OPP_TEAM_MEAN_PTS_QTR4', 'OPP_TEAM_MEAN_FG_PCT', 'OPP_TEAM_MEAN_FT_PCT', 'OPP_TEAM_MEAN_FG3_PCT', 'OPP_TEAM_MEAN_AST', 'OPP_TEAM_MEAN_REB', 'OPP_TEAM_MEAN_TOV', 'LAST_2_MA', 'LAST_3_MA', 'LAST_5_MA', 'MEAN_PTS_A', 'MEAN_FG_PCT_A', 'MEAN_FT_PCT_A', 'MEAN_FG3_PCT_A', 'MEAN_AST_A', 'MEAN_REB_A', 'MEAN_TOV_A', 'OPP_TEAM_LAST_3_PTS', 'OPP_TEAM_LAST_5_PTS', 'OPP_TEAM_LAST_3_FG_PCT', 'OPP_TEAM_LAST_5_FG_PCT', 'OPP_TEAM_LAST_3_FT_PCT', 'OPP_TEAM_LAST_5_FT_PCT', 'OPP_TEAM_LAST_3_FG3_PCT', 'OPP_TEAM_LAST_5_FG3_PCT', 'OPP_TEAM_LAST_3_AST', 'OPP_TEAM_LAST_5_AST', 'OPP_TEAM_LAST_3_REB', 'OPP_TEAM_LAST_5_REB', 'OPP_TEAM_LAST_3_TOV', 'OPP_TEAM_LAST_5_TOV']

    NBA_all_odds_away = NBA_all_odds_away.reset_index(drop=True)
    NBA_all_odds_home = NBA_all_odds_home.reset_index(drop=True).drop(['LAST_2_MA', 'LAST_3_MA', 'LAST_5_MA', 'MEAN_PTS_A', 'MEAN_FG_PCT_A', 'MEAN_FT_PCT_A', 'MEAN_FG3_PCT_A', 'MEAN_AST_A', 'MEAN_REB_A', 'MEAN_TOV_A', 'GAME_DATE_EST'], axis=1)

    NBA_ALL_ODDS_MERGED = pd.merge(NBA_all_odds_home, NBA_all_odds_away, how='inner')
    NBA_ALL_ODDS_MERGED = NBA_ALL_ODDS_MERGED.dropna(axis=0, how='any', inplace=False)
    month = []
    for item in NBA_ALL_ODDS_MERGED['DATE']:
        x = item[:7]
        month.append(x)
    NBA_ALL_ODDS_MERGED['MONTH'] = month

    columns_to_use = ['DATE', 'MONTH', 'GAME_ID', 'TEAM_ID', 'OPP_TEAM_ID', 'TEAM_ABBREVIATION', 'OPP_TEAM_ABBREVIATION', 'PTS', 'OPP_PTS', 'W/L 1/0', 'OPP_W/L_1/0', 'IDEAL_ODDS_ML', 'OPP_IDEAL_ODDS_ML', 'WINS', 'LOSSES', 'MEAN_PTS', 'MEAN_PTS_QTR1', 'MEAN_PTS_QTR2', 'MEAN_PTS_QTR3', 'MEAN_PTS_QTR4', 'MEAN_FG_PCT', 'MEAN_FT_PCT', 'MEAN_FG3_PCT', 'MEAN_AST', 'MEAN_REB', 'MEAN_TOV', 'LAST_2_MH', 'LAST_3_MH', 'LAST_5_MH', 'MEAN_PTS_H', 'MEAN_FG_PCT_H', 'MEAN_FT_PCT_H', 'MEAN_FG3_PCT_H', 'MEAN_AST_H', 'MEAN_REB_H', 'MEAN_TOV_H', 'OPP_TEAM_WINS', 'OPP_TEAM_LOSSES', 'OPP_TEAM_MEAN_PTS', 'OPP_TEAM_MEAN_PTS_QTR1', 'OPP_TEAM_MEAN_PTS_QTR2', 'OPP_TEAM_MEAN_PTS_QTR3', 'OPP_TEAM_MEAN_PTS_QTR4', 'OPP_TEAM_MEAN_FG_PCT', 'OPP_TEAM_MEAN_FT_PCT', 'OPP_TEAM_MEAN_FG3_PCT', 'OPP_TEAM_MEAN_AST', 'OPP_TEAM_MEAN_REB', 'OPP_TEAM_MEAN_TOV', 'LAST_2_MA', 'LAST_3_MA', 'LAST_5_MA', 'MEAN_PTS_A', 'MEAN_FG_PCT_A', 'MEAN_FT_PCT_A', 'MEAN_FG3_PCT_A', 'MEAN_AST_A', 'MEAN_REB_A', 'MEAN_TOV_A', 'LAST_3_PTS', 'LAST_5_PTS', 'LAST_3_FG_PCT', 'LAST_5_FG_PCT', 'LAST_3_FT_PCT', 'LAST_5_FT_PCT', 'LAST_3_FG3_PCT', 'LAST_5_FG3_PCT', 'LAST_3_AST', 'LAST_5_AST', 'LAST_3_REB', 'LAST_5_REB', 'LAST_3_TOV', 'LAST_5_TOV',  'OPP_TEAM_LAST_3_PTS', 'OPP_TEAM_LAST_5_PTS', 'OPP_TEAM_LAST_3_FG_PCT', 'OPP_TEAM_LAST_5_FG_PCT', 'OPP_TEAM_LAST_3_FT_PCT', 'OPP_TEAM_LAST_5_FT_PCT', 'OPP_TEAM_LAST_3_FG3_PCT', 'OPP_TEAM_LAST_5_FG3_PCT', 'OPP_TEAM_LAST_3_AST', 'OPP_TEAM_LAST_5_AST', 'OPP_TEAM_LAST_3_REB', 'OPP_TEAM_LAST_5_REB', 'OPP_TEAM_LAST_3_TOV', 'OPP_TEAM_LAST_5_TOV']
    NBA_ALL_ODDS_MERGED = NBA_ALL_ODDS_MERGED[columns_to_use]
    NBA_ALL_ODDS_MERGED = NBA_ALL_ODDS_MERGED.reset_index(drop=True)


    return NBA_ALL_ODDS_MERGED


def feature_and_target():
    features = ['WINS', 'LOSSES', 'MEAN_PTS', 'MEAN_PTS_H', 'MEAN_PTS_QTR1', 'MEAN_PTS_QTR2', 'MEAN_PTS_QTR3', 'MEAN_PTS_QTR4', 'MEAN_FG_PCT', 'MEAN_FG_PCT_H', 'MEAN_FT_PCT', 'MEAN_FT_PCT_H', 'MEAN_FG3_PCT', 'MEAN_FG3_PCT_H', 'MEAN_AST', 'MEAN_AST_H', 'MEAN_REB', 'MEAN_REB_H', 'MEAN_TOV', 'MEAN_TOV_H', 'LAST_2_MH', 'LAST_3_MH', 'LAST_5_MH', 'OPP_TEAM_WINS', 'OPP_TEAM_LOSSES', 'OPP_TEAM_MEAN_PTS', 'MEAN_PTS_A', 'OPP_TEAM_MEAN_PTS_QTR1', 'OPP_TEAM_MEAN_PTS_QTR2', 'OPP_TEAM_MEAN_PTS_QTR3', 'OPP_TEAM_MEAN_PTS_QTR4', 'OPP_TEAM_MEAN_FG_PCT', 'MEAN_FG_PCT_A', 'OPP_TEAM_MEAN_FT_PCT', 'MEAN_FT_PCT_A', 'OPP_TEAM_MEAN_FG3_PCT', 'MEAN_FG3_PCT_A', 'OPP_TEAM_MEAN_AST', 'MEAN_AST_A', 'OPP_TEAM_MEAN_REB', 'MEAN_REB_A', 'OPP_TEAM_MEAN_TOV', 'MEAN_TOV_A', 'LAST_2_MA', 'LAST_3_MA', 'LAST_5_MA', 'LAST_3_PTS', 'LAST_5_PTS', 'LAST_3_FG_PCT', 'LAST_5_FG_PCT', 'LAST_3_FT_PCT', 'LAST_5_FT_PCT', 'LAST_3_FG3_PCT', 'LAST_5_FG3_PCT', 'LAST_3_AST', 'LAST_5_AST', 'LAST_3_REB', 'LAST_5_REB', 'LAST_3_TOV', 'LAST_5_TOV',  'OPP_TEAM_LAST_3_PTS', 'OPP_TEAM_LAST_5_PTS', 'OPP_TEAM_LAST_3_FG_PCT', 'OPP_TEAM_LAST_5_FG_PCT', 'OPP_TEAM_LAST_3_FT_PCT', 'OPP_TEAM_LAST_5_FT_PCT', 'OPP_TEAM_LAST_3_FG3_PCT', 'OPP_TEAM_LAST_5_FG3_PCT', 'OPP_TEAM_LAST_3_AST', 'OPP_TEAM_LAST_5_AST', 'OPP_TEAM_LAST_3_REB', 'OPP_TEAM_LAST_5_REB', 'OPP_TEAM_LAST_3_TOV', 'OPP_TEAM_LAST_5_TOV']
    target = ['W/L 1/0']

    return features, target


def KNN(Xtr, ytr, Xte, yte): #KNN(near neightboor) test
    '''grid_p = {
        'n_neighbors': range(1, 20),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    gs = GridSearchCV(KNeighborsClassifier(), grid_p, verbose=1, cv=3, n_jobs=-1).fit(Xtr, ytr)
    print(f'The best parameters with the Grid Search are: {gs.best_params_}')
    The best parameters with the Grid Search are: {'metric': 'manhattan', 'n_neighbors': 16, 'weights': 'distance'}'''
    for i in range(1, 20):
        knn = KNeighborsClassifier(n_neighbors=i, weights='distance').fit(Xtr, ytr)
        if i == 1:
            s_te = knn.score(Xte, yte)
            s_tr = knn.score(Xtr, ytr)
            g = i
        if knn.score(Xte, yte) > s_te:
            s_te = knn.score(Xte, yte)
            s_tr = knn.score(Xtr, ytr)
            g = i
    y_pred = knn.predict(Xte)
    print(f'The best n_neighbors is: {g}')
    print(f'Accuracy of K-NN test on train set is: {knn.score(Xtr, ytr)}')
    print(f'Accuracy of K-NN test on test set is: {knn.score(Xte, yte)}')
    print('Confusion Matrix : \n' + str(confusion_matrix(yte, y_pred)))
    plt.figure()
    sns.heatmap(confusion_matrix(yte, y_pred) / np.sum(confusion_matrix(yte, y_pred)), annot=True, fmt='.2%',
                cmap='Blues')
    plt.show()
    return y_pred


def Decision_tree(Xtr, ytr, Xte, yte): #Decision tree test
    dt = DecisionTreeClassifier().fit(Xtr, ytr)
    print(f'Accuracy of DecisionTreeClassifier test on train set is: {dt.score(Xtr, ytr)}')
    print(f'Accuracy of DecisionTreeClassifier test on test set is: {dt.score(Xte, yte)}')
    return dt.predict(Xte)


def SVC_test(Xtr, ytr, Xte, yte): #SVC
    for i in [0.01, 0.1, 1, 2, 10]:
        svm = SVC(gamma=i).fit(Xtr, ytr)
        if i == 0.01:
            s_te = svm.score(Xte, yte)
            s_tr = svm.score(Xtr, ytr)
            g = i
        if svm.score(Xte, yte) > s_te:
            s_te = svm.score(Xte, yte)
            s_tr = svm.score(Xtr, ytr)
            g = i
    print(f'The best gamma is: {g}')
    print(f'Accuracy of SVC test on train set is: {s_tr}')
    print(f'Accuracy of SVC test on test set is: {s_te}')
    return svm.predict(Xte)


def logistic_reg(Xtr, ytr, Xte, yte):
    lr = LogisticRegression(penalty='l2', C=0.001, max_iter=4000).fit(Xtr, ytr)
    y_pred = lr.predict(Xte)
    print(f'Accuracy of Logistic Regression test on train set is: {lr.score(Xtr, ytr):.4f}')
    print(f'Accuracy of Logistic Regression test on test set is: {lr.score(Xte, yte):.4f}')
    print('Confusion Matrix : \n' + str(confusion_matrix(yte, y_pred)))
    return y_pred


def xgboost_test(Xtr, ytr, Xte, yte):
    xgb = XGBClassifier().fit(Xtr, ytr)
    y_pred = xgb.predict(Xte)
    print(f'Accuracy of Logistic Regression test on train set is: {xgb.score(Xtr, ytr)}')
    print(f'Accuracy of Logistic Regression test on test set is: {xgb.score(Xte, yte)}')
    print('Confusion Matrix : \n' + str(confusion_matrix(yte, y_pred)))
    return y_pred


def features_importances(Xtr, ytr):
    from sklearn.ensemble import GradientBoostingClassifier
    ## call model
    model = GradientBoostingClassifier()
    ## Importance
    model.fit(Xtr, ytr)
    importances = model.feature_importances_
    ## Put in a pandas dtf
    dtf_importances = pd.DataFrame({"IMPORTANCE":importances, "VARIABLE":Xtr.columns.tolist()}).sort_values("IMPORTANCE", ascending=False)
    dtf_importances['cumsum'] = dtf_importances['IMPORTANCE'].cumsum(axis=0)
    dtf_importances = dtf_importances.set_index("VARIABLE")
            
    ## Plot
    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=[100,100])
    fig.suptitle("Features Importance")
    ax[0].title.set_text('variables')
    dtf_importances[["IMPORTANCE"]].sort_values(by="IMPORTANCE").plot(kind="barh", legend=False, ax=ax[0]).grid(axis="x")
    ax[0].set(ylabel="")
    ax[1].title.set_text('cumulative')
    dtf_importances[["cumsum"]].plot(kind="line", linewidth=4, legend=False, ax=ax[1])
    ax[1].set(xlabel="", xticks=np.arange(len(dtf_importances)), xticklabels=dtf_importances.index)
    plt.xticks(rotation=70)
    plt.grid(axis='both')
    plt.show()


def scaler(X_tr, X_te):
    scaller = MinMaxScaler()
    X_train_transf = scaller.fit_transform(X_tr)
    X_test_transf = scaller.transform(X_te)
    return X_train_transf, X_test_transf


def game_list_nba_site():
    all_games_list = []
    for i in range(1, 974):
        game = 'https://stats.nba.com/game/0021900' + str(i).zfill(3)
        all_games_list.append(game)
        print(game)


def training_test():
    NBA_ALL_ODDS_MERGED = data_set()

    features, target = feature_and_target()
    X_train, X_test, y_train, y_test = train_test_split(NBA_ALL_ODDS_MERGED[features], NBA_ALL_ODDS_MERGED[target], random_state=10)


    #X_train_transf, X_test_transf = scaler(X_train, X_test)

    pred = logistic_reg(X_train, y_train, X_test, y_test)
    X_test['PREDICT'] = pred
    DB_test = pd.merge(X_test, NBA_ALL_ODDS_MERGED, how='left')
    DB_test = DB_test[['DATE', 'TEAM_ABBREVIATION', 'OPP_TEAM_ABBREVIATION', 'W/L 1/0', 'PREDICT', 'IDEAL_ODDS_ML', 'OPP_IDEAL_ODDS_ML']]
    DB_test['ODD_CHOSEN'] = np.where(DB_test['PREDICT'] == 1, DB_test['IDEAL_ODDS_ML'] - 1, DB_test['OPP_IDEAL_ODDS_ML'] - 1)
    DB_test['GAIN'] = np.where(DB_test['W/L 1/0'] == DB_test['PREDICT'], DB_test['ODD_CHOSEN'], -1)
    #DB_test
    DB_test_f = DB_test[DB_test['ODD_CHOSEN'] > 1]
    print(f'Betting 1 dollar in all winner the machine predict, at the end of {len(DB_test)} games, \n'
        f'the gain is {sum(DB_test["GAIN"]):.2f}. Returning {sum(DB_test["GAIN"]) / len(DB_test) * 100:.2f}%.')
    print('----------------------------------------------------------------------------------------------------------')
    print(f'Betting 1 dollar in all winner the machine predict with odd bigger than 2, at the end of {len(DB_test_f)} games, \n'
        f'the gain is {sum(DB_test_f["GAIN"]):.2f}. Returning {sum(DB_test_f["GAIN"]) / len(DB_test_f) * 100:.2f}%.')
    
    return DB_test, DB_test_f


def odds_1920():
    odds_1920 = pd.read_csv('odds1920.csv').dropna()
    odds_1920 = odds_1920[['DATE', 'TEAM_ABBREVIATION', 'OPP_TEAM_ABBREVIATION', 'PTS', 'OPP_PTS','ODDS_ML_BETONLINE', 'ODDS_ML_BOGDOG', 'ODDS_ML_BUMBET', 'ODDS_ML_INTERTOPS', 'ODDS_ML_OPENING', 'OPP_ODDS_ML_BETONLINE', 'OPP_ODDS_ML_BOGDOG', 'OPP_ODDS_ML_BUMBET', 'OPP_ODDS_ML_INTERTOPS', 'OPP_ODDS_ML_OPENING']]

    #ODDS HOME
    odds_1920['ODDS_ML_BETONLINE_R'] = np.where(odds_1920['ODDS_ML_BETONLINE'] > 0, odds_1920['ODDS_ML_BETONLINE'] / 100 + 1, (100 / odds_1920['ODDS_ML_BETONLINE'] - 1) * -1)
    odds_1920['ODDS_ML_BOGDOG_R'] = np.where(odds_1920['ODDS_ML_BOGDOG'] > 0, odds_1920['ODDS_ML_BOGDOG'] / 100 + 1, (100 / odds_1920['ODDS_ML_BOGDOG'] - 1) * -1)
    odds_1920['ODDS_ML_BUMBET_R'] = np.where(odds_1920['ODDS_ML_BUMBET'] > 0, odds_1920['ODDS_ML_BUMBET'] / 100 + 1, (100 / odds_1920['ODDS_ML_BUMBET'] - 1) * -1)
    odds_1920['ODDS_ML_INTERTOPS_R'] = np.where(odds_1920['ODDS_ML_INTERTOPS'] > 0, odds_1920['ODDS_ML_INTERTOPS'] / 100 + 1, (100 / odds_1920['ODDS_ML_INTERTOPS'] - 1) * -1)
    odds_1920['ODDS_ML_OPENING_R'] = np.where(odds_1920['ODDS_ML_OPENING'] > 0, odds_1920['ODDS_ML_OPENING'] / 100 + 1, (100 / odds_1920['ODDS_ML_OPENING'] - 1) * -1)

    #ODDS AWAY
    odds_1920['OPP_ODDS_ML_BETONLINE_R'] = np.where(odds_1920['OPP_ODDS_ML_BETONLINE'] > 0, odds_1920['OPP_ODDS_ML_BETONLINE'] / 100 + 1, (100 / odds_1920['OPP_ODDS_ML_BETONLINE'] - 1) * -1)
    odds_1920['OPP_ODDS_ML_BOGDOG_R'] = np.where(odds_1920['OPP_ODDS_ML_BOGDOG'] > 0, odds_1920['OPP_ODDS_ML_BOGDOG'] / 100 + 1, (100 / odds_1920['OPP_ODDS_ML_BOGDOG'] - 1) * -1)
    odds_1920['OPP_ODDS_ML_BUMBET_R'] = np.where(odds_1920['OPP_ODDS_ML_BUMBET'] > 0, odds_1920['OPP_ODDS_ML_BUMBET'] / 100 + 1, (100 / odds_1920['OPP_ODDS_ML_BUMBET'] - 1) * -1)
    odds_1920['OPP_ODDS_ML_INTERTOPS_R'] = np.where(odds_1920['OPP_ODDS_ML_INTERTOPS'] > 0, odds_1920['OPP_ODDS_ML_INTERTOPS'] / 100 + 1, (100 / odds_1920['OPP_ODDS_ML_INTERTOPS'] - 1) * -1)
    odds_1920['OPP_ODDS_ML_OPENING_R'] = np.where(odds_1920['OPP_ODDS_ML_OPENING'] > 0, odds_1920['OPP_ODDS_ML_OPENING'] / 100 + 1, (100 / odds_1920['OPP_ODDS_ML_OPENING'] - 1) * -1)

    #AVERAGE
    odds_1920['IDEAL_ODDS_ML'] = odds_1920[['ODDS_ML_BETONLINE_R', 'ODDS_ML_BOGDOG_R', 'ODDS_ML_BUMBET_R', 'ODDS_ML_INTERTOPS_R', 'ODDS_ML_OPENING_R']].mean(axis=1)
    odds_1920['OPP_IDEAL_ODDS_ML'] = odds_1920[['OPP_ODDS_ML_BETONLINE_R', 'OPP_ODDS_ML_BOGDOG_R', 'OPP_ODDS_ML_BUMBET_R', 'OPP_ODDS_ML_INTERTOPS_R', 'OPP_ODDS_ML_OPENING_R']].mean(axis=1)

    odds_1920 = odds_1920.sort_values('DATE')
    
    return odds_1920


def dataset_1920():
    df_odds = odds_1920()
    df1920 = pd.read_csv('dataset1920_total.csv', index_col=[0]).reset_index(drop=True).drop_duplicates().sort_values('GAME_ID')

    #HOMA/AWAY STATS
    df1920['MEAN_PTS'] = df1920.groupby('TEAM_ABBREVIATION')['PTS'].transform(lambda x: x.expanding().mean().shift())
    df1920['MEAN_PTS_QTR1'] = df1920.groupby('TEAM_ABBREVIATION')['PTS_QTR1'].transform(lambda x: x.expanding().mean().shift())
    df1920['MEAN_PTS_QTR2'] = df1920.groupby('TEAM_ABBREVIATION')['PTS_QTR2'].transform(lambda x: x.expanding().mean().shift())
    df1920['MEAN_PTS_QTR3'] = df1920.groupby('TEAM_ABBREVIATION')['PTS_QTR3'].transform(lambda x: x.expanding().mean().shift())
    df1920['MEAN_PTS_QTR4'] = df1920.groupby('TEAM_ABBREVIATION')['PTS_QTR4'].transform(lambda x: x.expanding().mean().shift())
    df1920['MEAN_FG_PCT'] = df1920.groupby('TEAM_ABBREVIATION')['FG_PCT'].transform(lambda x: x.expanding().mean().shift())
    df1920['MEAN_FT_PCT'] = df1920.groupby('TEAM_ABBREVIATION')['FT_PCT'].transform(lambda x: x.expanding().mean().shift())
    df1920['MEAN_FG3_PCT'] = df1920.groupby('TEAM_ABBREVIATION')['FG3_PCT'].transform(lambda x: x.expanding().mean().shift())
    df1920['MEAN_AST'] = df1920.groupby('TEAM_ABBREVIATION')['AST'].transform(lambda x: x.expanding().mean().shift())
    df1920['MEAN_REB'] = df1920.groupby('TEAM_ABBREVIATION')['REB'].transform(lambda x: x.expanding().mean().shift())
    df1920['MEAN_TOV'] = df1920.groupby('TEAM_ABBREVIATION')['TOV'].transform(lambda x: x.expanding().mean().shift())

    #HOME STATS
    df1920['MEAN_PTS_H'] = df1920[df1920['LOC'] == 1].groupby('TEAM_ABBREVIATION')['PTS'].transform(lambda x: x.expanding().mean().shift())
    df1920['MEAN_FG_PCT_H'] = df1920[df1920['LOC'] == 1].groupby('TEAM_ABBREVIATION')['FG_PCT'].transform(lambda x: x.expanding().mean().shift())
    df1920['MEAN_FT_PCT_H'] = df1920[df1920['LOC'] == 1].groupby('TEAM_ABBREVIATION')['FT_PCT'].transform(lambda x: x.expanding().mean().shift())
    df1920['MEAN_FG3_PCT_H'] = df1920[df1920['LOC'] == 1].groupby('TEAM_ABBREVIATION')['FG3_PCT'].transform(lambda x: x.expanding().mean().shift())
    df1920['MEAN_AST_H'] = df1920[df1920['LOC'] == 1].groupby('TEAM_ABBREVIATION')['AST'].transform(lambda x: x.expanding().mean().shift())
    df1920['MEAN_REB_H'] = df1920[df1920['LOC'] == 1].groupby('TEAM_ABBREVIATION')['REB'].transform(lambda x: x.expanding().mean().shift())
    df1920['MEAN_TOV_H'] = df1920[df1920['LOC'] == 1].groupby('TEAM_ABBREVIATION')['TOV'].transform(lambda x: x.expanding().mean().shift())


    #AWAY STATS
    df1920['MEAN_PTS_A'] =df1920[df1920['LOC'] == 0].groupby('TEAM_ABBREVIATION')['PTS'].transform(lambda x: x.expanding().mean().shift())
    df1920['MEAN_FG_PCT_A'] = df1920[df1920['LOC'] == 0].groupby('TEAM_ABBREVIATION')['FG_PCT'].transform(lambda x: x.expanding().mean().shift())
    df1920['MEAN_FT_PCT_A'] = df1920[df1920['LOC'] == 0].groupby('TEAM_ABBREVIATION')['FT_PCT'].transform(lambda x: x.expanding().mean().shift())
    df1920['MEAN_FG3_PCT_A'] = df1920[df1920['LOC'] == 0].groupby('TEAM_ABBREVIATION')['FG3_PCT'].transform(lambda x: x.expanding().mean().shift())
    df1920['MEAN_AST_A'] = df1920[df1920['LOC'] == 0].groupby('TEAM_ABBREVIATION')['AST'].transform(lambda x: x.expanding().mean().shift())
    df1920['MEAN_REB_A'] = df1920[df1920['LOC'] == 0].groupby('TEAM_ABBREVIATION')['REB'].transform(lambda x: x.expanding().mean().shift())
    df1920['MEAN_TOV_A'] = df1920[df1920['LOC'] == 0].groupby('TEAM_ABBREVIATION')['TOV'].transform(lambda x: x.expanding().mean().shift())


    #SEP AND MERGE THE DATASET
    #HOME
    df1920_home = df1920[df1920['LOC'] == 1]
    df1920_home = df1920_home.drop(['MEAN_PTS_A', 'MEAN_FG_PCT_A', 'MEAN_FT_PCT_A', 'MEAN_FG3_PCT_A', 'MEAN_AST_A', 'MEAN_REB_A', 'MEAN_TOV_A'], axis=1)


    df1920_away = df1920[df1920['LOC'] == 0]
    df1920_away = df1920_away[['GAME_ID', 'DATE', 'TEAM_ABBREVIATION', 'WINS', 'LOSSES', 'PTS', 'MEAN_PTS', 'MEAN_PTS_QTR1', 'MEAN_PTS_QTR2', 'MEAN_PTS_QTR3', 'MEAN_PTS_QTR4', 'MEAN_FG_PCT', 'MEAN_FT_PCT', 'MEAN_FG3_PCT', 'MEAN_AST', 'MEAN_REB', 'MEAN_TOV', 'MEAN_PTS_A', 'MEAN_FG_PCT_A', 'MEAN_FT_PCT_A', 'MEAN_FG3_PCT_A', 'MEAN_AST_A', 'MEAN_REB_A', 'MEAN_TOV_A']]
    df1920_away.columns = ['GAME_ID', 'DATE', 'OPP_TEAM_ABBREVIATION', 'OPP_TEAM_WINS', 'OPP_TEAM_LOSSES', 'OPP_PTS', 'OPP_TEAM_MEAN_PTS', 'OPP_TEAM_MEAN_PTS_QTR1', 'OPP_TEAM_MEAN_PTS_QTR2', 'OPP_TEAM_MEAN_PTS_QTR3', 'OPP_TEAM_MEAN_PTS_QTR4', 'OPP_TEAM_MEAN_FG_PCT', 'OPP_TEAM_MEAN_FT_PCT', 'OPP_TEAM_MEAN_FG3_PCT', 'OPP_TEAM_MEAN_AST', 'OPP_TEAM_MEAN_REB', 'OPP_TEAM_MEAN_TOV', 'MEAN_PTS_A', 'MEAN_FG_PCT_A', 'MEAN_FT_PCT_A', 'MEAN_FG3_PCT_A', 'MEAN_AST_A', 'MEAN_REB_A', 'MEAN_TOV_A']

    df1920_games = pd.merge(df1920_home, df1920_away, how ='inner')
    df1920_games = df1920_games.dropna(axis=0, how='any', inplace=False)

    df1920_games['W/L 1/0'] = np.where((df1920_games['PTS'] - df1920_games['OPP_PTS']) > 0, 1, 0)

    df1920_games['LAST_2_MH'] = df1920_games.groupby('TEAM_ABBREVIATION')['W/L 1/0'].transform(lambda x: x.rolling(window=2).sum().shift())
    df1920_games['LAST_3_MH'] = df1920_games.groupby('TEAM_ABBREVIATION')['W/L 1/0'].transform(lambda x: x.rolling(window=3).sum().shift())
    df1920_games['LAST_5_MH'] = df1920_games.groupby('TEAM_ABBREVIATION')['W/L 1/0'].transform(lambda x: x.rolling(window=5).sum().shift())
    df1920_games['LAST_2_MA'] = df1920_games.groupby('OPP_TEAM_ABBREVIATION')['W/L 1/0'].transform(lambda x: x.rolling(window=2).sum().shift())
    df1920_games['LAST_3_MA'] = df1920_games.groupby('OPP_TEAM_ABBREVIATION')['W/L 1/0'].transform(lambda x: x.rolling(window=3).sum().shift())
    df1920_games['LAST_5_MA'] = df1920_games.groupby('OPP_TEAM_ABBREVIATION')['W/L 1/0'].transform(lambda x: x.rolling(window=5).sum().shift())

    df1920_games = df1920_games.dropna().sort_values('DATE')

    df1920_games = pd.merge(df1920_games, df_odds[['TEAM_ABBREVIATION', 'OPP_TEAM_ABBREVIATION', 'PTS', 'OPP_PTS', 'ODDS_ML_BETONLINE_R', 'ODDS_ML_BOGDOG_R', 'ODDS_ML_BUMBET_R', 'ODDS_ML_INTERTOPS_R', 'ODDS_ML_OPENING_R', 'OPP_ODDS_ML_BETONLINE_R', 'OPP_ODDS_ML_BOGDOG_R', 'OPP_ODDS_ML_BUMBET_R', 'OPP_ODDS_ML_INTERTOPS_R', 'OPP_ODDS_ML_OPENING_R', 'IDEAL_ODDS_ML', 'OPP_IDEAL_ODDS_ML']], how='left')
    df1920_games = df1920_games.dropna()

    df1920_games['WINS'] = np.where(df1920_games['W/L 1/0'] == 1, df1920_games['WINS'] - 1, df1920_games['WINS'])
    df1920_games['LOSSES'] = np.where(df1920_games['W/L 1/0'] == 0, df1920_games['LOSSES'] - 1, df1920_games['LOSSES'])
    df1920_games['OPP_TEAM_WINS'] = np.where(df1920_games['W/L 1/0'] == 0, df1920_games['OPP_TEAM_WINS'] - 1, df1920_games['OPP_TEAM_WINS'])
    df1920_games['OPP_TEAM_LOSSES'] = np.where(df1920_games['W/L 1/0'] == 1, df1920_games['OPP_TEAM_LOSSES'] - 1, df1920_games['OPP_TEAM_LOSSES'])

    month = []
    for item in df1920_games['DATE']:
        x = item[:7]
        month.append(x)
    df1920_games['MONTH'] = month

    return df1920_games


def testing_1920():
    '''Function to test the profit of 1920 season untill the COVID break.
    DB_test: prediction of all games
    DB_test_a: only underdog predictions(the profit key!!!!!).
    DB_test_g: return by month of DB_test_a'''
    df1920_games = dataset_1920()
    database = data_set()
    features, target = feature_and_target()
    X_train = database[features]
    y_train = database[target]
    X_test = df1920_games[features]
    y_test = df1920_games[target]
    pred = logistic_reg(X_train, y_train, X_test, y_test)
    X_test['PREDICT'] = pred
    DB_test = pd.merge(X_test, df1920_games, how='left')
    DB_test = DB_test[['DATE', 'MONTH', 'TEAM_ABBREVIATION', 'OPP_TEAM_ABBREVIATION', 'PTS', 'OPP_PTS', 'W/L 1/0', 'PREDICT', 'IDEAL_ODDS_ML', 'OPP_IDEAL_ODDS_ML']]
    DB_test['ODD_CHOSEN'] = np.where(DB_test['PREDICT'] == 1, DB_test['IDEAL_ODDS_ML'] - 1, DB_test['OPP_IDEAL_ODDS_ML'] - 1)
    DB_test['GAIN'] = np.where(DB_test['W/L 1/0'] == DB_test['PREDICT'], DB_test['ODD_CHOSEN'], -1)
    DB_test_a = DB_test[DB_test['ODD_CHOSEN'] > 1]
    
    print(f'Betting 1 dollar in all winner the machine predict, at the end of {len(DB_test)} games, \n'
        f'the gain is {sum(DB_test["GAIN"]):.2f}. Returning {sum(DB_test["GAIN"]) / len(DB_test) * 100:.2f}%.')
    print('----------------------------------------------------------------------------------------------------------')
    print(f'Betting 1 dollar in all winner the machine predict with odd bigger than 2, at the end of {len(DB_test_a)} games, \n'
        f'the gain is {sum(DB_test_a["GAIN"]):.2f}. Returning {sum(DB_test_a["GAIN"]) / len(DB_test_a) * 100:.2f}%.')
    
    DB_test_g = DB_test_a.groupby('MONTH').mean()

    return DB_test, DB_test_a, DB_test_g


#def return_by_year():
