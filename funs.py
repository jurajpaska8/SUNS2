import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_data(filepath):
    return pd.read_csv(filepath, ',', index_col=0)


def drop_all_na(dataframe):
    return pd.DataFrame.dropna(dataframe)


# returns DF with multiplayer value
def export_multiplayer_games(steam_data, steamspy_data, path):
    steam_EDA = steam_data.copy(deep=True)
    steam_EDA['multiplayer'] = 0

    multiplayer_cols = [col for col in steamspy_data.columns if 'multiplayer' in col]
    idx = 0
    for id in steam_EDA.appid:
        row_num = steamspy_data.index[steamspy_data['appid'] == id].tolist()[0]
        i = 0
        for col in multiplayer_cols:
            col_num = steamspy_data.columns.get_loc(col)
            i += steamspy_data.iat[row_num, col_num]
        if i > 0:
            steam_EDA.at[idx, 'multiplayer'] = 1
        idx += 1
    steam_EDA.set_index('appid')
    steam_EDA.to_csv(path)
    return steam_EDA


def publishers_developers_encoded(steam_data, path):
    publishers = steam_data.publisher.value_counts()
    for p, count in publishers.items():
        steam_data['publisher'] = steam_data['publisher'].replace([p], count)  # good

    developers = steam_data.developer.value_counts()
    for d, count in developers.items():
        steam_data['developer'] = steam_data['developer'].replace([d], count)  # good

    steam_data.to_csv(path)


def add_features(steam_data):
    # categories
    steam_data['Multiplayer'] = 0
    steam_data['Singleplayer'] = 0
    steam_data['Action'] = 0
    steam_data['Adventure'] = 0
    steam_data['Indie'] = 0
    steam_data['Strategy'] = 0
    steam_data['RPG'] = 0
    steam_data['Casual'] = 0
    steam_data['Simulation'] = 0
    steam_data['Racing'] = 0
    steam_data['Sports'] = 0

    # tags
    steam_data['2d'] = 0
    steam_data['3d'] = 0
    steam_data['old_school'] = 0
    steam_data['classic'] = 0
    steam_data['war'] = 0
    steam_data['e_sports'] = 0
    steam_data['team_based'] = 0
    return steam_data

def category_encoded_tags(steam_data, steam_data_spy, label):
    cols_label = [col for col in steam_data_spy.columns if label in col]
    idx = 0
    for id in steam_data.appid:
        row_num = steam_data_spy.index[steam_data_spy['appid'] == id].tolist()[0]
        i = 0
        for col in cols_label:
            col_num = steam_data_spy.columns.get_loc(col)
            i += steam_data_spy.iat[row_num, col_num]
        if i > 0:
            steam_data.at[idx, label] = 1
        idx += 1


def category_encoded(steam_data):
    idx = 0
    for index in steam_data.appid:
        if 'Single-player' in steam_data.at[idx, 'categories']:
            steam_data.at[idx, 'Singleplayer'] = 1
        if 'Multi-player' in steam_data.at[idx, 'categories']:
            steam_data.at[idx, 'Multiplayer'] = 1
        if 'Action' in steam_data.at[idx, 'genres']:
            steam_data.at[idx, 'Action'] = 1
        if 'Adventure' in steam_data.at[idx, 'genres']:
            steam_data.at[idx, 'Adventure'] = 1
        if 'Indie' in steam_data.at[idx, 'genres']:
            steam_data.at[idx, 'Indie'] = 1
        if 'Strategy' in steam_data.at[idx, 'genres']:
            steam_data.at[idx, 'Strategy'] = 1
        if 'RPG' in steam_data.at[idx, 'genres']:
            steam_data.at[idx, 'RPG'] = 1
        if 'Casual' in steam_data.at[idx, 'genres']:
            steam_data.at[idx, 'Casual'] = 1
        if 'Simulation' in steam_data.at[idx, 'genres']:
            steam_data.at[idx, 'Simulation'] = 1
        if 'Racing' in steam_data.at[idx, 'genres']:
            steam_data.at[idx, 'Racing'] = 1
        if 'Sports' in steam_data.at[idx, 'genres']:
            steam_data.at[idx, 'Sports'] = 1
        idx += 1


def visualisation(df, cluster_count, id_label):
    features = df.columns
    labels = []
    for i in range(cluster_count):
        labels.append(i)

    dataframe_collection = {}
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    for i in range(cluster_count):
        dataframe_collection[i] = df.loc[df[id_label] == i]

    idx = 0
    for f in features:
        fig, ax = plt.subplots()
        vals = []
        for i in range(cluster_count):
            vals.append(dataframe_collection[i][f].mean())
        rects1 = ax.bar(x - width / 2, vals,
                        width, label=f)
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Scores')
        ax.set_title('Scores by feature')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        idx += 1
        plt.show()


def platforms_encode(dataframe):
    dataframe['windows'] = 0
    dataframe['linux'] = 0
    dataframe['mac'] = 0
    for index in range(len(dataframe.appid)):
        if 'windows' in dataframe.at[index, 'platforms']:
            dataframe.at[index, 'windows'] = 1
        if 'linux' in dataframe.at[index, 'platforms']:
            dataframe.at[index, 'linux'] = 1
        if 'mac' in dataframe.at[index, 'platforms']:
            dataframe.at[index, 'mac'] = 1