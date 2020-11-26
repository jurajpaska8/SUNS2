import pandas as pd


def load_data(filepath):
    return pd.read_csv(filepath, ',')


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


def category_encoded(steam_data):
    steam_data = pd.DataFrame
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

    steam_data['2D'] = 0
    steam_data['3D'] = 0
    steam_data['old_school'] = 0
    steam_data['classic'] = 0
    steam_data['war'] = 0
    steam_data['e_sports'] = 0
    steam_data['team_based'] = 0

    for index, row in steam_data:
        if 'Single-player' in row['categories']:
            row['Singleplayer'] = 1
        if 'Multi-player' in row['categories']:
            row['Multiplayer'] = 1
