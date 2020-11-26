import funs
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    ########################################################################
    # 1 - LOAD, MERGE, ENCODE, EDA, NORMALIZE###############################
    ########################################################################
    # LOAD
    stream_data = funs.load_data('./data4/steam.csv')
    # stream_data.set_index('appid')
    streamspy_data = funs.load_data('./data4/steamspy_tag_data.csv')
    # streamspy_data.set_index('appid')

    # MERGE
    merged_data = pd.merge(stream_data, streamspy_data, on="appid", right_index=True, left_index=True)

    # EDA - three questions
    # multiplayer more expensive, played, ranked in general
    # find multiplayers and exports it
    # funs.export_multiplayer_games(stream_data, streamspy_data, './data4/multiplayer.csv')
    steam_EDA = funs.load_data('./data4/multiplayer.csv') # TODO find out why is added first column
    multiplayer_games = steam_EDA.loc[steam_EDA['multiplayer'] == 1]
    singleplayer_games = steam_EDA.loc[steam_EDA['multiplayer'] == 0]

    singleplayer_played_hours_avg = singleplayer_games["average_playtime"].mean()/100
    multiplayer_played_hours_avg = multiplayer_games["average_playtime"].mean()/100

    singleplayer_price_avg = singleplayer_games["price"].mean()
    multiplayer_price_avg = multiplayer_games["price"].mean()

    singleplayer_feedbacks_avg = singleplayer_games["positive_ratings"].mean()/1000
    multiplayer_feedbacks_avg = multiplayer_games["positive_ratings"].mean()/1000

    # utilities : average prize / average playtime / average positive:negative feedbacks
    labels = ['price', 'hours/100', 'feedbacks/1000']
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, [singleplayer_price_avg, singleplayer_played_hours_avg, singleplayer_feedbacks_avg], width, label='SinglePlayer')
    rects2 = ax.bar(x + width / 2, [multiplayer_price_avg, multiplayer_played_hours_avg, multiplayer_feedbacks_avg], width, label='MultiPlayer')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Scores by feature')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.show()

    # cheaper games more answered
    steam_EDA['all_ratings'] = steam_EDA['positive_ratings'] + steam_EDA['negative_ratings']
    # split into 11 categories according to prize
    intervals = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, steam_EDA['price'].max() + 1]
    all_ratings_avg = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    positive_ratings_avg = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    relative_ratings_avg = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(len(intervals) - 1):
        tmp = steam_EDA[(intervals[i] <= steam_EDA['price']) & (steam_EDA['price'] < intervals[i + 1])]
        all_ratings_avg[i] = tmp['all_ratings'].mean()
        positive_ratings_avg[i] = tmp['positive_ratings'].mean()
        relative_ratings_avg[i] = (tmp['positive_ratings'] / tmp['all_ratings']).mean()

    labels = ['0-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30-35', '35-40', '40-45', '45-50', '50+']
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # t
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, all_ratings_avg,
                    width, label='All ratings')
    rects2 = ax.bar(x + width / 2, positive_ratings_avg,
                    width, label='Positive')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Ratings')
    ax.set_title('Ratings by price group')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.show()

    # relative ratings
    fig, ax = plt.subplots()

    # Using set_dashes() to modify dashing of an existing line
    line1, = ax.plot(labels, relative_ratings_avg, label='Relative ratings)')
    #line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break

    ax.legend()
    plt.show()

    # what features contain 'classic' games TODO

    # ENCODE developer, publisher, genre
    # funs.publishers_developers_encoded(stream_data, './data4/steamPubDevEncoded.csv')
    encoded = funs.load_data('./data4/steamPubDevEncoded.csv')
    encoded['Multiplayer'] = 0
    encoded['Singleplayer'] = 0
    encoded['Action'] = 0
    encoded['Adventure'] = 0
    encoded['Indie'] = 0
    encoded['Strategy'] = 0
    encoded['RPG'] = 0
    encoded['Casual'] = 0
    encoded['Simulation'] = 0
    encoded['Racing'] = 0
    encoded['Sports'] = 0

    encoded['2D'] = 0
    encoded['3D'] = 0
    encoded['old_school'] = 0
    encoded['classic'] = 0
    encoded['war'] = 0
    encoded['e_sports'] = 0
    encoded['team_based'] = 0

    idx = 0
    for index in encoded.appid:
        if 'Single-player' in encoded.at[idx, 'categories']:
            encoded.at[idx, 'Singleplayer'] = 1
        if 'Multi-player' in encoded.at[idx, 'categories']:
            encoded.at[idx, 'Multiplayer'] = 1
        if 'Action' in encoded.at[idx, 'genres']:
            encoded.at[idx, 'Action'] = 1
        if 'Adventure' in encoded.at[idx, 'genres']:
            encoded.at[idx, 'Adventure'] = 1
        if 'Indie' in encoded.at[idx, 'genres']:
            encoded.at[idx, 'Indie'] = 1
        if 'Strategy' in encoded.at[idx, 'genres']:
            encoded.at[idx, 'Strategy'] = 1
        if 'RPG' in encoded.at[idx, 'genres']:
            encoded.at[idx, 'RPG'] = 1
        if 'Casual' in encoded.at[idx, 'genres']:
            encoded.at[idx, 'Casual'] = 1
        if 'Simulation' in encoded.at[idx, 'genres']:
            encoded.at[idx, 'Simulation'] = 1
        if 'Racing' in encoded.at[idx, 'genres']:
            encoded.at[idx, 'Racing'] = 1
        if 'Sports' in encoded.at[idx, 'genres']:
            encoded.at[idx, 'Sports'] = 1
        idx += 1

    cols_2d = [col for col in streamspy_data.columns if '2d' in col]
    idx = 0
    for id in encoded.appid:
        row_num = streamspy_data.index[streamspy_data['appid'] == id].tolist()[0]
        i = 0
        for col in cols_2d:
            col_num = streamspy_data.columns.get_loc(col)
            i += streamspy_data.iat[row_num, col_num]
        if i > 0:
            encoded.at[idx, '2D'] = 1
        idx += 1
    # NORMALIZE

    ########################################################################
    # 2 - CLUSTERING########################################################
    ########################################################################
    # CLUSTERING 1 - number of clusters specified

    # CLUSTERING 2 - number of clusters unspecified

    end = 1
