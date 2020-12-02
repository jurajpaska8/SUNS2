from sklearn import preprocessing
import pandas as pd
import funs
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # steam_data = funs.load_data('./data4/final.csv')
    # funs.platforms_encode(steam_data)
    steam_data = funs.load_data('./data4/platforms.csv')

    steam_data_dropped = steam_data.drop(columns=['appid', 'name', 'platforms', 'required_age', 'categories', 'genres', 'owners', 'english', 'Multiplayer', 'Singleplayer', 'positive_ratings', 'negative_ratings', 'median_playtime'])

    # scaler = preprocessing.StandardScaler()
    # steam_data_dropped_scaled = pd.DataFrame(scaler.fit_transform(steam_data_dropped), columns=steam_data_dropped.columns)

    corr_steam = steam_data.corr()
    corr_steam_dropped = steam_data_dropped.corr()
    # corr_steam_data_dropped_scaled = steam_data_dropped_scaled.corr()

    win = steam_data_dropped.loc[steam_data_dropped['windows'] == 1]
    without_win = steam_data_dropped.loc[steam_data_dropped['windows'] == 0]

    linux = steam_data_dropped.loc[steam_data_dropped['linux'] == 1]
    without_linux = steam_data_dropped.loc[steam_data_dropped['linux'] == 0]

    mac = steam_data_dropped.loc[steam_data_dropped['mac'] == 1]
    without_mac = steam_data_dropped.loc[steam_data_dropped['mac'] == 0]

    labels = ['windows', 'linux', 'mac']
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    #param = 'price'
    #param = 'average_playtime'
    param = 'all_ratings'
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, [win[param].mean(), linux[param].mean(), mac[param].mean()],
                    width, label='With')
    rects2 = ax.bar(x + width / 2, [without_win[param].mean(), without_linux[param].mean(), without_mac[param].mean()],
                    width, label='Without')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('Price in euros')
    # ax.set_title('Price for game with/without supported OS')
    # ax.set_ylabel('Average playtime in hours')
    # ax.set_title('Average playtime for game with/without supported OS')
    ax.set_ylabel('Average ratings')
    ax.set_title('Average ratings for game with/without supported OS')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.show()

    end = 1