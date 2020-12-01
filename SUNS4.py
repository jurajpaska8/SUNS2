from random import randrange

from sklearn import preprocessing
from sklearn.cluster import MiniBatchKMeans, DBSCAN
from sklearn.decomposition import PCA

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
    streamspy_data = funs.load_data('./data4/steamspy_tag_data.csv')

    # MERGE
    merged_data = pd.merge(stream_data, streamspy_data, on="appid", right_index=True, left_index=True)

    # EDA - three questions
    # multiplayer more expensive, played, ranked in general
    # find multiplayers and exports it
    # funs.export_multiplayer_games(stream_data, streamspy_data, './data4/multiplayer.csv')
    steam_EDA = funs.load_data('./data4/multiplayer.csv')
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
    line1, = ax.plot(labels, relative_ratings_avg, label='Relative ratings)')
    ax.legend()
    plt.show()

    # ENCODE developer, publisher, genre
    # # funs.publishers_developers_encoded(stream_data, './data4/steamPubDevEncoded.csv')
    # encoded = funs.load_data('./data4/steamPubDevEncoded.csv')

    # # add features
    # encoded = funs.add_features(encoded)

    # # categories and genres from steam data
    # funs.category_encoded(encoded)

    # # tags from steam spy data
    # funs.category_encoded_tags(encoded, streamspy_data, '2d')
    # funs.category_encoded_tags(encoded, streamspy_data, '3d')
    # funs.category_encoded_tags(encoded, streamspy_data, 'old_school')
    # funs.category_encoded_tags(encoded, streamspy_data, 'classic')
    # funs.category_encoded_tags(encoded, streamspy_data, 'war')
    # funs.category_encoded_tags(encoded, streamspy_data, 'e_sports')
    # funs.category_encoded_tags(encoded, streamspy_data, 'team_based')
    # encoded.to_csv('./data4/encoded.csv')

    final = funs.load_data('./data4/final.csv')
    final['release_date'] = final['release_date'].str.split('-', expand=False).str[0].astype(int)
    final['owners'] = final['owners'].str.split('-', expand=False).str[1].astype(int)
    final_dropped = final.drop(columns=['appid', 'name', 'platforms', 'required_age', 'categories', 'genres', 'owners', 'english', 'Multiplayer', 'Singleplayer', 'positive_ratings', 'negative_ratings', 'median_playtime'])

    # what features contain 'classic' games
    corr = final.corr()
    corr_dropped = final_dropped.corr()

    # NORMALIZE
    scaler = preprocessing.StandardScaler()
    df_scaled = scaler.fit_transform(final_dropped)

    ########################################################################
    # 2 - CLUSTERING########################################################
    ########################################################################
    # CLUSTERING 1 - number of clusters specified
    # k means clustering
    n_clusters_kmeans = 10
    kmenas = MiniBatchKMeans(n_clusters_kmeans, init='k-means++', max_iter=500, n_init=10, verbose=True, max_no_improvement=50, batch_size=5000)
    kluster_model = kmenas.fit(df_scaled)
    labels_kmeans = kmenas.labels_
    final_dropped['cluster_id_kmeans'] = kmenas.labels_
    funs.visualisation(final_dropped, n_clusters_kmeans, 'cluster_id_kmeans')

    # CLUSTERING 2 - number of clusters unspecified
    # Compute DBSCAN
    db = DBSCAN(eps=3.5, min_samples=50).fit(df_scaled)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels_dbscan = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
    n_noise_ = list(labels_dbscan).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_dbscan)
    print('Estimated number of noise points: %d' % n_noise_)
    final_dropped['cluster_id_dbscan'] = labels_dbscan

    funs.visualisation(final_dropped, n_clusters_dbscan, 'cluster_id_dbscan')

    ########################################################################
    # 3 - 2D/3D ############################################################
    ########################################################################
    pca = PCA(n_components=3)
    x_pca = pca.fit(final_dropped.drop(columns=['cluster_id_dbscan'])).transform(final_dropped.drop(columns=['cluster_id_dbscan']))

    colors = ['navy', 'turquoise', 'darkorange', 'red', 'green', 'blue', 'yellow', 'dimgray', 'lime', 'hotpink']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    idx = 0
    plotted = []
    for x in x_pca[:5000]:
        xs = x[0]
        ys = x[1]
        zs = x[2]
        if labels_kmeans[idx] in plotted:
            ax.scatter(xs, ys, zs, color=colors[labels_kmeans[idx]])
        else:
            ax.scatter(xs, ys, zs, color=colors[labels_kmeans[idx]], label=labels_kmeans[idx])
            plotted.append(labels_kmeans[idx])
        idx += 1

    ax.set_xlabel('X[0]')
    ax.set_ylabel('X[1]')
    ax.set_zlabel('X[2]')
    ax.legend()
    plt.show()

    # 2d
    pca_2d = PCA(n_components=2)
    X_r = pca_2d.fit(final_dropped.drop(columns=['cluster_id_dbscan'])).transform(final_dropped.drop(columns=['cluster_id_dbscan']))

    fig = plt.figure()
    lw = 2

    plotted = []
    idx = 0
    for x in X_r[:1000]:
        xs = X_r[0]
        ys = X_r[1]

        if (labels_kmeans[idx] in plotted):
            plt.scatter(xs, ys, color=colors[labels_kmeans[idx]], alpha=.8, lw=lw)
        else:
            plt.scatter(xs, ys, color=colors[labels_kmeans[idx]], alpha=.8, lw=lw,
                        label=labels_kmeans[idx])
            plotted.append(labels_kmeans[idx])
        idx += 1

    plt.legend()
    plt.show()
    end = 1
