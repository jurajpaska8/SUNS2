import funs
import pandas as pd

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
    # find multiplayers and export it
    # stream_EDA = stream_data.copy(deep=True)
    # stream_EDA['multiplayer'] = 0
    #
    # multiplayer_cols = [col for col in streamspy_data.columns if 'multiplayer' in col]
    # idx = 0
    # for id in stream_EDA.appid:
    #     row_num = streamspy_data.index[streamspy_data['appid'] == id].tolist()[0]
    #     i = 0
    #     for col in multiplayer_cols:
    #         col_num = streamspy_data.columns.get_loc(col)
    #         i += streamspy_data.iat[row_num, col_num]
    #     if i > 0:
    #         stream_EDA.at[idx, 'multiplayer'] = 1
    #     idx += 1
    # stream_EDA.to_csv('./data4/multiplayer.csv')

    steam_EDA = funs.load_data('./data4/multiplayer.csv')


    # cheaper games more answered
    # what features contain 'classic' games

    # ENCODE developer, publisher, genre

    # NORMALIZE

    ########################################################################
    # 2 - CLUSTERING########################################################
    ########################################################################
    # CLUSTERING 1 - number of clusters specified

    # CLUSTERING 2 - number of clusters unspecified

    end = 1
