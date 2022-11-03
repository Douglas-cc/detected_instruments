import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyod.models.knn import KNN
from sklearn.manifold import TSNE


def detected_outilers(dataframe):
    clf = KNN()
    clf.fit(dataframe)
    '''
    labels binarias (0: inliers, 1: outliers) de previsão 
    e pontuações discrepantes dos dados de treinamento
    '''
    return clf.labels_ 


def graphic_anomalies(dataframe, title):
    result_f = TSNE(n_components = 2, random_state=1).fit_transform(dataframe)
    result_f = pd.DataFrame(result_f)

    pred = detected_outilers(dataframe)
    color= ['orange' if row == 1 else 'purple' for row in pred]

    plt.scatter(result_f[0],result_f[1], s=1, c=color)
    plt.title(title)


def remove_outilers(dataframe):
    pred = detected_outilers(dataframe)
    outilers_id =[pred[i] == 0 for i in range(pred.shape[0])]
    return dataframe.iloc[outilers_id,:]


def df_features_corr(df, limit_inf, limit_sup):
    df_corr = df.corr().unstack().reset_index().dropna()

    df_corr.rename(
        columns = {'level_0': 'features_a', 'level_1': 'features_b', 0:'correlacao'}, 
        inplace = True
    )

    df_corr = df_corr.query(f'correlacao > {limit_inf} and correlacao < {limit_sup}')
    features_select = df_corr.features_b.unique()
    features = np.concatenate((features_select, ['labels', 'instrumento']))
    
    return df[features]

def features_corr(dataframe, limit_inf, limit_sup):
    df_corr = dataframe.corr().unstack().reset_index().dropna()

    df_corr.rename(
        columns = {'level_0': 'features_a', 'level_1': 'features_b', 0:'correlacao'}, 
        inplace = True
    )

    df_corr = df_corr.query(f'correlacao > {limit_inf} and correlacao < {limit_sup}')
    features_select = df_corr.features_b.unique()
    features = np.concatenate((features_select, ['labels', 'instrumento']))
    
    return dataframe[features]
