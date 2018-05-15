#!/usr/bin/env python
#!-*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def cullum_check(cust_df):
    del(cust_df['Channel'])
    del(cust_df['Region'])
    return cust_df
    
def data_conv(cust_df):

    cust_array = np.array([cust_df["Fresh"].tolist(),
                           cust_df["Milk"].tolist(),
                           cust_df["Grocery"].tolist(),
                           cust_df["Frozen"].tolist(),
                           #cust_df["Milk"].tolist(),
                           cust_df["Detergents_Paper"].tolist(),
                           cust_df["Delicassen"].tolist()],
                           np.int32)
                           
    cust_array = cust_array.T
    return cust_array
    
def predition(cust_array):

    pred = KMeans(n_clusters=4).fit_predict(cust_array)
    return pred
    
def cluster_addition(cust_def, pred):
    cust_df['cluster_id'] = pred
    print (cust_df['cluster_id'].value_counts())
    return cust_df
        
def cluster2plt(cust_df):

    clusterinfo = pd.DataFrame()
    for i in range(4):
        clusterinfo['cluster' + str(i)] = cust_df[cust_df['cluster_id'] == i].mean()
    clusterinfo = clusterinfo.drop('cluster_id')
    
    my_plot = clusterinfo.T.plot(kind='bar', stacked=True, title="Mean_Value_of_4_Clusters")
#    my_plot.set_xticklabels(my_plot.xaxis.get_majorticklabels(), rotation=0) 
    plt.show()
    
if __name__== "__main__":

#   cust_df = pd.read_csv("http://pythondatascience.plavox.info/wp-content/uploads/2016/05/Wholesale_customers_data.csv")
   cust_df = pd.read_csv("./Wholesale_customers_data.csv")
   clean_cust_df = cullum_check(cust_df)
   convert_array = data_conv(clean_cust_df)
   pred_data = predition(convert_array)
   additional_cust = cluster_addition(clean_cust_df,pred_data)
   cluster2plt(additional_cust)
