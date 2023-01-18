import sklearn.cluster as cluster
from sklearn.preprocessing import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def preprocess_cluster(file_path):
    df = pd.read_csv(file_path)
    # dropping features with null values
    df = df[df['children'].notna()]
    df = df[df['country'].notna()]

    # features refactoring
    df["arrival_date_month"] = pd.to_datetime(df['arrival_date_month'], format='%B').dt.month
    df["arrival_date"] = pd.to_datetime({"year": df["arrival_date_year"].values,
                                         "month": df["arrival_date_month"].values,
                                        "day": df["arrival_date_day_of_month"].values})

    # dropping already preprocessed features
    df = df.drop(columns=['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month'])
    df["reservation_status_date"] = pd.to_datetime(df["reservation_status_date"], format='%Y-%m-%d')

    # handling features with null values
    for column in ['agent', 'company']:
        df[column] = df[column].fillna(df[column].mean())
    for column in ['arrival_date']:
        df[column] = df[column].fillna(df[column].mean())

    # dropping duplicates
    df.drop_duplicates(inplace=True)

    # transforming categorical features into numeric
    categoricalV = ["hotel", "meal", "country", "market_segment", "distribution_channel", "reserved_room_type",
                    "assigned_room_type", "deposit_type", "customer_type"]
    df[categoricalV[1:11]] = df[categoricalV[1:11]].astype('category')
    df[categoricalV[1:11]] = df[categoricalV[1:11]].apply(lambda x: LabelEncoder().fit_transform(x))
    df['hotel_Num'] = LabelEncoder().fit_transform(df['hotel'])
    print("Cluster Preprocessing Finished")
    return df


# plotting k-elbow graph
def kmeans_elbow(dataset_path):
    df_preprocessed = preprocess_cluster(dataset_path)
    df_Short = df_preprocessed[['lead_time', 'adr']]
    K = range(1, 12)
    wss = []
    print("Starting K-parameter hunting")
    for k in K:
        kmeans = cluster.KMeans(n_clusters=k, init="random")
        kmeans = kmeans.fit(df_Short)
        wss_iter = kmeans.inertia_
        wss.append(wss_iter)
    mycenters = pd.DataFrame({'Clusters': K, 'WSS': wss})
    sns.scatterplot(x='Clusters', y='WSS', data=mycenters, marker="+")


# plotting k-means graph
def kmeans_model(dataset_path):
    df_preprocessed = preprocess_cluster(dataset_path)
    kmeans = cluster.KMeans(n_clusters=4, init="random")
    print("Fitting Kmeans on 'lead time and ADR'")
    kmeans = kmeans.fit(df_preprocessed[['lead_time', 'adr']])
    df_preprocessed['Clusters'] = kmeans.labels_

    sns.lmplot(x="lead_time", y="adr", hue='Clusters', data=df_preprocessed)
    plt.ylim(0, 600)
    plt.xlim(0, 800)
    plt.show()
