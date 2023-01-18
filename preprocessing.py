import pandas as pd
from os import path
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Creating new feature: `Weekday vs Weekend`
pd.options.mode.chained_assignment = None


def week_function(feature1, feature2, data_source):
    data_source['weekend_or_weekday'] = 0
    for i in range(0, len(data_source)):
        if feature2.iloc[i] == 0 and feature1.iloc[i] > 0:
            data_source['weekend_or_weekday'].iloc[i] = 'stay_just_weekend'
        if feature2.iloc[i] > 0 and feature1.iloc[i] == 0:
            data_source['weekend_or_weekday'].iloc[i] = 'stay_just_weekday'
        if feature2.iloc[i] > 0 and feature1.iloc[i] > 0:
            data_source['weekend_or_weekday'].iloc[i] = 'stay_both_weekday_and_weekend'
        if feature2.iloc[i] == 0 and feature1.iloc[i] == 0:
            data_source['weekend_or_weekday'].iloc[i] = 'undefined_data'


def get_df(csv_path):
    return pd.read_csv(csv_path)


# main preprocessing function
def preprocess_to_hotel_data_model(file_path):
    path_dir = path.dirname(file_path) + "//" + "processed_" + path.basename(file_path)
    # loading dataset
    if not path.isfile(file_path):
        print("Error: CSV Dataset not found")
        return
    elif path.exists(path_dir):
        print("Info: Preprocessed file exists")
        return path_dir

    hotel_data = pd.read_csv(file_path)

    print("Starting pre-processing...")

    # refactoring arrival_date_month letters -> numbers
    hotel_data['arrival_date_month'].replace({'January': '1',
                                              'February': '2',
                                              'March': '3',
                                              'April': '4',
                                              'May': '5',
                                              'June': '6',
                                              'July': '7',
                                              'August': '8',
                                              'September': '9',
                                              'October': '10',
                                              'November': '11',
                                              'December': '12'}, inplace=True)

    week_function(hotel_data['stays_in_weekend_nights'], hotel_data['stays_in_week_nights'], hotel_data)

    hotel_data['arrival_date_month'] = hotel_data['arrival_date_month'].astype('int64')

    # merging children and babies features
    hotel_data['all_children'] = hotel_data['children'] + hotel_data['babies']

    hotel_data['adr'] = hotel_data['adr'].astype(float)

    # fixing all missing data
    hotel_data['children'] = hotel_data['children'].fillna(0)
    hotel_data['all_children'] = hotel_data['all_children'].fillna(0)
    hotel_data['country'] = hotel_data['country'].fillna(hotel_data['country'].mode().index[0])
    hotel_data['agent'] = hotel_data['agent'].fillna('0')
    hotel_data = hotel_data.drop(['company'], axis=1)

    # Changing data structure
    hotel_data['agent'] = hotel_data['agent'].astype(int)
    hotel_data['country'] = hotel_data['country'].astype(str)

    # encoding labels for categorical feature
    lblen = LabelEncoder()
    hotel_data['hotel'] = lblen.fit_transform(hotel_data['hotel'])
    hotel_data['arrival_date_month'] = lblen.fit_transform(hotel_data['arrival_date_month'])
    hotel_data['meal'] = lblen.fit_transform(hotel_data['meal'])
    hotel_data['country'] = lblen.fit_transform(hotel_data['country'])
    hotel_data['market_segment'] = lblen.fit_transform(hotel_data['market_segment'])
    hotel_data['distribution_channel'] = lblen.fit_transform(hotel_data['distribution_channel'])
    hotel_data['is_repeated_guest'] = lblen.fit_transform(hotel_data['is_repeated_guest'])
    hotel_data['reserved_room_type'] = lblen.fit_transform(hotel_data['reserved_room_type'])
    hotel_data['assigned_room_type'] = lblen.fit_transform(hotel_data['assigned_room_type'])
    hotel_data['deposit_type'] = lblen.fit_transform(hotel_data['deposit_type'])
    hotel_data['agent'] = lblen.fit_transform(hotel_data['agent'])
    hotel_data['customer_type'] = lblen.fit_transform(hotel_data['customer_type'])
    hotel_data['reservation_status'] = lblen.fit_transform(hotel_data['reservation_status'])
    hotel_data['weekend_or_weekday'] = lblen.fit_transform(hotel_data['weekend_or_weekday'])

    # Dropping some features from data
    hotel_data = hotel_data.drop(['reservation_status', 'children', 'reservation_status_date',
                                  'babies', 'total_of_special_requests', 'required_car_parking_spaces',
                                  'assigned_room_type', 'reserved_room_type', 'booking_changes',
                                  'is_repeated_guest'], axis=1)
    hotel_data.to_csv(path_dir, index=False)
    print("Preprocessing finished")

    return path_dir


# it plots the heatmap of features correlation
def find_correlation(df):
    plt.figure(figsize=(10, 10))
    # calculate the correlations
    correlations = df.corr()
    # plot the heatmap
    sns.heatmap(correlations, xticklabels=correlations.columns, yticklabels=correlations.columns, annot=True)
    # plot the cluster map
    sns.clustermap(correlations, xticklabels=correlations.columns, yticklabels=correlations.columns, annot=True)
    plt.show()
