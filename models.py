from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score, roc_auc_score, roc_curve, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import plotly.express as px
import seaborn as sb
import numpy as np
from preprocessing import *


# finding the best parameters for classification models
def random_forestcl_bestparams(hotel_data_tuning):
    y = hotel_data_tuning["is_canceled"]
    X = hotel_data_tuning.drop(["is_canceled"], axis=1)

    # Finding parameters for RF model
    model_rfc_gs = RandomForestClassifier()
    parameters_rfc = {
        'n_estimators': [100, 200, 500],
        'min_samples_split': [2, 4, 6, 8],
        'min_samples_leaf': [1, 2, 4, 6]
    }

    grid_search_rfc = GridSearchCV(estimator=model_rfc_gs, param_grid=parameters_rfc,
                                   cv=5, scoring='f1', verbose=True, n_jobs=-1)
    grid_search_rfc.fit(X, y)
    return grid_search_rfc.best_params_


def decisiontreecl_bestparams(hotel_data_tuning):
    y = hotel_data_tuning["is_canceled"]
    X = hotel_data_tuning.drop(["is_canceled"], axis=1)

    # Finding parameters for Decision Tree
    model_dtc_gs = DecisionTreeClassifier()
    parameters_dtc = {
        'criterion': ['gini', 'entropy'],
        'min_samples_split': [2, 4, 6, 8],
        'min_samples_leaf': [1, 2, 3, 4, 5],
        'max_features': ['auto', 'sqrt']
    }

    grid_search_dtc = GridSearchCV(estimator=model_dtc_gs, param_grid=parameters_dtc,
                                   cv=5, scoring='f1', verbose=True, n_jobs=-1)
    grid_search_dtc.fit(X, y)
    return grid_search_dtc.best_params_


def gradientboosting_bestparams(hotel_data_tuning):
    y = hotel_data_tuning["is_canceled"]
    X = hotel_data_tuning.drop(["is_canceled"], axis=1)

    gbc = GradientBoostingClassifier()
    parameters = {
        "learning_rate": [0.1, 0.15, 0.2],
        "min_samples_split": np.linspace(0.1, 0.5, 12),
        "min_samples_leaf": np.linspace(0.1, 0.5, 12),
        "max_depth": [3, 5, 8],
        "max_features": ["log2", "sqrt"],
        "criterion": ["friedman_mse", "mae"],
        "subsample": [0.9, 0.95, 1.0],
    }
    grid_search_gbc = GridSearchCV(estimator=gbc, param_grid=parameters, scoring='f1', verbose=True, cv=5, n_jobs=-1)
    grid_search_gbc.fit(X, y)
    return grid_search_gbc.best_params_


def adaboost_cl_bestparams(hotel_data_tuning):
    y = hotel_data_tuning["is_canceled"]
    X = hotel_data_tuning.drop(["is_canceled"], axis=1)

    abcl = AdaBoostClassifier()
    parameters = {'base_estimator__min_samples_split': [2, 3, 4, 5],
                  'base_estimator__min_samples_leaf': [1, 2, 3, 4, 5],
                  'base_estimator__max_depth': [2, 3, 4, 5], 'n_estimators': [100, 200, 300, 400, 500],
                  'learning_rate': [0.01, 0.1]
                  }
    grid_search_abcl = GridSearchCV(estimator=abcl, param_grid=parameters, scoring='f1', verbose=True, cv=5, n_jobs=-1)
    grid_search_abcl.fit(X, y)
    return grid_search_abcl.best_params_


# finding the best parameters for regression models
def ridge_bestparams(hotel_data_tuning):
    y = hotel_data_tuning["lead_time"]
    X = hotel_data_tuning.drop(["lead_time"], axis=1)
    parameters = {'alpha': [50, 75, 100, 200, 230, 250],
                  'random_state': [5, 10, 20, 50, ],
                  'max_iter': [0.1, 0.5, 1, 2, 3, 5]
                  }
    ridge = Ridge()
    grid_search_ridge = GridSearchCV(estimator=ridge, param_grid=parameters, scoring='f1', verbose=True, cv=5,
                                     n_jobs=-1)
    grid_search_ridge.fit(X, y)
    return grid_search_ridge.best_params_


def linearregression_bestparams(hotel_data_tuning):
    y = hotel_data_tuning["lead_time"]
    X = hotel_data_tuning.drop(["lead_time"], axis=1)
    parameters = {"copy_X": [True, False],
                  "fit_intercept": [True, False],
                  }
    lin = LinearRegression()
    grid_search_lin = GridSearchCV(estimator=lin, param_grid=parameters, scoring='f1', verbose=True, cv=5, n_jobs=-1)
    grid_search_lin.fit(X, y)
    return grid_search_lin.best_params_


def adaboostregression_bestparams(hotel_data_tuning):
    y = hotel_data_tuning["lead_time"]
    X = hotel_data_tuning.drop(["lead_time"], axis=1)
    parameters = {'n_estimators': [500, 1000, 2000],
                  'learning_rate': [.001, 0.01, .1],
                  'random_state': [1]
                  }
    ada = AdaBoostRegressor()
    grid_search_ada = GridSearchCV(estimator=ada, param_grid=parameters, scoring='f1', verbose=True, cv=5, n_jobs=-1)
    grid_search_ada.fit(X, y)
    return grid_search_ada.best_params_


def randomforest_regression_bestparams(hotel_data_tuning):
    y = hotel_data_tuning["lead_time"]
    X = hotel_data_tuning.drop(["lead_time"], axis=1)
    parameters = {
        'n_estimators': [100, 200, 500],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [4, 5, 6, 7, 8],
        'criterion': ['mse', 'mae']
    }
    rf_reg = RandomForestRegressor()
    grid_search_rf = GridSearchCV(estimator=rf_reg, param_grid=parameters, scoring='f1', verbose=True, cv=5, n_jobs=-1)
    grid_search_rf.fit(X, y)
    return grid_search_rf.best_params_


# finding the best parameters for stochastic models
def gaussiannaivecl_bestparams(hotel_data_tuning):
    y = hotel_data_tuning["is_canceled"]
    X = hotel_data_tuning.drop(["is_canceled"], axis=1)

    # Finding parameters for GNV model
    model_gncl = GaussianNB()
    parameters = {'var_smoothing': np.logspace(0, -9, num=100)}

    grid_search_gncl = GridSearchCV(estimator=model_gncl, param_grid=parameters,
                                    cv=5, scoring='f1', verbose=True, n_jobs=-1)
    grid_search_gncl.fit(X, y)
    return grid_search_gncl.best_params_


# in dataframe_preprocessed, out X_train, X_test, y_train, y_test
def scalar_xy(X_model, y_model):
    X_train, X_test, y_train, y_test = train_test_split(X_model, y_model, test_size=0.3, random_state=42)

    # Implement standard scaler method
    standardScalerX = StandardScaler()
    X_train = standardScalerX.fit_transform(X_train)
    X_test = standardScalerX.fit_transform(X_test)

    return X_train, X_test, y_train, y_test


# Classification models

# decisionTree in -> preprocessed dataframe
def decisiontree_cl(p_df):
    y_model = p_df["is_canceled"]
    X_model = p_df.drop(['is_canceled'], axis=1)
    X_train, X_test, y_train, y_test = scalar_xy(X_model, y_model)
    # cross validation
    kfold_cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    for train_index, test_index in kfold_cv.split(X_model, y_model):
        X_train, X_test = X_model.iloc[train_index], X_model.iloc[test_index]
        y_train, y_test = y_model.iloc[train_index], y_model.iloc[test_index]

    dtc = DecisionTreeClassifier(criterion='entropy', min_samples_split=8, min_samples_leaf=5, max_features='auto')

    dtc.fit(X_train, y_train)
    dtc_predict = dtc.predict(X_test)

    dtc_accuracy = accuracy_score(y_test, dtc_predict)
    dtc_conf_matrix = confusion_matrix(y_test, dtc_predict)
    dtc_class_report = classification_report(y_test, dtc_predict)
    dtc_prob = dtc.predict_proba(X_test)[:, 1]
    f1 = f1_score(y_test, dtc_predict)

    print(f"Accuracy Score of Decision Tree is : {dtc_accuracy}")
    print(f"Confusion Matrix : \n{dtc_conf_matrix}")
    print(f"Classification Report : \n{dtc_class_report}")
    plot_confusionmatrix(dtc_conf_matrix)
    roc_plot(y_test, dtc_prob)
    return f1


# RandomForest
def randomforest_cl(p_df):
    y_model = p_df["is_canceled"]
    X_model = p_df.drop(['is_canceled'], axis=1)
    X_train, X_test, y_train, y_test = scalar_xy(X_model, y_model)
    # cross validation
    kfold_cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    for train_index, test_index in kfold_cv.split(X_model, y_model):
        X_train, X_test = X_model.iloc[train_index], X_model.iloc[test_index]
        y_train, y_test = y_model.iloc[train_index], y_model.iloc[test_index]

    rfcl = RandomForestClassifier(min_samples_leaf=6, min_samples_split=6, n_estimators=100)
    rfcl.fit(X_train, y_train)
    rfcl_predict = rfcl.predict(X_test)

    rfcl_prob = rfcl.predict_proba(X_test)[:, 1]
    rfcl_accuracy = accuracy_score(y_test, rfcl_predict)
    rfcl_conf_matrix = confusion_matrix(y_test, rfcl_predict)
    rfcl_class_report = classification_report(y_test, rfcl_predict)
    f1 = f1_score(y_test, rfcl_predict)

    print(f"Accuracy Score of Random Forest is : {rfcl_accuracy}")
    print(f"Confusion Matrix : \n{rfcl_conf_matrix}")
    print(f"Classification Report : \n{rfcl_class_report}")
    print(f"f-1: \n{f1}")
    plot_confusionmatrix(rfcl_conf_matrix)
    roc_plot(y_test, rfcl_prob)

    return f1


# GradientBoosting
def grandientboosting_cl(p_df):
    y_model = p_df["is_canceled"]
    X_model = p_df.drop(['is_canceled'], axis=1)
    X_train, X_test, y_train, y_test = scalar_xy(X_model, y_model)
    # cross validation
    kfold_cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    for train_index, test_index in kfold_cv.split(X_model, y_model):
        X_train, X_test = X_model.iloc[train_index], X_model.iloc[test_index]
        y_train, y_test = y_model.iloc[train_index], y_model.iloc[test_index]

    gb_cl = GradientBoostingClassifier()
    gb_cl.fit(X_train, y_train)
    gb_cl_predict = gb_cl.predict(X_test)

    gb_cl_prob = gb_cl.predict_proba(X_test)[:, 1]
    gb_cl_accuracy = accuracy_score(y_test, gb_cl_predict)
    gb_cl_conf_matrix = confusion_matrix(y_test, gb_cl_predict)
    gb_cl_class_report = classification_report(y_test, gb_cl_predict)
    f1 = f1_score(y_test, gb_cl_predict)

    print(f"Accuracy Score of Gradient Boosting is : {gb_cl_accuracy}")
    print(f"Confusion Matrix : \n{gb_cl_conf_matrix}")
    print(f"Classification Report : \n{gb_cl_class_report}")
    plot_confusionmatrix(gb_cl_conf_matrix)
    roc_plot(y_test, gb_cl_prob)

    return f1


# AdaBoost
def adaboost_cl(p_df):
    y_model = p_df["is_canceled"]
    X_model = p_df.drop(['is_canceled'], axis=1)
    X_train, X_test, y_train, y_test = scalar_xy(X_model, y_model)
    # cross validation
    kfold_cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    for train_index, test_index in kfold_cv.split(X_model, y_model):
        X_train, X_test = X_model.iloc[train_index], X_model.iloc[test_index]
        y_train, y_test = y_model.iloc[train_index], y_model.iloc[test_index]

    ab_cl = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
    ab_cl.fit(X_train, y_train)
    ab_cl_predict = ab_cl.predict(X_test)

    ab_cl_prob = ab_cl.predict_proba(X_test)[:, 1]
    ab_cl_accuracy = accuracy_score(y_test, ab_cl_predict)
    ab_cl_conf_matrix = confusion_matrix(y_test, ab_cl_predict)
    ab_cl_class_report = classification_report(y_test, ab_cl_predict)
    f1 = f1_score(y_test, ab_cl_predict)

    print(f"Accuracy Score of Ada Boosting is : {ab_cl_accuracy}")
    print(f"Confusion Matrix : \n{ab_cl_conf_matrix}")
    print(f"Classification Report : \n{ab_cl_class_report}")
    print(f"f-1: \n{f1}")
    plot_confusionmatrix(ab_cl_conf_matrix)
    roc_plot(y_test, ab_cl_prob)

    return f1


# Regression models

def linearegression_reg(p_df):
    y_model = p_df["lead_time"]
    X_model = p_df.drop(["lead_time"], axis=1)
    # X_model = bestFeaturesSelection_regression(X_model, y_model, 10)
    X_train, X_test, y_train, y_test = scalar_xy(X_model, y_model)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    mae = metrics.mean_absolute_error(y_test, y_pred).round(3)
    print("Linear Regression Error")
    print('Mean Absolute Error_lng:', mae)
    print('Mean Squared Error_lng:', metrics.mean_squared_error(y_test, y_pred).round(3))
    print('Root Mean Squared Error_lng:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)).round(3))
    print('r2_score_lng:', r2_score(y_test, y_pred).round(3))
    print('Max Error LinReg:', metrics.max_error(y_test, y_pred).round(3))

    regression_plot(y_test, y_pred)

    return mae


def ridgeregression_rg(p_df):
    y_model = p_df["lead_time"]
    X_model = p_df.drop(["lead_time"], axis=1)
    # X_model = bestFeaturesSelection_regression(X_model, y_model, 10)
    X_train, X_test, y_train, y_test = scalar_xy(X_model, y_model)

    ridge = Ridge(alpha=50, max_iter=0.1, random_state=5)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)

    mae = metrics.mean_absolute_error(y_test, y_pred)
    print("Ridge Regression Error")
    print('Mean Absolute Error_ridge:', mae)
    print('Mean Squared Error_ridge:', metrics.mean_squared_error(y_test, y_pred).round(3))
    print('Root Mean Squared Error_ridge:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)).round(3))
    print('Max Error Ridge:', metrics.max_error(y_test, y_pred).round(3))
    print('r2_score_ridge:', r2_score(y_test, y_pred).round(3))

    regression_plot(y_test, y_pred)

    return mae


def adaboostregressor_rg(p_df):
    y_model = p_df["lead_time"]
    X_model = p_df.drop(["lead_time"], axis=1)

    # X_model = bestFeaturesSelection_regression(X_model, y_model, 10)
    X_train, X_test, y_train, y_test = scalar_xy(X_model, y_model)

    ABR = AdaBoostRegressor(n_estimators=100, random_state=42)

    # fit the regressor with x and y data
    ABR.fit(X_train, y_train)
    y_pred = ABR.predict(X_test)

    mae = metrics.mean_absolute_error(y_test, y_pred)
    print("AdaBoost Regression Error")
    print('Mean Absolute Error:', mae)
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('Max Error AdaBoost:', metrics.max_error(y_test, y_pred).round(3))
    print('r2_score_ABR:', r2_score(y_test, y_pred).round(3))

    regression_plot(y_test, y_pred)

    return mae


def randomforestregressor_rg(p_df):
    rfe = RandomForestRegressor(n_estimators=100, random_state=42)

    y_model = p_df["lead_time"]
    X_model = p_df.drop(["lead_time"], axis=1)

    # X_model = bestFeaturesSelection_regression(X_model, y_model, 10)
    X_train, X_test, y_train, y_test = scalar_xy(X_model, y_model)

    # fit the regressor with x and y data
    rfe.fit(X_train, y_train)
    y_pred = rfe.predict(X_test)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    print('Mean Absolute Error:', mae)
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('r2_score_RFE:', r2_score(y_test, y_pred).round(3))

    regression_plot(y_test, y_pred)

    return mae


# stochastic models

def gaussianaivebayes_cl(p_df):
    y_model = p_df["is_canceled"]
    X_model = p_df.drop(['is_canceled'], axis=1)
    X_train, X_test, y_train, y_test = scalar_xy(X_model, y_model)
    # cross validation
    kfold_cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

    for train_index, test_index in kfold_cv.split(X_model, y_model):
        X_train, X_test = X_model.iloc[train_index], X_model.iloc[test_index]
        y_train, y_test = y_model.iloc[train_index], y_model.iloc[test_index]

    gbcl = GaussianNB(var_smoothing=2e-9)
    gbcl.fit(X_train, y_train)
    y_pred = gbcl.predict(X_test)

    gbcl_prob = gbcl.predict_proba(X_test)[:, 1]
    gbcl_accuracy = accuracy_score(y_test, y_pred)
    gbcl_conf_matrix = confusion_matrix(y_test, y_pred)
    gbcl_class_report = classification_report(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy Score of Gaussian Naive Bayes is : {gbcl_accuracy}")
    print(f"Confusion Matrix : \n{gbcl_conf_matrix}")
    print(f"Classification Report : \n{gbcl_class_report}")
    print(f"F-1: {f1}")

    plot_confusionmatrix(gbcl_conf_matrix)
    roc_plot(y_test, gbcl_prob)


def multinomialnaivebayes_cl(p_df):
    y_model = p_df["is_canceled"]
    X_model = p_df.drop(['is_canceled'], axis=1)
    X_train, X_test, y_train, y_test = scalar_xy(X_model, y_model)
    # cross validation
    kfold_cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

    for train_index, test_index in kfold_cv.split(X_model, y_model):
        X_train, X_test = X_model.iloc[train_index], X_model.iloc[test_index]
        y_train, y_test = y_model.iloc[train_index], y_model.iloc[test_index]

    gbcl = MultinomialNB(alpha=0.6)
    gbcl.fit(X_train, y_train)
    y_pred = gbcl.predict(X_test)

    gbcl_prob = gbcl.predict_proba(X_test)[:, 1]
    gbcl_accuracy = accuracy_score(y_test, y_pred)
    gbcl_conf_matrix = confusion_matrix(y_test, y_pred)
    gbcl_class_report = classification_report(y_test, y_pred)

    print(f"Accuracy Score of Multinomial Naive Bayes is : {gbcl_accuracy}")
    print(f"Confusion Matrix : \n{gbcl_conf_matrix}")
    print(f"Classification Report : \n{gbcl_class_report}")

    plot_confusionmatrix(gbcl_conf_matrix)
    roc_plot(y_test, gbcl_prob)


def bernoullinaivebayes_cl(p_df):
    y_model = p_df["is_canceled"]
    X_model = p_df.drop(['is_canceled'], axis=1)
    X_train, X_test, y_train, y_test = scalar_xy(X_model, y_model)
    # cross validation
    kfold_cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

    for train_index, test_index in kfold_cv.split(X_model, y_model):
        X_train, X_test = X_model.iloc[train_index], X_model.iloc[test_index]
        y_train, y_test = y_model.iloc[train_index], y_model.iloc[test_index]

    gbcl = BernoulliNB(fit_prior=False)
    gbcl.fit(X_train, y_train)
    y_pred = gbcl.predict(X_test)

    gbcl_prob = gbcl.predict_proba(X_test)[:, 1]
    gbcl_accuracy = accuracy_score(y_test, y_pred)
    gbcl_conf_matrix = confusion_matrix(y_test, y_pred)
    gbcl_class_report = classification_report(y_test, y_pred)

    print(f"Accuracy Score of Bernoulli Naive Bayes is : {gbcl_accuracy}")
    print(f"Confusion Matrix : \n{gbcl_conf_matrix}")
    print(f"Classification Report : \n{gbcl_class_report}")

    plot_confusionmatrix(gbcl_conf_matrix)
    roc_plot(y_test, gbcl_prob)

# Plotting graphs

def plot_confusionmatrix(m):
    ax = sb.heatmap(m / np.sum(m), annot=True,
                    fmt='.2%', cmap='Blues')

    ax.set_title('Confusion Matrix\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')

    # Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])

    # Display the visualization of the Confusion Matrix.
    plt.show()
    plt.clf()


def roc_plot(y_test, y_prob):
    false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    plt.plot(false_positive_rate, true_positive_rate, label="AUC=" + str(roc_auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()
    plt.clf()


def regression_plot(true_value, predicted_value):
    plt.figure(figsize=(5, 5))
    plt.scatter(true_value, predicted_value, c='crimson')
    plt.yscale('log')
    plt.xscale('log')

    p1 = max(max(predicted_value), max(true_value))
    p2 = min(min(predicted_value), min(true_value))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')
    plt.show()
    plt.clf()


# models comparison

def compare_models(param, param_list, models_list):
    models_df = pd.DataFrame({
        'Model': models_list,
        param: param_list
    })
    models_df.sort_values(by=param, ascending=False)
    px.bar(data_frame=models_df, x='Model', y=param, color=param, template='plotly_dark',
           title='Models Comparison').show()
