from models import *
from cluster import *
FILE_PATH = ".//Dataset//hotel_bookings.csv"

if __name__ == "__main__":
    acc_list = []
    err_list = []
    model_list_classifier = ['Decision Tree Classifier', 'Random Forest Classifier',
                             'Gradient Boosting Classifier', 'Ada Boost Classifier']
    model_list_regressor = ['Linear Regression', 'Ridge Regression', 'Ada Boost Regressor', 'Random Forest Regression']

    # preprocessing
    preproc_path = preprocess_to_hotel_data_model(FILE_PATH)
    dataModel = get_df(preproc_path)
    # findCorrelation(dataModel)

    # finding best params with GridSearch (classifier)
    print(f"Decision Tree Best Params: {str(decisiontreecl_bestparams(dataModel))}")
    print(f"Random Forest Best Params: {str(random_forestcl_bestparams(dataModel))}")
    print(f"GradientBoosting Best Params: {str(gradientboosting_bestparams(dataModel))}")
    print(f"Ada Boost Best Params: {str(adaboost_cl_bestparams(dataModel))}")

    # finding best params with GridSearch (regressor)
    print(f"Linear Regression Best Params: {str(linearregression_bestparams(dataModel))}")
    print(f"Ridge Regression Best Params: {str(ridge_bestparams(dataModel))}")
    print(f"AdaBoost Regressor Best Params: {str(adaboostregression_bestparams(dataModel))}")
    print(f"Random Forest Regressor Best Params: {str(randomforest_regression_bestparams(dataModel))}")

    # finding stochastic best params with GridSearch (classifier)
    print(f"Gaussian Naive Bayes Best Params: {str(gaussiannaivecl_bestparams(dataModel))}")

    # learning Classification
    acc_list.append(decisiontree_cl(dataModel))
    acc_list.append(randomforest_cl(dataModel))
    acc_list.append(grandientboosting_cl(dataModel))
    acc_list.append(adaboost_cl(dataModel))

    # comparison models
    compare_models(param="F-1 Score", param_list=acc_list, models_list=model_list_classifier)

    # learning Regression
    err_list.append(linearegression_reg(dataModel))
    err_list.append(ridgeregression_rg(dataModel))
    err_list.append(adaboostregressor_rg(dataModel))
    err_list.append(randomforestregressor_rg(dataModel))

    # comparison models
    compare_models(param="MAE", param_list=err_list, models_list=model_list_regressor)

    # clustering
    kmeans_elbow(FILE_PATH)
    kmeans_model(FILE_PATH)

    # learning stochastic classification
    gaussianaivebayes_cl(dataModel)
    multinomialnaivebayes_cl(dataModel)
    bernoullinaivebayes_cl(dataModel)
