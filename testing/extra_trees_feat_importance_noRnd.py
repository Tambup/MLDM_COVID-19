import os
import numpy as np
import pandas as pd

from datetime import datetime

from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.metrics import r2_score, mean_squared_error, max_error
from sklearn.ensemble import ExtraTreesRegressor


import numpy as np
import pandas as pd

from datetime import datetime

from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.metrics import r2_score, mean_squared_error, max_error
from sklearn.ensemble import ExtraTreesRegressor

datasets_names = ['dataset_finale_medie.csv', 'dataset_finale_medie.csv', 'dataset_finale_medie_1_week.csv', 'dataset_finale_medie_1_week.csv', 
        'dataset_finale_medie_2_week.csv', 'dataset_finale_medie_2_week.csv', 'dataset_finale_medie_1_week.csv', 'dataset_finale_medie_1_week.csv', 'dataset_finale_medie_1_week.csv',
        'dataset_finale_medie_1_week.csv', 'dataset_finale_medie.csv', 'dataset_finale_medie.csv']

rnd_states = [22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22]
features_prediction = ['fut1_nuovi_positivi','fut2_nuovi_positivi',
                     'fut1_ricoverati_con_sintomi','fut2_ricoverati_con_sintomi',
                     'fut1_terapia_intensiva','fut2_terapia_intensiva',
                     'fut1_deceduti','fut2_deceduti',
                     'fut1_totale_ospedalizzati','fut2_totale_ospedalizzati',
                     'fut1_dimessi_guariti','fut2_dimessi_guariti']

coefficents = ['R2','RMSE','MAXERR']


predictor_columns = [
            'pass_Ammoniaca', 
            'pass_Benzene',
            'pass_Biossido di Azoto', 
            'pass_Biossido di Zolfo',
            'pass_Monossido di Azoto', 
            'pass_Monossido di Carbonio',
            'pass_Ossidi di Azoto', 
            'pass_Ozono', 
            'pass_PM10 (SM2005)',
            'pass_Particelle sospese PM2.5',
            'pass_Radiazione Globale', 
            'pass_Temperatura',
            'pass_deceduti',
            'pass_nuovi_positivi',
            'pass_ricoverati_con_sintomi', 
            'pass_tamponi']

n_estimators_list = [100]
criterion_list = ["mse", "mae"]
max_depth_list = [None, 5, 6]
min_samples_split_list = [2, 3]
min_samples_split_leaf_list = [3,5]
min_weight_fraction_leaf_list = [0.0, 0.1]
max_features_list = ["auto", "sqrt", "log2"]
max_leaf_nodes_list = [None, 10, 25]
min_impurity_decrease_list = [0.0, 0.1]

param_grd = { "n_estimators":n_estimators_list,
                "criterion": criterion_list,
                "max_depth": max_depth_list,
                "min_samples_split": min_samples_split_list,
                "min_samples_leaf": min_samples_split_leaf_list,
                "min_weight_fraction_leaf": min_weight_fraction_leaf_list,
                "max_features": max_features_list,
                "max_leaf_nodes": max_leaf_nodes_list,
                "min_impurity_decrease": min_impurity_decrease_list}

fi = open('result_feature_importance_noRnd.txt', 'w')

#INIZIO EFFETTIVO DELL'ESECUZIONE
for i in range(len(datasets_names)):  
    print(str(i))
    df_2 = pd.read_csv("/home/mldm/covid_BRTT/dataset_finali/"+datasets_names[i], parse_dates=["Data"], infer_datetime_format=True)
    rnd_state = rnd_states[i]
    ################################################################################
    train_2 = df_2.sample(frac=0.70, random_state=rnd_state)
    test_2 = df_2.drop(train_2.index)

    train_X_2 = train_2[predictor_columns]
    train_y_2 = train_2[features_prediction[i]]

    test_X_2 = test_2[predictor_columns]
    test_y_2 = test_2[features_prediction[i]]
    ################################################################################
    extra_tree_regressor = ExtraTreesRegressor(bootstrap=False,random_state=rnd_state) 
    ################################################################################
    imp=SimpleImputer(missing_values=np.nan, strategy="mean")
    imp=imp.fit(train_X_2)
    ################################################################################
    grid_regressor_2 = GridSearchCV(extra_tree_regressor, param_grd, 
                                        n_jobs=-1, 
                                        verbose=0)
    ################################################################################
    grid_regressor_2.fit(imp.transform(train_X_2), train_y_2)
    ################################################################################
    best_regressor_2 = grid_regressor_2.best_estimator_

    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    imputer = imputer.fit(test_X_2)
    test_X_imp = imputer.transform(test_X_2)

    predicted_2 = best_regressor_2.predict(test_X_imp)
    ################################################################################
    r2=float("{:.3f}".format(r2_score(test_y_2, predicted_2)))
    rmse=float("{:.3f}".format(mean_squared_error(test_y_2, predicted_2, squared=False)))
    maxerr=float("{:.3f}".format(max_error(test_y_2, predicted_2)))
    ################################################################################
    feature_importance = dict(zip(predictor_columns, best_regressor_2.feature_importances_))
    ordered_ft_importance = ""
    for w in sorted(feature_importance, key=feature_importance.get, reverse=True):
        ordered_ft_importance += str(w) + ": " + str(feature_importance[w]) + "\t"

    fi.write(datasets_names[i] + ":" + str(rnd_state) + " -> " + "R2: " + str(r2) + "RMSE: " + str(rmse) +
	"MAX_ERR: " + str(maxerr) + "feature_importance: {" + ordered_ft_importance + " }\n\n")

fi.close()
