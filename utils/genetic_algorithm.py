import numpy as np
import pandas as pd
import random
import decimal
import statsmodels.api as sm

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

data = pd.read_csv('data/WS1Data.csv', header=1)

# Data pre-processing
data['ad0_time_spent'] = data['End time'] - data['Start time']
data['ad1_time_spent'] = data['End time.1'] - data['Start time.1']
data['ad2_time_spent'] = data['End time.2'] - data['Start time.2']
data['ad3_time_spent'] = data['End time.3'] - data['Start time.3']
data['ad4_time_spent'] = data['End time.4'] - data['Start time.4']

data.rename(columns={'Ad':'ad0',
                     'Ad.1':'ad1',
                     'Ad.2':'ad2',
                     'Ad.3':'ad3',
                     'Ad.4':'ad4',
                     'Start time':'ad0_start_time',
                     'Start time.1':'ad1_start_time',
                     'Start time.2':'ad2_start_time',
                     'Start time.3':'ad3_start_time',
                     'Start time.4':'ad4_start_time'}, inplace=True)

# Data Modelling of User Clicks
ad0_df = pd.get_dummies(data['ad0'], prefix='ad0')
ad1_df = pd.get_dummies(data['ad1'], prefix='ad1')
ad2_df = pd.get_dummies(data['ad2'], prefix='ad2')
ad3_df = pd.get_dummies(data['ad3'], prefix='ad3')
ad4_df = pd.get_dummies(data['ad4'], prefix='ad4')

data = pd.concat([data, ad0_df, ad1_df, ad2_df, ad3_df, ad4_df], axis=1)

model_var = ['ad0_1','ad0_2','ad0_3','ad0_4','ad0_5',
             'ad1_1','ad1_2','ad1_3','ad1_4','ad1_5',
             'ad2_1','ad2_2','ad2_3','ad2_4','ad2_5',
             'ad3_1','ad3_2','ad3_3','ad3_4','ad3_5',
             'ad4_1','ad4_2','ad4_3','ad4_4','ad4_5',
             'ad0_start_time','ad1_start_time','ad2_start_time','ad3_start_time','ad4_start_time',
             'ad0_time_spent','ad1_time_spent','ad2_time_spent','ad3_time_spent','ad4_time_spent']

regression_data = data[model_var + ['User Clicks']].copy()
X = regression_data.iloc[:,:-1]
y = regression_data.iloc[:,-1]

def normalize_model_data(data):

    col_to_normalize = [col for col in data.columns if ('start_time' in col) | ('time_spent' in col)]
    data_norm = data.copy()
    for col in col_to_normalize:
        data_norm[col] = data_norm[col] / 24

    return data_norm

X_norm = normalize_model_data(X)

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor


# def grid_search(model, parameters, n_folds, scoring, X_train, y_train):
#     grid_search = GridSearchCV(model, param_grid=parameters, cv=n_folds, verbose=1,
#                                scoring=scoring, n_jobs=-1)
#
#     grid_search = grid_search.fit(X_train, y_train.values.ravel())
#
#     best_score = grid_search.best_score_
#     best_parameters = grid_search.best_params_
#     print(best_score)
#     print(best_parameters)
#
#     return (best_score, best_parameters)
#
# rf_parameters = {'n_estimators':[250,500,750],'max_depth':[5,10,15,20,25,30,35,40,45,50],'max_features':[35]}
#
# rf_best_accuracy, rf_best_parameters = grid_search(RandomForestRegressor(),
#                                                    rf_parameters,
#                                                    5,
#                                                    'r2',
#                                                    X,
#                                                    y)
#
# gb_parameters = {'n_estimators':[250, 500, 750, 1000, 1500], 'learning_rate':[0.005, 0.125, 0.25], 'max_depth':[3, 5,10,15],
#                  'max_features':[35], 'subsample':[0.3, 0.4, 0.5, 0.6, 0.7]}
#
# gb_best_accuracy, gb_best_parameters = grid_search(GradientBoostingRegressor(loss='ls'),
#                                                    gb_parameters,
#                                                    5,
#                                                    'r2',
#                                                    X,
#                                                    y)
#
# svm_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
#                      'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
#                     {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
#                      'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
#                     {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}]
#
# svm_best_accuracy, svm_best_parameters = grid_search(SVR(),
#                                                    svm_parameters,
#                                                    5,
#                                                    'r2',
#                                                    X_norm,
#                                                    y)
#
# mlp_parameters = {'hidden_layer_sizes':[(64,64,64),(128,128,128),(256,256,256),(512,512,512)],
#                   'alpha':[0.5,0.3,0.2,0.1,0.05],
#                   'batch_size':[4,8,16,32,64]}
#
# mlp_best_accuracy, mlp_best_parameters = grid_search(MLPRegressor(),
#                                                    mlp_parameters,
#                                                    5,
#                                                    'r2',
#                                                    X_norm,
#                                                    y)
#
#
# cv = KFold(n_splits=5, random_state=612, shuffle=False)
# linear_reg_scores = []
# mlp_scores = []
# random_forest_scores = []
# gradient_boost_scores = []
#
# for train_index, test_index in cv.split(X_norm):
#     X_train, X_test, y_train, y_test = X_norm.iloc[train_index,:], X_norm.iloc[test_index,:], y.iloc[train_index], y.iloc[test_index]
#
#     linear_reg_model = LinearRegression()
#     linear_reg_model.fit(X_train, y_train)
#     linear_reg_scores.append(linear_reg_model.score(X_test, y_test))
#
#     mlp_model = MLPRegressor(alpha=mlp_best_parameters['alpha'],
#                              batch_size=mlp_best_parameters['batch_size'],
#                              hidden_layer_sizes=mlp_best_parameters['hidden_layer_sizes'])
#     mlp_model.fit(X_train, y_train)
#     mlp_scores.append(mlp_model.score(X_test, y_test))
#
#     random_forest_model = RandomForestRegressor(n_estimators=rf_best_parameters['n_estimators'],
#                                                 max_depth=rf_best_parameters['max_depth'],
#                                                 max_features=rf_best_parameters['max_features'])
#     random_forest_model.fit(X_train, y_train)
#     random_forest_scores.append(random_forest_model.score(X_test, y_test))
#
#     gradient_boost_model = GradientBoostingRegressor(n_estimators=gb_best_parameters['n_estimators'],
#                                                      max_depth=gb_best_parameters['max_depth'],
#                                                      max_features=gb_best_parameters['max_features'],
#                                                      learning_rate=gb_best_parameters['learning_rate'],
#                                                      subsample=gb_best_parameters['subsample'])
#     gradient_boost_model.fit(X_train, y_train)
#     gradient_boost_scores.append(gradient_boost_model.score(X_test, y_test))
#
# model_comparison = pd.DataFrame({'rf':random_forest_scores,
#                                  'gb':gradient_boost_scores,
#                                  'mlp':mlp_scores,
#                                  'lr':linear_reg_scores})
# model_comparison.describe()

gb_best_parameters = {'learning_rate': 0.125, 'max_depth': 3, 'max_features': 35, 'n_estimators': 1000, 'subsample': 0.4}

click_model = GradientBoostingRegressor(n_estimators=gb_best_parameters['n_estimators'],
                                           max_depth=gb_best_parameters['max_depth'],
                                           max_features=gb_best_parameters['max_features'],
                                           learning_rate=gb_best_parameters['learning_rate'],
                                           subsample=gb_best_parameters['subsample'])

click_model.fit(X_norm, y)

# click_model = sm.OLS(endog=X_norm['User Clicks'],
#                      exog=X_norm[model_var])).fit()
# click_model.summary()

ad_banner_cost = {
    'ad0': 15,
    'ad1': 10,
    'ad2': 8,
    'ad3': 8,
    'ad4': 12
}

ads = ['ad0', 'ad1', 'ad2', 'ad3', 'ad4']

def calculate_costs(data):
    costs = 0
    for ad in ads:
        costs += data[ad + '_time_spent'] * ad_banner_cost[ad]

    return costs


def generate_web_banners():
    # Produce an array of chosen banners for each of the five websites
    show_noshow = np.random.choice(np.arange(0, 2), 5, replace=True)
    number_of_adverts = np.count_nonzero(show_noshow)
    banners = np.random.choice(np.arange(1, 7), number_of_adverts, replace=False)

    nonzero_index = np.nonzero(show_noshow)[0]

    webbanners = show_noshow.copy()
    for i in range(len(nonzero_index)):
        webbanners[nonzero_index[i]] = banners[i]

    webbanners_dict = {
        'ad0': webbanners[0],
        'ad1': webbanners[1],
        'ad2': webbanners[2],
        'ad3': webbanners[3],
        'ad4': webbanners[4]
    }

    return webbanners_dict


def generate_plan(webbanners_dict):
    plan = pd.DataFrame({'ad0': [0], 'ad1': [0], 'ad2': [0], 'ad3': [0], 'ad4': [0],
                         'ad0_start_time': [0], 'ad1_start_time': [0], 'ad2_start_time': [0], 'ad3_start_time': [0],
                         'ad4_start_time': [0],
                         'ad0_time_spent': [0], 'ad1_time_spent': [0], 'ad2_time_spent': [0], 'ad3_time_spent': [0],
                         'ad4_time_spent': [0]})

    for ad in ads:
        plan[ad] = webbanners_dict[ad]
        if webbanners_dict[ad] != 0:
            plan[ad] = webbanners_dict[ad]
            plan[ad + '_start_time'] = float(decimal.Decimal(random.randrange(0, 241)) / 10)
            plan[ad + '_time_spent'] = float(decimal.Decimal(random.randrange(0, 241)) / 10)
            plan[ad + '_time_spent'] = np.where(plan[ad + '_start_time'] + plan[ad + '_time_spent'] > 24,
                                                24 - plan[ad + '_start_time'],
                                                plan[ad + '_time_spent'])
            if plan[ad + '_time_spent'].values == 0:
                plan[ad] = 0
                plan[ad + '_start_time'] = 0

    return plan


def generate_marketing_plans(number_of_plans):
    marketing_plans = pd.DataFrame()

    counter = 1
    while counter <= number_of_plans:
        webbanners_dict = generate_web_banners()
        plan = generate_plan(webbanners_dict)
        if (calculate_costs(plan).values >= 100) & (calculate_costs(plan).values <= 300):
            marketing_plans = marketing_plans.append(plan)
            counter += 1

    return marketing_plans


def predict_clicks(data_to_predict):
    data = data_to_predict.copy()

    ad0_data = pd.get_dummies(pd.concat([data[['ad0']], pd.DataFrame({'ad0': [1, 2, 3, 4, 5, 6]})])['ad0'],
                              prefix='ad0').iloc[:-6, :]
    ad1_data = pd.get_dummies(pd.concat([data[['ad1']], pd.DataFrame({'ad1': [1, 2, 3, 4, 5, 6]})])['ad1'],
                              prefix='ad1').iloc[:-6, :]
    ad2_data = pd.get_dummies(pd.concat([data[['ad2']], pd.DataFrame({'ad2': [1, 2, 3, 4, 5, 6]})])['ad2'],
                              prefix='ad2').iloc[:-6, :]
    ad3_data = pd.get_dummies(pd.concat([data[['ad3']], pd.DataFrame({'ad3': [1, 2, 3, 4, 5, 6]})])['ad3'],
                              prefix='ad3').iloc[:-6, :]
    ad4_data = pd.get_dummies(pd.concat([data[['ad4']], pd.DataFrame({'ad4': [1, 2, 3, 4, 5, 6]})])['ad4'],
                              prefix='ad4').iloc[:-6, :]

    data = pd.concat([ad0_data, ad1_data, ad2_data, ad3_data, ad4_data,
                      data[['ad0_start_time', 'ad1_start_time', 'ad2_start_time', 'ad3_start_time', 'ad4_start_time',
                            'ad0_time_spent', 'ad1_time_spent', 'ad2_time_spent', 'ad3_time_spent', 'ad4_time_spent']]],
                     axis=1)

    # user_clicks = click_model.predict(sm.add_constant(data[model_var], has_constant='add'))
    user_clicks = click_model.predict(normalize_model_data(data[model_var]))

    return user_clicks


def selectOne(population):
    max = sum(population)
    selection_probs = [c / max for c in population]
    return np.random.choice(len(population), p=selection_probs)


def roulette_wheel_selection(marketing_plans, number_to_keep):
    user_clicks_array = marketing_plans['predicted_clicks'].values

    keep_index = []
    for i in range(number_to_keep):
        keep_index.append(selectOne(user_clicks_array))

    return marketing_plans.iloc[keep_index, :]


def generate_cross_over_child(parent1, parent2):
    new_child_table = pd.DataFrame()
    for i in np.arange(0, parent1.shape[0]):
        parent_choice = np.random.choice([1, 2], 1)[0]
        if parent_choice == 1:
            new_child_table = new_child_table.append(parent1.iloc[i, :])
        else:
            new_child_table = new_child_table.append(parent2.iloc[i, :])

    return new_child_table


def crossover(parents, number_of_childs):
    parent1_list = list(np.random.choice(np.arange(0, parents.shape[0]), number_of_childs, replace=True))
    parent2_list = list(np.random.choice(np.arange(0, parents.shape[0]), number_of_childs, replace=True))

    new_child_df = pd.DataFrame()
    for child in range(0, number_of_childs):

        parent1_adnumber = pd.DataFrame({'adnumber': ['ad0', 'ad1', 'ad2', 'ad3', 'ad4']})
        parent1_banner = parents[['ad0', 'ad1', 'ad2', 'ad3', 'ad4']].iloc[parent1_list[child], :].reset_index(
            drop=True)
        parent1_start_time = parents[['ad0_start_time', 'ad1_start_time', 'ad2_start_time', 'ad3_start_time',
                                      'ad4_start_time']].iloc[parent1_list[child], :].reset_index(drop=True)
        parent1_time_spent = parents[['ad0_time_spent', 'ad1_time_spent', 'ad2_time_spent', 'ad3_time_spent',
                                      'ad4_time_spent']].iloc[parent1_list[child], :].reset_index(drop=True)
        parent1_values = pd.concat([parent1_adnumber, parent1_banner, parent1_start_time, parent1_time_spent], axis=1)
        parent1_values.columns = ['adnumber', 'banner', 'start_time', 'time_spent']

        parent2_adnumber = pd.DataFrame({'adnumber': ['ad0', 'ad1', 'ad2', 'ad3', 'ad4']})
        parent2_banner = parents[['ad0', 'ad1', 'ad2', 'ad3', 'ad4']].iloc[parent2_list[child], :].reset_index(
            drop=True)
        parent2_start_time = parents[['ad0_start_time', 'ad1_start_time', 'ad2_start_time', 'ad3_start_time',
                                      'ad4_start_time']].iloc[parent2_list[child], :].reset_index(drop=True)
        parent2_time_spent = parents[['ad0_time_spent', 'ad1_time_spent', 'ad2_time_spent', 'ad3_time_spent',
                                      'ad4_time_spent']].iloc[parent2_list[child], :].reset_index(drop=True)
        parent2_values = pd.concat([parent2_adnumber, parent2_banner, parent2_start_time, parent2_time_spent], axis=1)
        parent2_values.columns = ['adnumber', 'banner', 'start_time', 'time_spent']

        new_child = generate_cross_over_child(parent1_values, parent2_values)
        new_child['costs'] = new_child['adnumber'].apply(lambda x: ad_banner_cost[x]) * new_child['time_spent']

        while (sum(new_child[new_child['banner'] != 0]['banner'].value_counts() > 1) > 0) | (
                new_child['costs'].sum() > 300):
            new_child = generate_cross_over_child(parent1_values, parent2_values)
            new_child['costs'] = new_child['adnumber'].apply(lambda x: ad_banner_cost[x]) * new_child['time_spent']

        new_child_record = pd.DataFrame(
            [[new_child['banner'].values[0], new_child['banner'].values[1], new_child['banner'].values[2],
              new_child['banner'].values[3], new_child['banner'].values[4],
              new_child['start_time'].values[0], new_child['start_time'].values[1], new_child['start_time'].values[2],
              new_child['start_time'].values[3], new_child['start_time'].values[4],
              new_child['time_spent'].values[0], new_child['time_spent'].values[1], new_child['time_spent'].values[2],
              new_child['time_spent'].values[3], new_child['time_spent'].values[4],
              new_child['costs'].sum()]])
        new_child_record.columns = ['ad0', 'ad1', 'ad2', 'ad3', 'ad4',
                                    'ad0_start_time', 'ad1_start_time', 'ad2_start_time', 'ad3_start_time',
                                    'ad4_start_time',
                                    'ad0_time_spent', 'ad1_time_spent', 'ad2_time_spent', 'ad3_time_spent',
                                    'ad4_time_spent',
                                    'cost']
        new_child_df = new_child_df.append(new_child_record, ignore_index=True)

    for var in ['ad0', 'ad1', 'ad2', 'ad3', 'ad4']:
        new_child_df[var] = new_child_df[var].astype('int')

    new_child_df['predicted_clicks'] = predict_clicks(new_child_df)

    return new_child_df


def mutate_scramble(marketing_plans, mutation_scramble_rate):
    scrambled_marketing_plans = pd.DataFrame()
    i = 0
    while i < marketing_plans.shape[0]:
        plan = marketing_plans.iloc[[i]]

        if random.randrange(0, 100) / 100 <= mutation_scramble_rate:
            shuffle_index = list(np.arange(0, 5))
            random.shuffle(shuffle_index)

            if random.choice([0,1,2]) == 0:
                plan_ad_banners = plan[['ad0', 'ad1', 'ad2', 'ad3', 'ad4']].copy()
                shuffled_plan_ad_banners = plan_ad_banners.iloc[:, shuffle_index].copy()
                shuffled_plan_ad_banners.columns = plan_ad_banners.columns

                shuffled_plan_start_time = plan[
                    ['ad0_start_time', 'ad1_start_time', 'ad2_start_time', 'ad3_start_time', 'ad4_start_time']].copy()

                shuffled_plan_time_spent = plan[
                    ['ad0_time_spent', 'ad1_time_spent', 'ad2_time_spent', 'ad3_time_spent', 'ad4_time_spent']].copy()

                # plan_record = pd.concat([shuffled_plan_ad_banners, shuffled_plan_start_time, shuffled_plan_time_spent],
                #                         axis=1)
                # plan_record['cost'] = calculate_costs(plan_record)

            if random.choice([0, 1, 2]) == 1:
                shuffled_plan_ad_banners = plan[['ad0', 'ad1', 'ad2', 'ad3', 'ad4']].copy()

                plan_start_time = plan[
                    ['ad0_start_time', 'ad1_start_time', 'ad2_start_time', 'ad3_start_time', 'ad4_start_time']].copy()
                shuffled_plan_start_time = plan_start_time.iloc[:, shuffle_index].copy()
                shuffled_plan_start_time.columns = plan_start_time.columns

                shuffled_plan_time_spent = plan[
                    ['ad0_time_spent', 'ad1_time_spent', 'ad2_time_spent', 'ad3_time_spent', 'ad4_time_spent']].copy()

                # plan_record = pd.concat([shuffled_plan_ad_banners, shuffled_plan_start_time, shuffled_plan_time_spent],
                #                         axis=1)
                # plan_record['cost'] = calculate_costs(plan_record)

            if random.choice([0, 1, 2]) == 2:
                shuffled_plan_ad_banners = plan[['ad0', 'ad1', 'ad2', 'ad3', 'ad4']].copy()

                shuffled_plan_start_time = plan[
                    ['ad0_start_time', 'ad1_start_time', 'ad2_start_time', 'ad3_start_time', 'ad4_start_time']].copy()

                plan_time_spent = plan[
                    ['ad0_time_spent', 'ad1_time_spent', 'ad2_time_spent', 'ad3_time_spent', 'ad4_time_spent']].copy()
                shuffled_plan_time_spent = plan_time_spent.iloc[:, shuffle_index].copy()
                shuffled_plan_time_spent.columns = plan_time_spent.columns

                # plan_record = pd.concat([shuffled_plan_ad_banners, shuffled_plan_start_time, shuffled_plan_time_spent],
                #                         axis=1)
                # plan_record['cost'] = calculate_costs(plan_record)

            plan_record = pd.concat([shuffled_plan_ad_banners, shuffled_plan_start_time, shuffled_plan_time_spent],
                                    axis=1)
            plan_record['cost'] = calculate_costs(plan_record)

        else:
            plan_ad_banners = plan[['ad0', 'ad1', 'ad2', 'ad3', 'ad4']].copy()
            plan_start_time = plan[
                ['ad0_start_time', 'ad1_start_time', 'ad2_start_time', 'ad3_start_time', 'ad4_start_time']].copy()
            plan_time_spent = plan[
                ['ad0_time_spent', 'ad1_time_spent', 'ad2_time_spent', 'ad3_time_spent', 'ad4_time_spent']].copy()

            plan_record = pd.concat([plan_ad_banners, plan_start_time, plan_time_spent], axis=1)
            plan_record['cost'] = calculate_costs(plan_record)

        if plan_record['cost'].values[0] <= 300:
            scrambled_marketing_plans = scrambled_marketing_plans.append(plan_record, ignore_index=True)
            i += 1

    scrambled_marketing_plans['predicted_clicks'] = predict_clicks(scrambled_marketing_plans)

    return scrambled_marketing_plans


def mutate_gaussian(marketing_plans, mutation_gaussian_rate):
    marketing_plans_gaussian = pd.DataFrame()
    for i in range(0, marketing_plans.shape[0]):
        marketing_plan = marketing_plans.iloc[i:i + 1, :].copy()
        if (random.randrange(0, 101) / 100) <= mutation_gaussian_rate:
            ad_to_mutate = ads[random.randrange(0, 5)]
            if marketing_plan[ad_to_mutate].values[0] == 0:
                running_banners = list(set(marketing_plan[['ad0', 'ad1', 'ad2', 'ad3', 'ad4']].values[0]))
                banners = list(np.arange(0, 6))
                marketing_plan[ad_to_mutate] = random.choice(
                    [banner for banner in banners if banner not in running_banners])
                marketing_plan[ad_to_mutate + '_start_time'] = float(decimal.Decimal(random.randrange(0, 241)) / 10)
                marketing_plan[ad_to_mutate + '_time_spent'] = float(decimal.Decimal(random.randrange(0, 241)) / 10)
                marketing_plan['cost'] = calculate_costs(marketing_plan)
                while marketing_plan['cost'].values[0] > 300:
                    marketing_plan[ad_to_mutate + '_start_time'] = float(decimal.Decimal(random.randrange(0, 241)) / 10)
                    marketing_plan[ad_to_mutate + '_time_spent'] = float(decimal.Decimal(random.randrange(0, 241)) / 10)
                    marketing_plan['cost'] = calculate_costs(marketing_plan)
            else:
                marketing_plan[ad_to_mutate + '_start_time'] = float(decimal.Decimal(random.randrange(0, 241)) / 10)
                marketing_plan[ad_to_mutate + '_time_spent'] = float(decimal.Decimal(random.randrange(0, 241)) / 10)
                marketing_plan['cost'] = calculate_costs(marketing_plan)
                if marketing_plan[ad_to_mutate + '_time_spent'].values == 0:
                    marketing_plan[ad_to_mutate] = 0
                    marketing_plan[ad_to_mutate + '_start_time'] = 0
                while marketing_plan['cost'].values[0] > 300:
                    marketing_plan[ad_to_mutate + '_start_time'] = float(decimal.Decimal(random.randrange(0, 241)) / 10)
                    marketing_plan[ad_to_mutate + '_time_spent'] = float(decimal.Decimal(random.randrange(0, 241)) / 10)
                    marketing_plan['cost'] = calculate_costs(marketing_plan)
                    if marketing_plan[ad_to_mutate + '_time_spent'].values == 0:
                        marketing_plan[ad_to_mutate] = 0
                        marketing_plan[ad_to_mutate + '_start_time'] = 0

            marketing_plan['predicted_clicks'] = predict_clicks(marketing_plan)
            marketing_plans_gaussian = marketing_plans_gaussian.append(marketing_plan, ignore_index=True)
        else:
            marketing_plans_gaussian = marketing_plans_gaussian.append(marketing_plan, ignore_index=True)

    return marketing_plans_gaussian
