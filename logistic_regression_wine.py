import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hashlib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import time

from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
import warnings
from sklearn.metrics import confusion_matrix
import seaborn as sns
from imblearn.over_sampling import SMOTE

# ignore warnings
warnings.filterwarnings("ignore")


def choose_fixed_indices(df):
    # Use the hash of the DataFrame's contents as the seed value
    hash_value = hashlib.md5(df.to_numpy().tobytes()).hexdigest()
    seed_value = int(hash_value, 16) % 2 ** 32

    # Use the fixed seed value to select a subset of row indices from the DataFrame
    rng = np.random.default_rng(seed_value)
    indices = rng.choice(df.index, size=5, replace=False)

    # Create a new DataFrame with the selected rows removed
    selected_rows = df.loc[indices]
    new_df = df.drop(indices)
    return selected_rows, new_df


def check_correlation(df, attribute):
    corr_matrix = df.corr()
    return corr_matrix[attribute]


def plot_regularize(df, cls_features, cls_labels, savefig=False):
    fig = plt.figure()
    ax = plt.subplot(111)

    colors = ['blue', 'green', 'red', 'cyan',
              'magenta', 'yellow', 'black',
              'pink', 'lightgreen', 'lightblue',
              'gray', 'orange']

    weights, params = [], []
    for c in np.arange(-4., 6.):
        lr = LogisticRegression(penalty='l1', C=10. ** c, solver='liblinear',
                                multi_class='ovr', random_state=42)
        lr.fit(cls_features, cls_labels)
        weights.append(lr.coef_[1])
        params.append(10 ** c)

    weights = np.array(weights)

    for column, color in zip(range(weights.shape[1]), colors):
        plt.plot(params, weights[:, column],
                 label=df.columns[column],
                 color=color)
    plt.axhline(0, color='black', linestyle='--', linewidth=3)
    plt.xlim([10 ** (-5), 10 ** 5])
    plt.ylabel('Weight coefficient')
    plt.xlabel('C (inverse regularization strength)')
    plt.xscale('log')
    plt.legend(loc='upper left')
    if savefig:
        plt.savefig('regularisation.png')
    plt.show()
    return


def grid_Search_new(x_data, y_data, estimator, param_grid, penalty_value):
    grid = GridSearchCV(estimator, param_grid, cv=5, n_jobs=-1)
    grid.fit(x_data, y_data)
    print(penalty_value, "-->", grid.best_params_)
    # print(grid.best_params_)
    return grid.best_params_


class WineDataAnalyzer:
    def __init__(self, dataframe):
        self.df = dataframe

    def plot_hist(self, column, savefig=False, filename=None):
        quality_counts = self.df['quality'].value_counts().sort_index()
        quality_counts.plot(kind='bar')
        plt.xlabel('Sensory preference')
        plt.ylabel('Frequency(Wine samples)')
        plt.grid(axis='y')
        if savefig:
            if not filename:
                filename = f'{column}_hist.png'
            plt.savefig(filename)
        plt.show()

    def plot_correlation_heatmap(self, savefig=False, filename=None):
        corr = self.df.corr()
        plt.subplots(figsize=(15, 10))
        sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True,
                    cmap=sns.diverging_palette(220, 20, as_cmap=True))
        if savefig:
            if not filename:
                filename = 'corr.png'
            plt.savefig(filename)
        plt.show()

    def analyse_data_structure(self, bins=50, savefig=False, filename=None):
        self.df.hist(bins=bins, figsize=(20, 15))  # plot histogram for attributes
        if savefig:
            if not filename:
                filename = 'histogram_features.png'
            plt.savefig(filename)
        plt.show()


class ModelFitter:
    def __init__(self, clf):
        self.clf = clf
        self.train_predict = None
        self.test_predict = None

    def fit(self, x_data_train, y_data_train, x_data_test, y_data_test, penalty):
        start_time = time.time()
        self.clf.fit(x_data_train, y_data_train)
        end_time = time.time()
        self.train_predict, self.test_predict = self.clf.predict(x_data_train), self.clf.predict(x_data_test)
        train_score, sparsity, test_score = round(self.clf.score(x_data_train, y_data_train), 4), round(np.mean(
                self.clf.coef_ == 0) * 100, 4), round(self.clf.score(x_data_test, y_data_test), 4)
        train_precision, test_precision = round(precision_score(y_data_train, self.train_predict, average='weighted'),
                                                4), \
                                          round(precision_score(y_data_test, self.test_predict, average='weighted'), 4)
        exec_time = round(end_time - start_time, 4)
        # Calculate the confusion matrix
        conf_train = confusion_matrix(y_data_train, self.train_predict)
        conf_test = confusion_matrix(y_data_test, self.test_predict)
        return train_score, test_score, sparsity, train_precision, test_precision, exec_time, conf_train, conf_test

    def predict(self, x_data, y_data):
        # Use the trained model to make predictions on new data
        print('Original labels', np.array(y_data))
        print('Predicted labels', self.clf.predict(x_data))

    def print_results(self, train_score, test_score, sparsity, train_precision, test_precision, exec_time, penalty):
        print("Best {} norm stats,".format(penalty))
        print("Train Accuracy:", train_score, "Test Accuracy:", test_score, "Sparsity:", sparsity, "Train Precision:",
              train_precision, "Test Precision:", test_precision, "Execution Time:", exec_time)


class OrderedModelPredictor:
    def __init__(self, X_train_stat, y_train_stat, X_test_stat, y_test_stat):
        self.X_train_stat = X_train_stat
        self.y_train_stat = y_train_stat
        self.X_test_stat = X_test_stat
        self.y_test_stat = y_test_stat

    def fit(self):
        self.model = OrderedModel(self.y_train_stat, self.X_train_stat, distr='logit')
        self.model.unique_y = np.unique(self.y_train_stat)
        self.result = self.model.fit(method='bfgs', disp=False)

    def predict_labels(self):
        self.fit()
        self.train_stat_preds = self.result.predict(self.X_train_stat)
        self.test_stat_pred = self.result.predict(self.X_test_stat)
        self.y_train_pred_labels = np.argmax(self.train_stat_preds, axis=1)
        self.y_test_pred_labels = np.argmax(self.test_stat_pred, axis=1)

    def predict(self, X_new, y_new):
        self.fit()
        preds = self.result.predict(X_new)
        pred_labels = np.argmax(preds, axis=1)
        pred_real_labels = np.array([self.model.unique_y[i] for i in pred_labels])
        pred_accuracy = accuracy_score(y_new, pred_real_labels)
        pred_precision = precision_score(y_new, pred_real_labels, average="weighted")
        print('Original labels', np.array(y_new))
        print('Predicted labels',pred_real_labels)
        return pred_accuracy, pred_precision

    def print_results(self):
        self.predict_labels()
        train_pred_labels = np.array([self.model.unique_y[i] for i in self.y_train_pred_labels])
        # print(train_pred_labels)
        test_pred_labels = np.array([self.model.unique_y[i] for i in self.y_test_pred_labels])
        # print(test_pred_labels)
        conf_train = confusion_matrix(self.y_train_stat, train_pred_labels)
        conf_test = confusion_matrix(self.y_test_stat, test_pred_labels)
        train_accu_stat = accuracy_score(self.y_train_stat, train_pred_labels)
        test_accu_stat = accuracy_score(self.y_test_stat, test_pred_labels)
        train_precision_matrix = precision_score(self.y_train_stat, train_pred_labels, average="weighted")
        test_precision_matrix = precision_score(self.y_test_stat, test_pred_labels, average="weighted")
        print('Training accuracy:', round(train_accu_stat * 100, 2), 'Testing accuracy:',
              round(test_accu_stat * 100, 2),
              'Training precision:', round(train_precision_matrix, 4), 'Testing precision:',
              round(test_precision_matrix, 4))
        return conf_train, conf_test


if __name__ == '__main__':
    data = pd.read_csv("winequality-red.csv", delimiter=';')  # load wine dataset
    analyser = WineDataAnalyzer(data)
    analyser.plot_hist('quality') #plot the histogram for quality
    analyser.plot_correlation_heatmap() #plot the heatmap of the correlation
    analyser.analyse_data_structure() #plot hist for numerical attribute
    model_test_data, train_data = choose_fixed_indices(data)  # dataset for model check
    X_mod_check = model_test_data.drop('quality', axis=1)  # Independent variables
    y_mod_check = model_test_data['quality']  # dependent variable
    X = train_data.drop('quality', axis=1)
    y = train_data['quality']
    pd.set_option('display.max_columns', None)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,
                                                        random_state=42)  # Split the dataset for training and testing
    y_train_stat = y_train.copy()  # target variables for ordinal model
    y_test_stat = y_test.copy()

    # Normalize the features using StandardScaler
    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)  # for logistic regression
    X_test = scalar.transform(X_test)
    X_train_stat = X_train.copy()  # for ordinal model
    X_test_stat = X_test.copy()
    X_mod_check = scalar.transform(X_mod_check)  # Check ordinal/logistic regression model

    # =======================Logistic regression=============
    print('========================Logistic Regression(No feature elimination)========================================')
    estimator = LogisticRegression(max_iter=1000, class_weight='balanced', tol=0.00001, random_state=42)
    # without norm
    param_no_norm = [{'C': np.logspace(-5, 0, 10)}]
    no_norm_best_param = grid_Search_new(X_train, y_train, estimator, param_no_norm, 'No Norm')
    clf_no_norm = LogisticRegression(max_iter=1000, class_weight='balanced', tol=0.00001, random_state=42,
                                     **no_norm_best_param)
    model_fitter = ModelFitter(clf_no_norm)
    train_score, test_score, sparsity, train_precision, test_precision, exec_time, conf_train, conf_test_log = \
        model_fitter.fit(X_train, y_train, X_test, y_test, 'no_norm')
    model_fitter.print_results(train_score, test_score, sparsity, train_precision, test_precision, exec_time, 'no_norm')

    # l1 norm
    param_l1 = [{'C': np.logspace(-5, 0, 10), 'penalty': ['l1'], 'solver': ['liblinear'],
                 'multi_class': ['ovr', 'Multinomial']}]
    l1_norm_best_param = grid_Search_new(X_train, y_train, estimator, param_l1, 'l1 Norm')
    clf_l1_norm = LogisticRegression(max_iter=1000, class_weight='balanced', tol=0.00001, random_state=42,
                                     **l1_norm_best_param)
    model_fitter = ModelFitter(clf_l1_norm)
    train_score, test_score, sparsity, train_precision, test_precision, exec_time, conf_train, conf_test = \
        model_fitter.fit(X_train, y_train, X_test, y_test, 'l1')
    model_fitter.print_results(train_score, test_score, sparsity, train_precision, test_precision, exec_time, 'l1')
    plot_regularize(data, X_train, y_train)
    best_logistic_conf = conf_test
    best_logistic_model_fitter = model_fitter

    # l2 norm
    param_l2 = [{'C': np.logspace(-5, 0, 10), 'penalty': ['l2'], 'solver': ['lbfgs', 'liblinear'],
                 'multi_class': ['ovr', 'Multinomial']}]
    l2_norm_best_param = grid_Search_new(X_train, y_train, estimator, param_l2, 'l2 Norm')
    clf_l2_norm = LogisticRegression(max_iter=1000, class_weight='balanced', tol=0.00001, random_state=42,
                                     **l2_norm_best_param)
    model_fitter = ModelFitter(clf_l2_norm)
    train_score, test_score, sparsity, train_precision, test_precision, exec_time, conf_train, conf_test = \
        model_fitter.fit(X_train, y_train, X_test, y_test, 'l2')
    model_fitter.print_results(train_score, test_score, sparsity, train_precision, test_precision, exec_time, 'l2')

    # Elastic norm
    estimator = LogisticRegression(max_iter=1000, random_state=42)
    para_elastic = [{'C': np.logspace(-5, 0, 10), 'penalty': ['elasticnet'], 'solver': ['saga'],
                     'l1_ratio': np.linspace(0, 1, 10)}]
    elastic_norm_best_param = grid_Search_new(X_train, y_train, estimator, para_elastic, 'Elastic Norm')
    clf_elastic_norm = LogisticRegression(max_iter=1000, class_weight='balanced', tol=0.00001, random_state=42,
                                          **elastic_norm_best_param)
    model_fitter = ModelFitter(clf_elastic_norm)
    train_score, test_score, sparsity, train_precision, test_precision, exec_time, conf_train, conf_test = \
        model_fitter.fit(X_train, y_train, X_test, y_test, 'elastic_net')
    model_fitter.print_results(train_score, test_score, sparsity, train_precision, test_precision, exec_time, 'elastic')

    print('==========Ordinal regression========')
    smote = SMOTE()
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_stat, y_train_stat)
    ordinal_regression = OrderedModelPredictor(X_train_resampled, y_train_resampled, X_test_stat, y_test_stat)
    conf_train_stat, conf_test_stat = ordinal_regression.print_results()
    best_conf_test_stat = conf_test_stat
    best_ordinal_regression = ordinal_regression

    print('=================================Eliminated features==============================')
    X_train_check_eliminate = np.delete(X_train, [9], axis=1)  # for eliminate features logistic
    X_test_check_eliminate = np.delete(X_test, [9], axis=1)

    X_train_stat_eliminate = np.delete(X_train_stat, [9], axis=1)  # for eliminate features orderedmodel
    X_test_stat_eliminate = np.delete(X_test_stat, [9], axis=1)
    X_mod_check_eliminated = np.delete(X_mod_check, [9], axis=1)

    estimator = LogisticRegression(max_iter=1000, class_weight='balanced', tol=0.00001, random_state=42)
    # without norm
    param_no_norm = [{'C': np.logspace(-5, 0, 10)}]
    no_norm_best_param = grid_Search_new(X_train_check_eliminate, y_train, estimator, param_no_norm, 'No Norm')
    clf_no_norm = LogisticRegression(max_iter=1000, class_weight='balanced', tol=0.00001, random_state=42,
                                     **no_norm_best_param)
    model_fitter = ModelFitter(clf_no_norm)
    train_score, test_score, sparsity, train_precision, test_precision, exec_time, conf_train, conf_test_log = \
        model_fitter.fit(X_train_check_eliminate, y_train, X_test_check_eliminate, y_test, 'no_norm')
    model_fitter.print_results(train_score, test_score, sparsity, train_precision, test_precision, exec_time, 'no_norm')
    #print(model_fitter.predict(X_mod_check_eliminated, y_mod_check))

    # l1 norm
    param_l1 = [{'C': np.logspace(-5, 0, 10), 'penalty': ['l1'], 'solver': ['liblinear'],
                 'multi_class': ['ovr', 'Multinomial']}]
    l1_norm_best_param = grid_Search_new(X_train_check_eliminate, y_train, estimator, param_l1, 'l1 Norm')
    clf_l1_norm = LogisticRegression(max_iter=1000, class_weight='balanced', tol=0.00001, random_state=42,
                                     **l1_norm_best_param)
    model_fitter = ModelFitter(clf_l1_norm)
    train_score, test_score, sparsity, train_precision, test_precision, exec_time, conf_train, conf_test = \
        model_fitter.fit(X_train_check_eliminate, y_train, X_test_check_eliminate, y_test, 'l1')
    model_fitter.print_results(train_score, test_score, sparsity, train_precision, test_precision, exec_time, 'l1')

    # l2 norm
    param_l2 = [{'C': np.logspace(-5, 0, 10), 'penalty': ['l2'], 'solver': ['lbfgs', 'liblinear'],
                 'multi_class': ['ovr', 'Multinomial']}]
    l2_norm_best_param = grid_Search_new(X_train_check_eliminate, y_train, estimator, param_l2, 'l2 Norm')
    clf_l2_norm = LogisticRegression(max_iter=1000, class_weight='balanced', tol=0.00001, random_state=42,
                                     **l2_norm_best_param)
    model_fitter = ModelFitter(clf_l2_norm)
    train_score, test_score, sparsity, train_precision, test_precision, exec_time, conf_train, conf_test = \
        model_fitter.fit(X_train_check_eliminate, y_train, X_test_check_eliminate, y_test, 'l2')
    model_fitter.print_results(train_score, test_score, sparsity, train_precision, test_precision, exec_time, 'l2')

    # Elastic norm
    estimator = LogisticRegression(max_iter=1000)
    para_elastic = [{'C': np.logspace(-5, 0, 10), 'penalty': ['elasticnet'], 'solver': ['saga'],
                     'l1_ratio': np.linspace(0, 1, 10)}]
    elastic_norm_best_param = grid_Search_new(X_train_check_eliminate, y_train, estimator, para_elastic, 'Elastic Norm')
    clf_elastic_norm = LogisticRegression(max_iter=1000, class_weight='balanced', tol=0.00001, random_state=42,
                                          **elastic_norm_best_param)
    model_fitter = ModelFitter(clf_elastic_norm)
    train_score, test_score, sparsity, train_precision, test_precision, exec_time, conf_train, conf_test = \
        model_fitter.fit(X_train_check_eliminate, y_train, X_test_check_eliminate, y_test, 'elastic_net')
    model_fitter.print_results(train_score, test_score, sparsity, train_precision, test_precision, exec_time, 'elastic')

    print('==========Ordinal regression========')
    smote = SMOTE()
    X_train_stat_eliminate_resampled, y_train_resampled = smote.fit_resample(X_train_stat_eliminate, y_train_stat)

    ordinal_regression = OrderedModelPredictor(X_train_stat_eliminate_resampled, y_train_resampled,
                                               X_test_stat_eliminate, y_test_stat)
    conf_train_stat, conf_test_stat = ordinal_regression.print_results()

    print('============================= Condusion matric for best logistic regression model=====')
    print(best_logistic_conf)
    print('============================= Confusion matric for best ordinal logistic regression model=====')
    print(best_conf_test_stat)

    print('========================== Best Logistic regression model with model check dataset========================')
    best_logistic_model_fitter.predict(X_mod_check, y_mod_check)
    print('==========================================================================================')

    print('========================== Best ordinal regression model with model check dataset========================')
    best_ordinal_regression.predict(X_mod_check, y_mod_check)
    print('==========================================================================================')
