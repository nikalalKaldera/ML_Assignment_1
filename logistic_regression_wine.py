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
from sklearn.metrics import precision_score, make_scorer
import warnings
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
    
# ignore warnings
warnings.filterwarnings("ignore")


def analyse_data_structure(dataframe):
    print(dataframe.describe())
    dataframe.hist(bins=50, figsize=(20, 15))  # plot histogram for attributes
    plt.savefig(fname="his.png")
    plt.show()
    return


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


def plot_regularize(df, cls_features, cls_labels):
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
                 label=df.columns[column + 1],
                 color=color)
    plt.axhline(0, color='black', linestyle='--', linewidth=3)
    plt.xlim([10 ** (-5), 10 ** 5])
    plt.ylabel('Weight coefficient')
    plt.xlabel('C (inverse regularization strength)')
    plt.xscale('log')
    plt.legend(loc='upper left')
    plt.show()
    return


def grid_Search(x_train, y_train):
    param_grid = [
        {'penalty': ['l1', 'l2', 'elasticnet'], 'C': np.logspace(-5, 0, 10), 'l1_ratio': np.linspace(0, 1, 10)}
    ]
    # Create a logistic regression model
    logreg = LogisticRegression(max_iter=1000, class_weight='balanced', tol=0.00001)
    # Create a precision scorer
    precision_scorer = make_scorer(precision_score, average='weighted')
    # Create a GridSearchCV object
    grid_search = GridSearchCV(logreg, param_grid, cv=5, n_jobs=-1, scoring=precision_scorer)

    # Fit the GridSearchCV object to the data
    grid_search.fit(x_train, y_train)
    print(grid_search.best_params_)
    return


def plot_hist(df):
    quality_counts = df['quality'].value_counts().sort_index()
    quality_counts.plot(kind='bar')
    plt.xlabel('Sensory preference')
    plt.ylabel('Frequency(Wine samples)')
    plt.grid(axis='y')
    plt.show()
    return


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
        if penalty == 'l2':
            train_score, sparsity, test_score = round(self.clf.score(x_data_train, y_data_train), 4), round(np.mean(
                np.abs(self.clf.coef_) < 0.015) * 100, 4), round(self.clf.score(x_data_test, y_data_test), 4)
        else:
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
        return precision_score(y_data, (self.clf.predict(x_data)), average='weighted')

    def plot_confusion_matrix(self, conf_matrix):
        labels = [3, 4, 5, 6, 7, 8]
        cm_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)
        # Set the axis labels and title
        plt.figure(figsize=(8, 6))
        sns.set(font_scale=1.4)  # Adjust font size
        sns.heatmap(cm_df, annot=True, cmap="YlGnBu", fmt='g', xticklabels=labels, yticklabels=labels)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        plt.show()

    def print_results(self, train_score, test_score, sparsity, train_precision, test_precision, exec_time, penalty):
        print("---------{} norm----------".format(penalty))
        print("Train Accuracy:", train_score, "Test Accuracy:", test_score, "Sparsity:", sparsity, "Train Precision:",
              train_precision, "Test Precision:", test_precision, "Execution Time:", exec_time)


class OrderedModelPredictor:
    def __init__(self, X_train_stat, y_train_stat, X_test_stat, y_test_stat):
        self.X_train_stat = X_train_stat
        self.y_train_stat = y_train_stat
        self.X_test_stat = X_test_stat
        self.y_test_stat = y_test_stat

    def fit(self):
        self.model = OrderedModel(self.y_train_stat, self.X_train_stat, distr='probit')
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
        print('Training accuracy:', round(train_accu_stat*100, 2), 'Testing accuracy:', round(test_accu_stat*100, 2),
              'Training precision:', round(train_precision_matrix, 4), 'Testing precision:',
              round(test_precision_matrix, 4))
        return conf_train, conf_test


if __name__ == '__main__':
    # load wine dataset
    data = pd.read_csv("winequality-red.csv", delimiter=';')
    # plot_hist(data)
    # print(y.tail())
    # analyse_data_structure(data)
    # data["toatal_acidity"] = data["fixed acidity"] + data["volatile acidity"] + data["citric acid"]
    # data["acidity_sulpher_ratio"] = data["toatal_acidity"] / data["sulphates"]
    # data["acidity_sulpher_ratio"] = (data["fixed acidity"] + data["volatile acidity"] + data["citric acid"]) / data["sulphates"]
    # data["chloro_suplher_ratio"] = data["chlorides"] / data["sulphates"]
    # data = data.drop('citric acid', axis=1)
    # data = data.drop('density', axis=1)
    # cor2 = check_correlation(data, "quality")
    # print(cor2)
    # print(data.columns)
    model_test_data, train_data = choose_fixed_indices(data)
    # print(train_data.head())
    # pd.set_option('display.max_columns', None)
    # print(model_test_data.head())
    # print("Length of train data :", len(train_d59.92 Testing accuracy: 54.86 ata))
    # print("Length of model testing data", len(model_test_data))
    # train_data_stat = train_data.copy()
    X_mod_check = model_test_data.drop('quality', axis=1)
    y_mod_check = model_test_data['quality']
    X = train_data.drop('quality', axis=1)
    y = train_data['quality']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # # copy for ordinal regression model training
    # X_train_stat = X_train.copy()
    # X_test_stat = X_test.copy()
    y_train_stat = y_train.copy()
    y_test_stat = y_test.copy()

    # Normalize the features using StandardScaler
    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.transform(X_test)   #for logistic regression

    X_train_stat = X_train.copy()# for orderedmodel
    X_test_stat = X_test.copy()

    # Turn up tolerance for faster convergence
    clf_l1 = LogisticRegression(penalty='l1', C=0.04, solver='saga', multi_class='multinomial', tol=0.00001,
                                class_weight='balanced',
                                random_state=42)  # 0.6031 71.2121 0.5705
    # clf_l1 = LogisticRegression(penalty = 'l1', C = 1.0, solver = 'liblinear', multi_class='ovr', random_state=42) # 0.6031 7.5758 0.5768
    # clf_l1 = LogisticRegression(penalty='l1', C=0.05, solver='liblinear', multi_class='ovr', random_state=42) # 0.5835 69.697 0.558
    # clf_l1 = LogisticRegression(penalty='l1', C=.08, solver='liblinear', random_state=42) #0.582 63.6364 0.5611
    # clf_l1 = LogisticRegression(penalty='l1', C=.08, solver='liblinear', random_state=42) # 0.582 63.6364 0.5611
    # clf_l1 = LogisticRegression(penalty='l1', C=0.01, solver='saga', multi_class='multinomial', class_weight='balanced') #0.2996 86.3636 0.2853
    # clf_l1 = LogisticRegression(penalty='l1', C=0.09, solver='liblinear', multi_class='ovr', verbose=1, tol=0.00001, random_state=42) #0.5804 62.1212 0.5643

    # l2 multinomial
    # clf_l2 = LogisticRegression(penalty='l2', C=0.001, solver='lbfgs', multi_class='multinomial', verbose=1, random_state=42) #0.5702 57.5758 0.5643
    # clf_l2 = LogisticRegression(penalty='l2', C=0.001, solver='newton-cg', multi_class='multinomial', verbose=1, random_state=42) #0.5702 57.5758 0.5643
    # clf_l2 = LogisticRegression(penalty='l2', C=0.001, solver='sag', multi_class='multinomial', verbose=1, random_state=42)  # 0.5702 57.5758 0.5643
    # clf_l2 = LogisticRegression(penalty='l2', C=0.1, solver='saga', multi_class='multinomial', verbose=1, random_state=42) #0.6243 7.5758 0.5611
    # clf_l2 = LogisticRegression(penalty='l2', C=0.001, solver='saga', multi_class='multinomial', verbose=1, random_state=42)  # 0.5702 57.5758 0.5643
    # l2 ovr
    # clf_l2 = LogisticRegression(penalty='l2', C=0.001, solver='lbfgs', multi_class='ovr', verbose=1, random_state=42) #0.56 56.0606 0.5611
    # clf_l2 = LogisticRegression(penalty='l2', C=0.001, solver='liblinear', multi_class='ovr', verbose=1, random_state=42) #0.5765 62.1212 0.5611
    # clf_l2 = LogisticRegression(penalty='l2', C=0.001, solver='newton-cg', multi_class='ovr', verbose=1, random_state=42) #0.56 56.0606 0.5611
    # clf_l2 = LogisticRegression(penalty='l2', C=0.001, solver='newton-cholesky', multi_class='ovr', verbose=1, random_state=42) #0.56 56.0606 0.5611
    # clf_l2 = LogisticRegression(penalty='l2', C=0.001, solver='sag', multi_class='ovr', verbose=1, random_state=42) #0.56 56.0606 0.5611
    #clf_l2 = LogisticRegression(penalty='l2', C=0.001, solver='saga', multi_class='ovr', class_weight='balanced',
                                # random_state=42)  # 0.56 56.0606 0.5611
    # clf_l2 = LogisticRegression(penalty='l2', C=1, solver='newton-cg', multi_class='ovr', class_weight='balanced',
    #                             random_state=42)  # 0.56 56.0606 0.5611
    clf_l2 = LogisticRegression(penalty='l2', C=0.005994842503189409, solver='liblinear', multi_class='ovr', tol=0.00001, class_weight='balanced',
                                random_state=42)

    # elastic-net
    # clf_elastic = LogisticRegression(penalty='elasticnet', C=0.01, l1_ratio=0.2, solver='saga', class_weight='balanced',
    #                                  multi_class='multinomial', random_state=42)  # 0.5851 71.2121 0.5705
    clf_elastic = LogisticRegression(penalty='elasticnet', C=0.007, l1_ratio=0.2, solver='saga', class_weight='balanced', tol=0.00001,
                                     random_state=42) #0.6027, 0.5763
    # no penalty
    # clf_no_penalty = LogisticRegression(solver='lbfgs', verbose=1, random_state=42) #0.6212 0.558
    # clf_no_penalty = LogisticRegression(solver='newton-cg', verbose=1, random_state=42) #0.6212 0.558
    # clf_no_penalty = LogisticRegression(solver='newton-cholesky', verbose=1, random_state=42) #0.6031 0.5768
    clf_no_penalty = LogisticRegression(solver='sag', class_weight='balanced', random_state=42)  # 0.6212 0.558
    # clf_no_penalty = LogisticRegression(solver='saga', verbose=1, random_state=42) #0.6204 0.0 0.558

    # print('Best C % .4f' % clf.C_)
    # l1 norm
    model_fitter = ModelFitter(clf_l1)
    train_score, test_score, sparsity, train_precision, test_precision, exec_time, conf_train, conf_test = \
        model_fitter.fit(X_train, y_train, X_test, y_test, 'l1')
    model_fitter.print_results(train_score, test_score, sparsity, train_precision, test_precision, exec_time, 'l1')
    # x_model_check = model_test_data('quality')
    # y_model_check = model_test_data.drop('quality', axis=1)
    # print(X_mod_check)
    X_mod_check = scalar.transform(X_mod_check)
    # print(X_mod_check)
    print(model_fitter.predict(X_mod_check, y_mod_check))


    # l2 norm
    model_fitter = ModelFitter(clf_l2)
    train_score, test_score, sparsity, train_precision, test_precision, exec_time, conf_train, conf_test = model_fitter.fit(X_train, y_train,
                                                                                                     X_test, y_test,
                                                                                                     'l2')
    model_fitter.print_results(train_score, test_score, sparsity, train_precision, test_precision, exec_time, 'l2')

    # elastic norm
    model_fitter = ModelFitter(clf_elastic)
    train_score, test_score, sparsity, train_precision, test_precision, exec_time, conf_train, conf_test = model_fitter.fit(X_train, y_train,
                                                                                                     X_test, y_test,
                                                                                                     'elastic_net')
    model_fitter.print_results(train_score, test_score, sparsity, train_precision, test_precision, exec_time, 'elastic')
    # model_fitter.plot_confusion_matrix(conf_test)

    # no norm
    model_fitter = ModelFitter(clf_no_penalty)
    train_score, test_score, sparsity, train_precision, test_precision, exec_time, conf_train, conf_test_log = model_fitter.fit(X_train, y_train,
                                                                                                     X_test, y_test,
                                                                                                     'no_norm')
    model_fitter.print_results(train_score, test_score, sparsity, train_precision, test_precision, exec_time, 'no_norm')
    print(conf_test_log)
    print('==================Grid Search=========================================')
    # grid_Search(X_train, y_train)
    # plot_regularize(data, X_train, y_train)
    # print("Length of final model train data", len(train_set))
    # print(test_set.head())
    # print("Length of final test_data", len(test_set))
    # train_set.plot(kind="scatter", x="fixed acidity", y="volatile acidity")
    # plt.show()
    '''
    #plot scatter plot for attributes
    train_set.plot(kind="scatter", x="fixed acidity", y="alcohol", c="quality",  s=np.interp(train_set["volatile acidity"], (train_set["volatile acidity"].min(), train_set["volatile acidity"].max()), (10, 100)), label="volatile acidity", cmap ="viridis", alpha=0.7, colorbar=True)
    corr = train_set.corr()
    plt.subplots(figsize=(15, 10))

    #plot correlation matrics for the attributes
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))
    plt.savefig(fname="corr.png")
    plt.show()
    '''
    # correlation_check
    # cor1 = check_correlation(train_set, "quality") #check the co-orelation for original dataset
    # data["toatal_acidity"] = data["fixed acidity"] + data["volatile acidity"] + data["citric acid"] #total acidity of wine
    # data["alcohol_acidity_balance"] = data["alcohol"]/data["toatal_acidity"] #balance between alohol and acidity in wine
    # data["chloro_suplher_ratio"] = data["chlorides"]/data["sulphates"] #Chlorides to sulphates ratio(atio of these two compounds in a wine)
    # data["acidity_sulpher_ratio"] = data["toatal_acidity"]/data["sulphates"]
    # cor2 = check_correlation(data, "quality")
    # # print(cor1)
    # print(cor2)

    print('==========Ordinal regression========')
    ordinal_regression = OrderedModelPredictor(X_train_stat, y_train_stat, X_test_stat, y_test_stat)
    conf_train_stat, conf_test_stat = ordinal_regression.print_results()
    # print(conf_test)
    print(ordinal_regression.predict(X_mod_check, y_mod_check))
