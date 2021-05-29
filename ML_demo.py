# Load libraries
import pandas as pd
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


class Dataset:
    def __init__(self, path_name, class_names):
        self.path_name = path_name
        self.class_names = class_names
        self.main_dataframe = read_csv(path_name, names=class_names)

    # returns shape of dataset (size, number of attributes)
    def get_shape(self):
        print(self.main_dataframe.shape)

    # returns first 20 items
    def get_first_20(self):
        print(self.main_dataframe.head(20))

    # returns first n items
    def get_first_n(self, n):
        print(self.main_dataframe.head(n))

    # returns basic statistical data
    def describe(self):
        print(self.main_dataframe.describe())

    # groups dataset by class
    def group_by_class(self):
        print(self.main_dataframe.groupby('class').size())

    # box-and-whisker plots
    def plot_box(self, x, y):
        self.main_dataframe.plot(kind='box', subplots=True, layout=(x, y), sharex=False, sharey=False)
        pyplot.show()

    def plot_hist(self):
        self.main_dataframe.hist()
        pyplot.show()

    def plot_scatter(self):
        scatter_matrix(self.main_dataframe)
        pyplot.show()

    def evaluate(self, n, k):
        array = self.main_dataframe.values
        X = array[:, 0:n]
        y = array[:, n]
        X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
        models = list()
        models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('CART', DecisionTreeClassifier()))
        models.append(('NB', GaussianNB()))
        models.append(('SVM', SVC(gamma='auto')))

        results = list()
        names = list()
        for name, model in models:
            kfold = StratifiedKFold(n_splits=k, random_state=1, shuffle=True)
            cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
            results.append(cv_results)
            names.append(name)
            print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

        pyplot.boxplot(results, labels=names)
        pyplot.title('Algorithm Comparison')
        pyplot.show()

        # Make predictions on validation dataset
        model = SVC(gamma='auto')
        model.fit(X_train, Y_train)
        predictions = model.predict(X_validation)

        # Evaluate predictions
        print(accuracy_score(Y_validation, predictions))
        print(confusion_matrix(Y_validation, predictions))
        print(classification_report(Y_validation, predictions))
        

path = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
ds = Dataset(path, names)
ds.get_shape()
ds.get_first_20()
ds.describe()
ds.plot_box(2, 2)
ds.plot_hist()
ds.plot_scatter()
ds.evaluate(4, 10)


