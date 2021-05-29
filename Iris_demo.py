from ML_demo import Dataset

path = "/Users/woo/PycharmProjects/IrisDataset/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
ds = Dataset(path, names)
ds.get_shape()
ds.get_first_20()
ds.describe()
ds.plot_box(2, 2)
ds.plot_hist()
ds.plot_scatter()
ds.evaluate(4, 10)



