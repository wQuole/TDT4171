import pickle, os, sklearn
path = "C:/Users/wquole/code/pythonCode/TDT4171/deep_learning/data"
os.chdir(path)


data = pickle.load(open("sklearn-data.pickle", "rb"))
for v in data["x_train"]:
    print(v) 