import pickle
from os import path
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score



def _run(filepath):
    data = pickle.load(open(filepath, "rb"))
    x_train, y_train, x_test, y_test = data["x_train"], data["y_train"], data["x_test"], data["y_test"]
    vector = HashingVectorizer(stop_words="english", binary=True)
    transformed_x_train = vector.fit_transform(x_train)
    transformed_x_test = vector.fit_transform(x_test)

    '''DecisionTreeClassifier'''
    dtc_classifier = DecisionTreeClassifier(criterion="gini")

    # fit the transformed data
    dtc_classifier.fit(X=transformed_x_train,y=y_train)

    # make predictions for the test set
    dtc_pred = dtc_classifier.predict(X=transformed_x_test)


    '''Naive Bayes w/BernoulliNB'''
    nb_classifier = BernoulliNB()

    # fit the transformed data
    nb_classifier.fit(X=transformed_x_train,y=y_train)

    # make predictions for the test set
    nb_pred = nb_classifier.predict(X=transformed_x_test)


    '''Accuracy'''
    print_accuracy(nb_pred, y_test)

    # DecisionTreeClassifier --> The accuracy for this classifier:	0.8645118288796274
    # BernoulliNB -->  The accuracy for this classifier:	0.8380500735474381


def print_accuracy(y_pred, y_true):
    print("The accuracy for this classifier:\t{}".format(accuracy_score(y_pred, y_true)))

def get_file_path(file):
    return path.abspath(file)


def main():
    dataset_sklearn = "data/sklearn-data.pickle"
    _run(get_file_path(dataset_sklearn))


if __name__ == '__main__':
    main()





