import pickle
from os import path
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score



def _run(filepath, classifier='dtc'):
    '''
    Run method for classifying movie reviews,
    using Decision Tree Classifier and Bernoulli Naive Bayes
    '''
    # prepare data
    data = pickle.load(open(filepath, "rb"))
    x_train, y_train, x_test, y_test = data["x_train"], data["y_train"], data["x_test"], data["y_test"]

    # init vectorizer
    vector = HashingVectorizer(stop_words="english", binary=True)

    # transform the training and test data
    transformed_X_train = vector.fit_transform(x_train)
    transformed_X_test = vector.fit_transform(x_test)


    '''DecisionTreeClassifier'''
    dtc_classifier = DecisionTreeClassifier(criterion="entropy")

    # fit the transformed data
    dtc_classifier.fit(transformed_X_train, y_train)

    # make predictions for the test set
    dtc_pred = dtc_classifier.predict(transformed_X_test)


    '''Naive Bayes w/BernoulliNB'''
    bnb_classifier = BernoulliNB()

    # fit the transformed data
    bnb_classifier.fit(transformed_X_train, y_train)

    # make predictions for the test set
    bnb_pred = bnb_classifier.predict(transformed_X_test)


    '''Accuracy'''
    if classifier == 'dtc':
        print_accuracy(dtc_pred, y_test)
    elif classifier == 'bnb':
        print_accuracy(bnb_pred, y_test)
    else:
        print("Choose between the two classifiers 'dtc' and 'bnb'")

    # DecisionTreeClassifier    --> The accuracy for this classifier:	0.8645118288796274
    # BernoulliNB               -->  The accuracy for this classifier:	0.8380500735474381


def print_accuracy(y_pred, y_true):
    print(f"The accuracy for this classifier:\t{accuracy_score(y_pred, y_true)}")


def main():
    dataset_sklearn = "data/sklearn-data.pickle"
    _run(path.abspath(dataset_sklearn), classifier='bnb')


if __name__ == '__main__':
    main()





