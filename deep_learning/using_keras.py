import pickle
import tensorflow as tf
from sys import argv

from os import path
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense
from keras.utils import plot_model


def preprocess_data(filepath):
    '''
    Preprocess the data from the pickle file
    '''
    data = pickle.load(open(filepath, "rb"))
    X_train, y_train, X_test, y_test = data["x_train"], data["y_train"], data["x_test"], data["y_test"]
    vocab_size, max_length = data["vocab_size"], data["max_length"]

    # Dataset to huge for 6 hours of compute
    partition = round(len(X_train)/200)

    X_train = pad_sequences(sequences=X_train[:partition], maxlen=max_length)
    X_test = pad_sequences(sequences=X_test[:partition], maxlen=max_length)

    y_train = to_categorical(y_train[:partition], num_classes=2)
    y_test = to_categorical(y_test[:partition], num_classes=2)

    return X_train, X_test, y_train, y_test, vocab_size, max_length


def create_model(vocab_size, max_length):
    '''
    :param vocab_size: int --> number of different words in vocabulary
    :param max_length: int --> longest review in dataset
    :return: keras.engine.sequential.Sequential trained model
    '''
    # initialize the Sequential model
    model = Sequential()

    # add layers to the model
    model.add(Embedding(input_dim=vocab_size, output_dim=256, input_length=max_length))
    model.add(LSTM(256))
    model.add(Dense(2, activation="sigmoid"))

    # compile the model
    # chose RMSProp as the optimizer, since it is usually a good choice for recurrent neural networks.
    model.compile(optimizer=RMSprop(), loss='binary_crossentropy', metrics=['accuracy'])

    # print model
    model.summary()

    return model


def train(model, X_train, y_train, batch_size=256, epochs=10, device=argv[1].lower()):
    '''
    :param model:   untrained keras.engine.sequential.Sequential model
    :param X_train: numpy.ndarray containing  training data
    :param y_train: numpy.ndarray containing target (label) data
    :param batch_size: int --> number of samples per gradient update
    :param epochs:  int --> number of epochs to train the model
    :param device:  sys.argv[1] --> user inut 'cpu' or 'gpu'
    '''
    # run on GPU --> {kaggle: 'Tesla P100-PCIE-16GB', personal_desktop: 'GTX 1080 Ti'}
    if device == 'gpu':
        print(f"\nUsing {device} to train")
        with tf.device('/GPU:0'):
            # train the model
            model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

    # run on CPU --> {personal_macbook: '2,2 GHz i7 8750h', personal_desktop: '5.0 GHz i9 9900k'}
    else:
        print(f"Using {device} to train")
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

    # save the model
    model.save("LSTM_model.h5")


def eval(model, X_test, y_test):
    '''
    :param model:   keras.engine.sequential.Sequential --> trained model
    :param X_test:  numpy.ndarray --> containing test data
    :param y_test:  numpy.ndarray --> containing target (label) data
    '''
    # evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=1)  # Should acquire at least 90 % accuracy from the LSTM
    print('Test loss:\t', loss,
          '\nTest accuracy:\t', acc)


def main():
    # initialize the data
    dataset_sklearn = "data/keras-data.pickle"
    X_train, X_test, y_train, y_test, vocab_size, max_length = preprocess_data(path.abspath((dataset_sklearn)))

    # create model, train it and save it
    model = create_model(vocab_size, max_length)
    history = train(model, X_train, y_train)


    # evaluate the trained model on test data
    eval(model, X_test, y_test)
    plot_model(model=model, show_shapes=1, show_layer_names=1)
    '''
    >>> Test loss:        0.1628359776079664
    >>> Test accuracy:    0.9487619514586909
    '''

if __name__ == '__main__':
    main()







