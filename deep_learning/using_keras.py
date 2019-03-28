import pickle
import tensorflow as tf
from os import path
from tensorflow import keras


def preprocess_data(filepath):
    data = pickle.load(open(filepath, "rb"))
    X_train, y_train, X_test, y_test, vocab_size, max_length = data["x_train"], data["y_train"], data["x_test"], data[
        "y_test"], data["vocab_size"], data["max_length"]

    prepped_X_train = keras.preprocessing.sequence.pad_sequences(sequences=X_train, maxlen=max_length)
    prepped_X_test = keras.preprocessing.sequence.pad_sequences(sequences=X_test, maxlen=max_length)
    prepped_y_train = keras.utils.to_categorical(y_train, num_classes=2)
    prepped_y_test = keras.utils.to_categorical(y_test, num_classes=2)

    return prepped_X_train, prepped_X_test, prepped_y_train, prepped_y_test, vocab_size, max_length


def classify(X_train, X_test, y_train, y_test, vocab_size, max_length):
    # initialize the Sequential model
    model = keras.Sequential()

    # add layers to the model
    model.add(keras.layers.Embedding(input_dim=vocab_size, output_dim=128))
    model.add(keras.layers.LSTM(activation='relu', units=128))
    model.add(keras.layers.Dense(units=2))

    # compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.RMSprop(),
                  metrics=['accuracy'])

    # run on GTX 1080 Ti
    with tf.device('/GPU:0'):
        # train the model
        model.fit(X_train, y_train, batch_size=128, epochs=4, verbose=1)

    # save the model
    model.save("lstm_model.h5")

    # predict on test-set
    pred = model.predict(X=X_test, y=y_test)
    print('Prediction:\t'.format(pred))

    # evaluate
    score = model.evaluate(X_test, y_test, verbose=1)  # Should acquire 90 % accuracy from the LSTM
    print('Test loss:\t', score[0],
          '\nTest accuracy:\t', score[1])


def get_file_path(file):
    return path.abspath(file)


def main():
    dataset_sklearn = "data/keras-data.pickle"
    X_train, X_test, y_train, y_test, vocab_size, max_length = preprocess_data(get_file_path(dataset_sklearn))
    classify(X_train, X_test, y_train, y_test, vocab_size, max_length)


if __name__ == '__main__':
    main()







