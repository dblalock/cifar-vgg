
from __future__ import print_function
import keras
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
from keras import backend as K
from keras import regularizers


LABELS_TRAIN_PATH = 'cifar100_labels_train.npy'
LABELS_TEST_PATH = 'cifar100_labels_test.npy'
MODEL_SAVE_PATH = 'cifar100vgg_model.h5'
SOFTMAX_INPUTS_TRAIN_PATH = 'cifar100_softmax_inputs_train.npy'
SOFTMAX_OUTPUTS_TRAIN_PATH = 'cifar100_softmax_outputs_train.npy'
SOFTMAX_INPUTS_TEST_PATH = 'cifar100_softmax_inputs_test.npy'
SOFTMAX_OUTPUTS_TEST_PATH = 'cifar100_softmax_outputs_test.npy'
SOFTMAX_W_PATH = 'cifar100_softmax_W.npy'
SOFTMAX_B_PATH = 'cifar100_softmax_b.npy'


class cifar100vgg:
    def __init__(self, train=False, saveas=None):
        self.num_classes = 100
        self.weight_decay = 0.0005
        self.x_shape = [32, 32, 3]

        self.model = self.build_model()
        if train:
            self.model = self.train(self.model)
        else:
            self.model.load_weights('cifar100vgg.h5')

        if saveas:
            self.model.save(saveas)

    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

        model = Sequential()
        weight_decay = self.weight_decay

        model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=self.x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, name='dense_softmax'))
        model.add(Activation('softmax'))
        return model


    def normalize(self,X_train,X_test):
        #this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(X_train,axis=(0,1,2,3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        print(mean)
        print(std)
        X_train = (X_train-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7)
        return X_train, X_test

    def normalize_production(self,x):
        #this function is used to normalize instances in production according to saved training set statistics
        # Input: X - a training set
        # Output X - a normalized training set according to normalization constants.

        #these values produced during first training and are general for the standard cifar10 training set normalization
        mean = 121.936
        std = 68.389
        return (x-mean)/(std+1e-7)

    def predict(self,x,normalize=True,batch_size=50):
        if normalize:
            x = self.normalize_production(x)
        return self.model.predict(x,batch_size)

    def train(self,model):

        #training parameters
        batch_size = 128
        maxepoches = 250
        learning_rate = 0.1
        lr_decay = 1e-6
        lr_drop = 20

        # The data, shuffled and split between train and test sets:
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train, x_test = self.normalize(x_train, x_test)

        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)


        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))
        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)


        #data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)



        #optimization details
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])


        # training process in a for loop with learning rate drop every 25 epoches.

        historytemp = model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=maxepoches,
                            validation_data=(x_test, y_test),callbacks=[reduce_lr],verbose=2)
        model.save_weights('cifar100vgg.h5')
        return model


def main():
    save_labels = True
    save_model = True
    check_model = True
    save_softmax_params = True
    save_test_activations = True
    save_train_activations = True
    check_test_activations = True
    check_train_activations = True

    (X_train, y_train), (X_test, y_test) = cifar100.load_data()
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    if save_labels:
        print("Saving train and test labels...")
        np.save(LABELS_TRAIN_PATH, y_train, allow_pickle=False)
        np.save(LABELS_TEST_PATH, y_test, allow_pickle=False)

    if save_model:
        print("Saving model...")
        model = cifar100vgg(saveas=MODEL_SAVE_PATH)
    model = keras.models.load_model(MODEL_SAVE_PATH)

    def normalize(X):
        mean = 121.936
        std = 68.389
        return (X - mean) / (std + 1e-7)

    X_train = normalize(X_train)  # no better than chance without this line
    X_test = normalize(X_test)  # no better than chance without this line

    if check_model:
        limit_n = 1000
        y_probs_hat = model.predict(X_test[:limit_n])
        y_hat = np.argmax(y_probs_hat, 1)
        # wrong = y_hat != y_test.ravel()[:limit_n]
        correct = y_hat == y_test.ravel()[:limit_n]

        acc = np.mean(correct)
        print("the test accuracy is: ", acc)
        assert acc > .65  # sanity check to make sure it worked

    # now pull out the activations for X_train and X_test, as well as the
    # weights in the final softmax layer
    softmax = model.get_layer('dense_softmax')
    inp_tensor = softmax.input
    out_tensor = softmax.output
    model_in = model.input
    W, b = softmax.get_weights()
    print("W, b shapes: ", W.shape, b.shape)
    print("inp, outp tensors: ", inp_tensor, out_tensor)

    if save_softmax_params:
        print("Saving softmax parameters...")
        np.save(SOFTMAX_W_PATH, W, allow_pickle=False)
        np.save(SOFTMAX_B_PATH, b, allow_pickle=False)

    if save_train_activations or save_test_activations:
        N_train = len(X_train)
        N_test = len(X_test)
        nbatches_train = 100
        nbatches_test = 10
        batch_sz_train = N_train // nbatches_train
        batch_sz_test = N_test // nbatches_test
        assert nbatches_train * batch_sz_train == N_train
        assert nbatches_test * batch_sz_test == N_test

        input_sz = K.int_shape(inp_tensor)[-1]
        output_sz = K.int_shape(out_tensor)[-1]
        sess = K.get_session()

    # ------------------------------------------------ test activations
    if save_test_activations:
        print("Saving test activations (softmax input and output)...")
        inputs_test = np.empty((N_test, input_sz), dtype=np.float32)
        outputs_test = np.empty((N_test, output_sz), dtype=np.float32)
        for i in range(nbatches_test):
            print("running on batch {}/{}...".format(i + 1, nbatches_test))
            start_idx = i * batch_sz_test
            end_idx = start_idx + batch_sz_test
            X = X_test[start_idx:end_idx]
            inp, outp = sess.run([inp_tensor, out_tensor],
                                 feed_dict={model_in: X})
            inputs_test[start_idx:end_idx] = inp
            outputs_test[start_idx:end_idx] = outp

            # sanity check accuracy for this batch
            lbls = y_test[start_idx:end_idx].ravel()
            logits = outp
            lbls_hat = np.argmax(logits, axis=1)
            print("acc: ", np.mean(lbls == lbls_hat))

        print("inputs_test min: ", np.min(inputs_test))
        print("inputs_test max: ", np.max(inputs_test))
        print("outputs_test min", np.min(outputs_test))
        print("outputs_test max", np.max(outputs_test))
        np.save(SOFTMAX_INPUTS_TEST_PATH, inputs_test, allow_pickle=False)
        np.save(SOFTMAX_OUTPUTS_TEST_PATH,
                outputs_test, allow_pickle=False)

    # ------------------------------------------------ training activations
    if save_train_activations:
        print("Saving train activations (softmax input and output)...")
        inputs_train = np.empty((N_train, input_sz), dtype=np.float32)
        outputs_train = np.empty((N_train, output_sz), dtype=np.float32)
        for i in range(nbatches_train):
            print("running on batch {}/{}...".format(i + 1, nbatches_train))
            start_idx = i * batch_sz_train
            end_idx = start_idx + batch_sz_train
            X = X_train[start_idx:end_idx]
            inp, outp = sess.run([inp_tensor, out_tensor],
                                 feed_dict={model_in: X})
            inputs_train[start_idx:end_idx] = inp
            outputs_train[start_idx:end_idx] = outp

            # sanity check accuracy for this batch
            lbls = y_train[start_idx:end_idx].ravel()
            logits = outp
            lbls_hat = np.argmax(logits, axis=1)
            print("acc: ", np.mean(lbls == lbls_hat))

        print("inputs_train min: ", np.min(inputs_train))
        print("inputs_train max: ", np.max(inputs_train))
        print("outputs_train min", np.min(outputs_train))
        print("outputs_train max", np.max(outputs_train))
        np.save(SOFTMAX_INPUTS_TRAIN_PATH, inputs_train, allow_pickle=False)
        np.save(SOFTMAX_OUTPUTS_TRAIN_PATH, outputs_train,
                allow_pickle=False)

    # ------------------------------------------------ check train activations
    if check_test_activations:
        X = np.load(SOFTMAX_INPUTS_TEST_PATH)
        Y = np.load(SOFTMAX_OUTPUTS_TEST_PATH)
        W = np.load(SOFTMAX_W_PATH)
        b = np.load(SOFTMAX_B_PATH)

        print("---- softmax for test data")
        print("X shape: ", X.shape)
        print("Y shape: ", Y.shape)
        print("W shape: ", W.shape)
        print("b shape: ", b.shape)

        Y_hat = (X @ W) + b
        diffs = Y - Y_hat
        mse = np.mean(diffs * diffs) / np.var(Y)
        print("mse: ", mse)
        assert mse < 1e-7

        y = np.load(LABELS_TEST_PATH).ravel()
        logits_hat = Y_hat
        y_hat = np.argmax(logits_hat, axis=1).ravel()
        print("acc: ", np.mean(y == y_hat))
        print("class counts: ", np.bincount(y))

    # ------------------------------------------------ check train activations
    if check_train_activations:
        X = np.load(SOFTMAX_INPUTS_TRAIN_PATH)
        Y = np.load(SOFTMAX_OUTPUTS_TRAIN_PATH)
        W = np.load(SOFTMAX_W_PATH)
        b = np.load(SOFTMAX_B_PATH)

        print("---- softmax for train data")
        print("X shape: ", X.shape)
        print("Y shape: ", Y.shape)
        print("W shape: ", W.shape)
        print("b shape: ", b.shape)

        Y_hat = (X @ W) + b
        diffs = Y - Y_hat
        mse = np.mean(diffs * diffs) / np.var(Y)
        print("mse: ", mse)
        assert mse < 1e-7

        y = np.load(LABELS_TRAIN_PATH).ravel()
        logits_hat = Y_hat
        y_hat = np.argmax(logits_hat, axis=1).ravel()
        print("acc: ", np.mean(y == y_hat))
        print("class counts: ", np.bincount(y))


if __name__ == '__main__':
    main()
