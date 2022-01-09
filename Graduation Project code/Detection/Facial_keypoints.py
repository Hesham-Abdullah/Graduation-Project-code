# Training Code

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, Activation, BatchNormalization
from keras.layers import Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.callbacks import ModelCheckpoint, History, LearningRateScheduler, TensorBoard, EarlyStopping, ReduceLROnPlateau
from datetime import datetime

def data_loader():
    
    # Load dataset file
    data_frame = pd.read_csv(r'..\data\Detection_Data\FacialKeyPoints\training.csv')
    
    data_frame['Image'] = data_frame['Image'].apply(lambda i: np.fromstring(i, sep=' '))
    data_frame = data_frame.dropna()  # Get only the data with 15 keypoints
   
    # Extract Images pixel values
    imgs_array = np.vstack(data_frame['Image'].values)/ 255.0
    imgs_array = imgs_array.astype(np.float32)    # Normalize, target values to (0, 1)
    imgs_array = imgs_array.reshape(-1, 96, 96, 1)
        
    # Extract labels (key point cords)
    labels_array = data_frame[data_frame.columns[:-1]].values
    labels_array = (labels_array - 48) / 48    # Normalize, traget cordinates to (-1, 1)
    labels_array = labels_array.astype(np.float32) 
    
    # shuffle the train data
    # imgs_array, labels_array = shuffle(imgs_array, labels_array, random_state=9)  
    
    return imgs_array, labels_array

def sanity_check():
    imgs, labels = data_loader()
    print(imgs.shape)
    print(labels.shape)

    n=0
    labels[n] = (labels[n]*48)+48
    image = np.squeeze(imgs[n])
    plt.imshow(image, cmap='gray')
    plt.plot(labels[n][::2], labels[n][1::2], 'ro')
    plt.show()


def create_model():
    model = Sequential()
    
    model.add(Conv2D(16, (3,3), padding='same', input_shape=X_train.shape[1:])) # Input shape: (96, 96, 1)
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(16, (3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=2))
    
    model.add(Conv2D(32, (3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=2))
    
    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=2))
    
    model.add(Conv2D(128, (3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=2))
    
    model.add(Conv2D(256, (3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(256, (3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=2))
    
    # Convert all values to 1D array
    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(30))
    
    return model
def calls():
    checkpoint = ModelCheckpoint(
        filepath='checkpoints\checkpoint.hdf5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        period=1)

    early = EarlyStopping(monitor='val_loss',
                        min_delta=0,
                        patience=10,
                        verbose=1,
                        mode='auto')

    class myCallBack(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('val_accuracy') > 0.90):
                print ('\nReached 0.998 Validation accuracy!')
                self.model.stop_training = True

    my_call = myCallBack()

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                                verbose=1, mode='auto', min_delta=0,
                                cooldown=0, min_lr=1.0e-04)

    lr_schedule = LearningRateScheduler(lambda epoch: 1e-5 * 10**(epoch / 10))
    return [checkpoint, my_call, early]#lr_schedule

def plot_charts(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label = 'Training Accuracy')
    plt.plot(epochs, val_acc, 'b', label = 'Validation Accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.show()

    plt.plot(epochs, loss, 'r', label = 'Training Loss')
    plt.plot(epochs, val_loss, 'b', label = 'Validation Loss')
    plt.title('Training and validation Loss')
    plt.legend()
    plt.figure()
    plt.show()

if __name__ == "__main__":
    X_train, y_train = data_loader()
    epochs = 60
    batch_size = 64

    model = create_model()
    hist = History()
    mon = calls()

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    model_fit = model.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, callbacks=mon, verbose=1)
    
    plot_charts(model_fit)

    now2 = datetime.now()
    timestamp2 = datetime.timestamp(now2)
    model.save(f'checkpoints\model_{timestamp2}.h5')

