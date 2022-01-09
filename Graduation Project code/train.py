"""
Train our RNN on extracted features or images.
"""
import tensorflow as tf
import keras
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import ResearchModels
from data import DataSet
import time
import os.path

def train(data_type, seq_length, model, saved_model=None,
          class_limit=None, image_shape=None,
          load_to_memory=False, batch_size=32, nb_epoch=100):
    # Helper: Save the model.
    checkpointer = ModelCheckpoint(
        filepath=os.path.join('data', 'checkpoints', model + '-' + data_type + \
            '.{epoch:03d}-{val_loss:.3f}.hdf5'),
        verbose=1,
        save_best_only=True)

    # Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join('data', 'logs', model))

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=20)
    class myCallBack(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs):
            if logs.get('val_loss') <= 0.05:
                print('Enough Training For Now ya Zo2!!')
                self.model.stop_training = True

    call = myCallBack()
    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join('data', 'logs', model + '-' + 'training-' + \
        str(timestamp) + '.log'))

    # Get the data and process it.
    if image_shape is None:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit
        )
    else:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit,
            image_shape=image_shape
        )

    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    steps_per_epoch = (len(data.data) * 0.8) // batch_size
    v_steps = (len(data.data) * 0.2) // batch_size
    if load_to_memory:
        # Get data.
        X, y = data.get_all_sequences_in_memory('train', data_type)
        X_test, y_test = data.get_all_sequences_in_memory('test', data_type)
    else:
        # Get generators.
        if model == 'ctc':
            generator = data.ctc_frame_generator(batch_size, 'train', data_type)
            val_generator = data.ctc_frame_generator(batch_size, 'test', data_type)
        else:
            generator = data.frame_generator(batch_size, 'train', data_type)
            val_generator = data.frame_generator(batch_size, 'test', data_type)


    # Get the model.
    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #set_session(tf.Session(config=config))
    # print(f'>>>>>>>{generator}')
    ctc_input_shape = (300, 300, 3) #(batch_size, data.max_frames,300*300*3)
    rm = ResearchModels(len(data.classes), model, seq_length, saved_model, ctc_input_shape=ctc_input_shape)
    
    # Fit!
    if load_to_memory:
        # Use standard fit.
        rm.model.fit(
            X,
            y,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1,
            callbacks=[tb, csv_logger, call,checkpointer],
            epochs=nb_epoch)
    else:
        # Use fit generator.
        rm.model.fit_generator(
            generator=generator,
            steps_per_epoch=steps_per_epoch,
            epochs=nb_epoch,
            verbose=1,
            callbacks=[tb, csv_logger, checkpointer, call],
            validation_data=val_generator,
            validation_steps=v_steps,
            workers=12)

def main():
    """These are the main training settings. Set each before running
    this file."""
    # model can be one of lstm, lrcn, mlp, conv_3d, c3d
    model = 'ctc'
    saved_model = None  # None or weights file
    class_limit = None  # int, can be 1-101 or None
    seq_length = 30
    load_to_memory = False  # pre-load the sequences into memory
    batch_size = 1 
    nb_epoch = 10

    # Chose images or features and image shape based on network.
    if model in ['conv_3d', 'c3d', 'lrcn','custome']:
        data_type = 'images'
        image_shape = (300, 300, 3)
    elif model == 'ctc':
        data_type = 'images'
        image_shape = (300, 300, 3)

    elif model in ['lstm', 'mlp']:
        data_type = 'features'
        image_shape = None
    else:
        raise ValueError("Invalid model. See train.py for options.")

    train(data_type, seq_length, model, saved_model=saved_model,
          class_limit=class_limit, image_shape=image_shape,
          load_to_memory=load_to_memory, batch_size=batch_size, nb_epoch=nb_epoch)

if __name__ == '__main__':
    main()
