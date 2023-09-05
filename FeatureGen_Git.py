import argparse
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import re
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Flatten, Bidirectional, LSTM, GRU, Dropout, Masking, Input, GaussianNoise, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, History, ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import confusion_matrix


parser = argparse.ArgumentParser(description='Train the model')
parser.add_argument('-promlen', dest='PromLen', help='Length of Promoter', nargs='?', default=50)
parser.add_argument('-padsize', dest='padSize', help='Size of padding on both sides', nargs='?', default=12)
parser.add_argument('-sessiondir', dest='sessiondir', help='Session Dir', nargs='?', default='.')
parser.add_argument('-modelName', dest='modelName', help='Name of the model, see models.py', nargs='?',
                    default='model_lstm')
parser.add_argument('-batchsize', dest='batchSize', help='Batch size', nargs='?', default=64)
parser.add_argument('-epochs', dest='epochs', help='Number of training epochs', nargs='?', default=40)
args, unknown = parser.parse_known_args()

def read_fasta_file(file_path):
    sequences = []
    with open(file_path, 'r') as file:
        sequence = ""
        for line in file:
            if line.startswith(">"):
                if sequence:
                    sequences.append(sequence)
                    sequence = ""
                continue
            sequence += line.strip()
        if sequence:  # add the last sequence
            sequences.append(sequence)
    return sequences

def mutate_sequence(sequence_list):
    sequences_final = []
    new_sequences = []
    for sequence in sequence_list:
        if len(sequence) > 52:
            # Looking at the first 3 NT
            if sequence[0] == 'T':
                new_sequences.append('C' + sequence[1:])
            elif sequence[0] == 'C':
                new_sequences.append('T' + sequence[1:])
            elif sequence[0] == 'G':
                new_sequences.append('A' + sequence[1:])
            elif sequence[0] == 'A':
                new_sequences.append('G' + sequence[1:])

            if sequence[1] == 'T':
                new_sequences.append(sequence[:1] + 'C' + sequence[2:])
            elif sequence[1] == 'C':
                new_sequences.append(sequence[:1] + 'T' + sequence[2:])
            elif sequence[1] == 'G':
                new_sequences.append(sequence[:1] + 'A' + sequence[2:])
            elif sequence[1] == 'A':
                new_sequences.append(sequence[:1] + 'G' + sequence[2:])

            if sequence[2] == 'T':
                new_sequences.append(sequence[:2] + 'C' + sequence[3:])
            elif sequence[2] == 'C':
                new_sequences.append(sequence[:2] + 'T' + sequence[3:])
            elif sequence[2] == 'G':
                new_sequences.append(sequence[:2] + 'A' + sequence[3:])
            elif sequence[2] == 'A':
                new_sequences.append(sequence[:2] + 'G' + sequence[3:])

            # Looking at the last 3 NT
            if sequence[-1] == 'T':
                new_sequences.append(sequence[:-1] + 'C')
            elif sequence[-1] == 'C':
                new_sequences.append(sequence[:-1] + 'T')
            elif sequence[-1] == 'G':
                new_sequences.append(sequence[:-1] + 'A')
            elif sequence[-1] == 'A':
                new_sequences.append(sequence[:-1] + 'G')

            if sequence[-2] == 'T':
                new_sequences.append(sequence[:-2] + 'C' + sequence[-1:])
            elif sequence[-2] == 'C':
                new_sequences.append(sequence[:-2] + 'T' + sequence[-1:])
            elif sequence[-2] == 'G':
                new_sequences.append(sequence[:-2] + 'A' + sequence[-1:])
            elif sequence[-2] == 'A':
                new_sequences.append(sequence[:-2] + 'G' + sequence[-1:])

            if sequence[-3] == 'T':
                new_sequences.append(sequence[:-3] + 'C' + sequence[-2:])
            elif sequence[-3] == 'C':
                new_sequences.append(sequence[:-3] + 'T' + sequence[-2:])
            elif sequence[-3] == 'G':
                new_sequences.append(sequence[:-3] + 'A' + sequence[-2:])
            elif sequence[-3] == 'A':
                new_sequences.append(sequence[:-3] + 'G' + sequence[-2:])

    sequence_list += new_sequences
    return sequence_list

def Sliding_window_aug(sequence_list, step_size):
    kmers = []
    sequence_zero = []
    for sequence in sequence_list:
        if '0' not in sequence and len(sequence) == args.padSize + args.PromLen:
            n_iterations = (len(sequence) - args.PromLen) // step_size + 1

            for i in range(n_iterations):
                start_index = i * step_size
                end_index = start_index + args.PromLen
                kmer = sequence[start_index:end_index]
                kmers.append(kmer)
        else:
            sequence_zero.append(sequence)
    kmers = kmers + sequence_zero
    return kmers

def Anne_one_hot_encode(seq):
    # One hot encode your sequences for the CNN
    mapping = dict(zip("ACGT", range(4)))
    seq2 = [mapping[i] for i in seq]
    return np.eye(4)[seq2]

def write_log(S):
    # write to console and logfile
    print(S)
    f = open(args.sessiondir + '/' + "log_ppp_{model_name}.log".format(model_name=args.modelName), "a")
    f.write(S + '\n')
    f.close()

def Validation_report(features, labels, name):
    # F1 = A measure that combines precision and recall is the harmonic mean of precision and recall, the traditional F-measure or balanced F-score:
    # F1 = 2 * (precision * recall) / (precision + recall)
    # precision = TP / TP + FP
    # recall = TP / TP + FN
    predicted_labels = model.predict(features)
    cm = confusion_matrix(np.argmax(labels, axis=1), np.argmax(predicted_labels, axis=1))
    TP = cm[1][1]
    FP = cm[0][1]
    FN = cm[1][0]
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * (precision * recall) / (precision + recall)

    write_log('Confusion matrix of ' + name + '\n' + str(cm) + '| ' + args.sessiondir)
    write_log('Precision  TP / TP + FP                              ' + str(round(precision, 2)))
    write_log('Recall     TP / TP + FN                              ' + str(round(recall, 2)))
    write_log('F1 = 2 * (precision * recall) / (precision + recall) ' + str(round(F1, 2)))

    # Confusion matrix info is in the log, but one extra clean file:
    f = open(args.sessiondir + '/' + "Confusion_matrix.txt", "a")
    f.write(args.sessiondir + '\n')
    f.write('Confusion matrix of ' + name + '\n' + str(cm) + '\n')
    f.write('Precision  TP / TP + FP                              ' + str(round(precision, 2)) + '\n')
    f.write('Recall     TP / TP + FN                              ' + str(round(recall, 2)) + '\n')
    f.write('F1 = 2 * (precision * recall) / (precision + recall) ' + str(round(F1, 2)) + '\n\n')
    f.write('Precision\tRecall\tF1\n')
    f.write(str(round(precision, 2)) + '\t' + str(round(recall, 2)) + '\t' + str(round(F1, 2)) + '\n\n')
    f.close()

regulon = read_fasta_file('regulon.txt')

non_regulon = read_fasta_file('non_regulon.txt')

# Data augmentation via making the sliding window, adding in the reverse complement, and adding base flipping

for idx, item in enumerate(regulon):
    regulon[idx] = re.sub('[BDEFHIJKLMNOPQRSUVWXYZ]', 'G', item)

# Include substitutions from C <-> T in the first 3 nucleotides of the sequence
regulon = mutate_sequence(regulon)

# Sliding window
regulon = Sliding_window_aug(regulon, 3)

# Makes sure all the sequences are the same length and removes any non ACTG with a G
for idx, item in enumerate(regulon):
    regulon[idx] = item.replace(item, item[-args.PromLen:])
for i in non_regulon.copy():
    if len(i) < args.PromLen:
        non_regulon.remove(i)
for idx, item in enumerate(non_regulon):
    non_regulon[idx] = item.replace(item, item[-args.PromLen:])
for idx, item in enumerate(regulon):
    regulon[idx] = re.sub('[^ACTG0]', 'G', item.strip().upper())
print(len(regulon))
print(len(non_regulon))

test_set_fraction = 0.1
test_set_percentage = 100 / (100 * test_set_fraction)

training_sequences = []
training_response = []
test_sequences = []
test_response = []

# Assignes a value to the sequences so machine can discriminate between motif and non-motif
for i in range(0, len(regulon)):
    if i % test_set_percentage == 0:
        test_sequences.append(regulon[i])
        test_response.append(1)
    else:
        training_sequences.append(regulon[i])
        training_response.append(1)

for i in range(0, len(non_regulon)):
    if i % test_set_percentage == 0:
        test_sequences.append(non_regulon[i])
        test_response.append(0)
    else:
        training_sequences.append(non_regulon[i])
        training_response.append(0)


integer_encoder = LabelEncoder()
one_hot_encoder = OneHotEncoder(categories='auto')

train_features = []
for sequence in training_sequences: train_features.append(Anne_one_hot_encode(sequence))
np.set_printoptions(threshold=40)
train_features = tf.keras.preprocessing.sequence.pad_sequences(train_features)
train_features = np.stack(train_features)


test_features = []
for sequence in test_sequences:    test_features.append(Anne_one_hot_encode(sequence))
np.set_printoptions(threshold=40)
test_features = tf.keras.preprocessing.sequence.pad_sequences(test_features)
test_features = np.stack(test_features)

train_labels = training_response
one_hot_encoder = OneHotEncoder(categories='auto')
train_labels = np.array(train_labels).reshape(-1, 1)
train_labels = one_hot_encoder.fit_transform(train_labels).toarray()

test_labels = test_response
one_hot_encoder = OneHotEncoder(categories='auto')
test_labels = np.array(test_labels).reshape(-1, 1)
test_labels = one_hot_encoder.fit_transform(test_labels).toarray()

# Save the augmented data to a NumPy array
np.save('train_features.npy', train_features)
np.save('train_labels.npy', train_labels)
np.save(args.sessiondir + '/test_features.npy', test_features)
np.save(args.sessiondir + '/test_labels.npy', test_labels)

def model_lstm_bi_2(input_shape):
    model = Sequential(name='model_lstm_bi')
    model.add(Masking(mask_value=0.0, input_shape=input_shape))
    model.add(Bidirectional(LSTM(units=45, return_sequences=True, name='lstm')))
    model.add(Conv1D(filters=32, kernel_size=10, padding='same', name='conv1d'))
    model.add(MaxPooling1D(pool_size=6, padding='same', name='max_pooling'))
    model.add(Dropout(0.5))
    model.add(Conv1D(filters=32, kernel_size=5, padding='same', name='conv1d_2'))
    model.add(MaxPooling1D(pool_size=5, padding='same', name='max_pooling_2'))
    model.add(Flatten(name='flatten'))
    model.add(Dropout(0.5))
    model.add(Dense(26, activation='relu', name='dense'))
    model.add(Dense(17, activation='relu', name='dense_2'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax', name='prediction'))
    return model

train_features = np.load(args.sessiondir + '/train_features.npy')
train_labels = np.load(args.sessiondir + '/train_labels.npy')
test_features = np.load(args.sessiondir + '/test_features.npy')
test_labels = np.load(args.sessiondir + '/test_labels.npy')

input_shape = (args.PromLen, 4)
model = model_lstm_bi_2(input_shape=input_shape)
model.build(input_shape=input_shape)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
print(model.summary())

checkpoint_filepath = args.sessiondir + '/' + args.modelName + '.h5'

checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False, monitor='val_loss', mode='min',
                             save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', patience=8, mode='min', restore_best_weights=True)

X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.1, random_state=0)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(args.batchSize)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_dataset = val_dataset.batch(args.batchSize)

history = model.fit(X_train, y_train, epochs=args.epochs, verbose=2, validation_data=(X_val, y_val),
                    shuffle=True, callbacks=[checkpoint, reduce_lr, early_stopping])



Validation_report(test_features, test_labels, args.modelName)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

model.save(args.sessiondir + '/' + args.modelName)