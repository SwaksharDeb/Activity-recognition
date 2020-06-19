import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import array
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from numpy import savetxt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import scipy.stats as stats
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam

#importing the dataset
dataset_user04 = pd.read_csv("../dataset/Training/field final/final_field_user04_dataset.csv")
dataset_user07 = pd.read_csv("../dataset/Training/field final/final_field_user07_dataset.csv")
dataset_user08 = pd.read_csv("../dataset/Training/field final/final_field_user08_dataset.csv")
dataset_user18 = pd.read_csv("../dataset/Training/field final/final_field_user18_dataset.csv")
dataset_user38 = pd.read_csv("../dataset/Training/field final/final_field_user38_dataset.csv")
dataset_user51 = pd.read_csv("../dataset/Training/field final/final_field_user51_dataset.csv")

dataset = pd.concat([dataset_user04, dataset_user07, dataset_user08, dataset_user18, dataset_user38, dataset_user51], ignore_index = True)

# balance the dataset
act_id_1 = dataset[dataset['act_id']==1].head(2028).copy()
act_id_2 = dataset[dataset['act_id']==2].head(2028).copy()
act_id_3 = dataset[dataset['act_id']==3].head(2028).copy()
act_id_4 = dataset[dataset['act_id']==4].head(2028).copy()
act_id_5 = dataset[dataset['act_id']==5].head(2028).copy()
act_id_6 = dataset[dataset['act_id']==6].head(2028).copy()
act_id_7 = dataset[dataset['act_id']==7].head(2028).copy()
act_id_8 = dataset[dataset['act_id']==8].head(2028).copy()
act_id_9 = dataset[dataset['act_id']==9].head(2028).copy()
act_id_10 = dataset[dataset['act_id']==10].head(2028).copy()
act_id_11 = dataset[dataset['act_id']==11].head(2028).copy()
act_id_12 = dataset[dataset['act_id']==12].head(2028).copy()

balanced_data = pd.DataFrame()
balanced_data = balanced_data.append([act_id_1, act_id_2, act_id_3, act_id_4, act_id_5, act_id_6, act_id_7, act_id_8, act_id_9, act_id_10, act_id_11, act_id_12])
balanced_data.reset_index(drop=True, inplace = True)
#balanced_data.shape

# Data processing
X = balanced_data.iloc[:, [0, 1, 2]].values
#X_test = balanced_data.iloc[-500:, [0, 1, 2]].values
Y = balanced_data.iloc[:, 3].values
#Y_test = balanced_data.iloc[-500:, 3].values
Y = Y - 1
#Y_test = Y_test -1

# feature scaling
sc = StandardScaler()
X_scaled = sc.fit_transform(X)
#X_test_scaled = sc.fit_transform(X_test)

#X_scaled, X_test, y_, y_test = train_test_split(X_scaled, Y, test_size = 0.1, random_state = 0, stratify = Y)

# frame preparation
Fs = 20
frame_size = Fs*4 # 80
hop_size =  1

def get_frames(df, frame_size, hop_size,label_data):

    N_FEATURES = 3

    frames = []
    labels = []
    for i in range(0, len(df) - frame_size, hop_size):
        x = df[i: i + frame_size, 0]
        y = df[i: i + frame_size, 1]
        z = df[i: i + frame_size, 2]

        # Retrieve the most often used label in this segment
        label = stats.mode(label_data[i: i + frame_size])[0][0]
        frames.append([x, y, z])
        labels.append(label)

    # Bring the segments into a better shape
    frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
    labels = np.asarray(labels)

    return frames, labels

X_, y_ = get_frames(X_scaled, frame_size, hop_size, Y)
#X_test, y_test = get_frames(X_test, frame_size, hop_size, Y_test)


#train test split
X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size = 0.1, random_state = 0, stratify = y_)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state = 0, stratify = y_train)

X_train = X_train.reshape(-1, frame_size, 3, 1)
X_val = X_val.reshape(-1, frame_size, 3, 1)
X_test = X_test.reshape(-1, frame_size, 3, 1)

# Building the model
model = Sequential()

model.add(Conv2D(16, (2, 2), activation = 'relu', padding = 'same', input_shape = X_train[0].shape))
model.add(Dropout(0.1))
model.add(Conv2D(32, (2, 2), activation='relu', padding= 'same'))
model.add(Dropout(0.2))
#model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
#model.add(Dropout(0.2))
model.add(Flatten())
#model.add(Dense(128, activation = 'relu'))
#model.add(Dropout(0.2))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(12, activation='softmax'))

# Model summary
model.summary()

# Defining the optimizer
model.compile(optimizer=Adam(learning_rate = 0.001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# Fitting the model
history = model.fit(X_train, y_train, epochs = 50, validation_data= (X_val, y_val), verbose=1)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test accuracy: {}".format(test_accuracy))

prediction = model.predict(X_test)
prediction = np.argmax(prediction, axis = 1)