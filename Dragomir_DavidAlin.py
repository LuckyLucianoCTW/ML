import numpy as np
from scipy.io import wavfile
from sklearn.model_selection import train_test_split 
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
files = []
test = []
np_arr_test = []
np_arr = []
train_label = []
valdiation = []
validation_label = []
file_test = []
#citirea datelor 
with open("train.txt") as f:
  for line in f:
    filename = [elt.strip() for elt in line.split(',')]
    files.append(filename)
    wav_filePath = 'train/train/' + filename[0]
    fs, data = wavfile.read(wav_filePath) 
    np_arr.append(data)
    train_label.append(int(filename[1]))
with open("test.txt") as f:
  for line in f:
    test.append(line) 
    wav_filePath = 'test/test/' + line[:len(line) - 1]
    file_test.append(line[:len(line) - 1])
    fs, data = wavfile.read(wav_filePath) 
    np_arr_test.append(data)
with open("validation.txt") as f:
  for line in f:
    filename = [elt.strip() for elt in line.split(',')]
    files.append(filename)
    wav_filePath = 'validation/validation/' + filename[0]
    fs, data = wavfile.read(wav_filePath) 
    valdiation.append(data)
    validation_label.append(int(filename[1]))
#Transformam datele din fisiere in numpy arrays
train_array_np = np.array(np_arr)
print("Train shape : ",train_array_np.shape)
test_array_np = np.array(np_arr_test)
print("Test shape : ",test_array_np.shape) 
train_label_array_np = np.array(train_label)
validation_array_np = np.array(valdiation)
validation_label_np = np.array(validation_label)
print("Validation shape : ",validation_array_np.shape)
#impartim datele de antrenare
X_train, X_test, y_train, y_test = train_test_split(train_array_np,train_label_array_np,test_size=0.2)  
class_count = 2
#Construirea Modelului   
model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=(16000, 1)))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(class_count, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#Extinderea datelor x cu inca o axa si transformarea datelor y in matrici 
y_train = keras.utils.to_categorical(y_train, num_classes=class_count) 
X_train = np.expand_dims(X_train, axis=2)
X_Validation = np.expand_dims(validation_array_np, axis=2)
y_Validation = keras.utils.to_categorical(validation_label_np, num_classes=class_count) 
model.fit(X_train, y_train, batch_size=16, epochs=40)
X_predict = np.expand_dims(test_array_np, axis=2)
#Folosim pentru a afla din datele de test cine are masca
pred = model.predict_classes(X_predict)

file = open("submission.txt", "w")
for i in range (len(file_test)):
  s = file_test[i] + "," + str(pred[i]) + "\n"
  file.write(s)
file.close() 
score, acc = model.evaluate(X_Validation, y_Validation, batch_size=16)
print("Score = ",score)
print("Acc = ", acc)