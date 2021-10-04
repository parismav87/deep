from tensorflow import keras
import tensorflow as tf
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, LeakyReLU, MaxPooling2D
from tensorflow.python.client import device_lib
from keras.models import model_from_json, load_model
#from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os
import json
from collections import OrderedDict as OD
from collections import Counter


print("reading json...")
jsonFile = open("labels.json", "r")
allGames = json.load(jsonFile, object_pairs_hook=OD)
print("finished reading json...")

# config = tf.compat.v1.ConfigProto(device_count = {'GPU': 1 , 'CPU': 1} ) 
# sess = tf.compat.v1.Session(config=config) 
# tf.compat.v1.keras.backend.set_session(sess)
# print(device_lib.list_local_devices())


epochs = 10
batchSize = 64
numClasses = 2


print("reading videos...")

validationX = []
validationY = []
testX = []
testY = []

nrVideos = 0

for filename in os.listdir('../output_segmented'):

	trainX = []
	trainY = []


	print(filename)


	if os.path.exists('tension.h5'):
		model = load_model('tension.h5')
		print("Loaded model from disk")

	info = filename.split("_")
	matchId = info[0]
	playerId = info[1]
	gameId = info[2]

	fullPath = '../output_segmented/' + filename
	cap = cv2.VideoCapture(fullPath)

	if playerId == "9": #validation
		frameNr = 0
		while cap.isOpened():
			ret, frame = cap.read()
			
			if ret == True:
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				
				if frameNr< len(allGames[playerId][matchId][gameId]['annotations']['tension']['annotators_amber_mild'])-30 and allGames[playerId][matchId][gameId]['annotations']['tension']['annotators_amber_mild'][frameNr+30] != -2 and allGames[playerId][matchId][gameId]['annotations']['tension']['annotators_amber_mild'][frameNr+30] != 0: #-2 means they are in disagreement, 0 means tension is not changing
					frame = np.reshape(frame,[128,128,1])
					validationX.append(frame)

					# data.append(frame)
					#labels.append(allGames[playerId][matchId][gameId]['annotations']['tension']['annotators_amber_mild'][frameNr+30]) #adding the 1 sec annotator lag (30 frames @ 30 fps)
					if allGames[playerId][matchId][gameId]['annotations']['tension']['annotators_amber_mild'][frameNr+30] == -1:
						validationY.append([1,0]) #case -1
							# pass
					else:
						validationY.append([0,1]) #case 1
				frameNr += 1
				
				# frameNr += 30 #@ 30 fps
				if cv2.waitKey(25) & 0xFF == ord('q'):
					break
			else:
				break


		cap.release()
		cv2.destroyAllWindows()

	elif playerId == "7" or playerId == "3": # test

		frameNr = 0
		while cap.isOpened():
			ret, frame = cap.read()
			
			if ret == True:
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				
				if frameNr< len(allGames[playerId][matchId][gameId]['annotations']['tension']['annotators_amber_mild'])-30 and allGames[playerId][matchId][gameId]['annotations']['tension']['annotators_amber_mild'][frameNr+30] != -2 and allGames[playerId][matchId][gameId]['annotations']['tension']['annotators_amber_mild'][frameNr+30] != 0: #-2 means they are in disagreement, 0 means tension is not changing
					frame = np.reshape(frame,[128,128,1])
					testX.append(frame)
					#labels.append(allGames[playerId][matchId][gameId]['annotations']['tension']['annotators_amber_mild'][frameNr+30]) #adding the 1 sec annotator lag (30 frames @ 30 fps)
					if allGames[playerId][matchId][gameId]['annotations']['tension']['annotators_amber_mild'][frameNr+30] == -1:
						testY.append([1,0]) #case -1
					else:
						testY.append([0,1]) #case 1

				frameNr += 1
				

				# frameNr += 30 #@ 30 fps
				if cv2.waitKey(25) & 0xFF == ord('q'):
					break
			else:
				break


		cap.release()
		cv2.destroyAllWindows()

	else:
		frameNr = 0
		while cap.isOpened():
			ret, frame = cap.read()
			
			if ret == True:
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				
				if frameNr< len(allGames[playerId][matchId][gameId]['annotations']['tension']['annotators_amber_mild'])-30 and allGames[playerId][matchId][gameId]['annotations']['tension']['annotators_amber_mild'][frameNr+30] != -2 and allGames[playerId][matchId][gameId]['annotations']['tension']['annotators_amber_mild'][frameNr+30] != 0: #-2 means they are in disagreement, 0 means tension is not changing
					frame = np.reshape(frame,[128,128,1])
					trainX.append(frame)
					#labels.append(allGames[playerId][matchId][gameId]['annotations']['tension']['annotators_amber_mild'][frameNr+30]) #adding the 1 sec annotator lag (30 frames @ 30 fps)
					if allGames[playerId][matchId][gameId]['annotations']['tension']['annotators_amber_mild'][frameNr+30] == -1:
						trainY.append([1,0]) #case -1
					else:
						trainY.append([0,1]) #case 1

				frameNr += 1
				

				# frameNr += 30 #@ 30 fps
				if cv2.waitKey(25) & 0xFF == ord('q'):
					break
			else:
				break


		cap.release()
		cv2.destroyAllWindows()

	print("finished reading video...")


	print("preparing model...")

	trainX = np.array(trainX)
	trainY = np.array(trainY)
	#print(Counter(labels))
	# print(len(trainX), len(trainY))
	#print(Counter(labels))


	#normalize pixel values
	trainX = trainX.astype('float32')
	trainX = trainX / 255.


	print("Shape of train, validation and test sets: ")
	# print(trainX.shape, trainY.shape, validationX.shape, validationY.shape, testX.shape, testY.shape)
	print(trainX.shape, trainY.shape)

	if nrVideos == 155:
		print('reached last video')
		break

	if not os.path.exists('tension.h5'):
		model = Sequential()
		model.add(Conv2D(16, kernel_size=(3, 3), activation='linear', input_shape=(128,128,1), padding='same')) #grayscale images
		# model.add(LeakyReLU(alpha=0.1))
		model.add(MaxPooling2D((2, 2), padding='same'))
		model.add(Dropout(0.30))
		model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', padding='same'))
		# model.add(LeakyReLU(alpha=0.1))
		model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
		model.add(Dropout(0.40))
		# model.add(Conv2D(16, (9, 9), activation='linear', padding='same'))
		# model.add(LeakyReLU(alpha=0.1))                  
		# model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
		# model.add(Dropout(0.40))
		model.add(Flatten())
		model.add(Dense(64, activation='relu',  kernel_regularizer=keras.regularizers.l2(0.001)))
		# model.add(LeakyReLU(alpha=0.1))
		model.add(Dropout(0.50))
		model.add(Dense(numClasses, activation='softmax',  kernel_regularizer=keras.regularizers.l2(0.001)))
		model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy']) #can also use RMSprop
		model.summary()

	if playerId!="3" and playerId!="7" and playerId!="9":
		# print(trainX.shape)
		tf.reshape(trainX, [trainX.shape[0],128,128,1])
		trainX = tf.convert_to_tensor(trainX)
		print(trainX.shape)
		
		modelTrain = model.fit(trainX, trainY, batch_size=batchSize, epochs=epochs, verbose=2) 

		model.save('tension.h5')
		nrVideos+=1


testX = np.array(testX)
testY = np.array(testY)
validationX = np.array(validationX)
validationY = np.array(validationY)
validationX = validationX.astype('float32')
testX = testX.astype('float32')
testX = testX / 255.  #0-255 to 0-1
validationX = validationX / 255.  #0-255 to 0-1

modelTrain = model.fit(trainX, trainY, validation_data=(validationX, validationY), batch_size=batchSize, epochs=epochs, verbose=2) 
model.save('tension.h5')

# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model.h5")
# print("Saved model to disk")


print(modelTrain.history.keys())
acc = modelTrain.history['accuracy']
val_acc = modelTrain.history['val_accuracy']
loss = modelTrain.history['loss']
val_loss = modelTrain.history['val_loss']
epochs = 10

print(acc)
print(val_acc)

plt.figure()
plt.plot(range(epochs), acc, 'bo', label='Training acc')
plt.plot(range(epochs), val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('validationAcc.png')


plt.figure()
plt.plot(range(epochs), loss, 'bo', label='Training loss')
plt.plot(range(epochs), val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('validationLoss.png')



results = model.evaluate(testX, testY, batch_size = batchSize, verbose=1)
print("RESULTS")
print(results)








