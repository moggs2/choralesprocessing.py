import pandas as pd
import numpy as np
import tensorflow as tf
import glob

def preparedataset(thepath):

 totalchorales=len(glob.glob(thepath))
 newarray=np.zeros([totalchorales,3000])
 i=0
 maxlist=[]
 minlist=[]
 for file in glob.glob(thepath):
    thearray=pd.read_csv(file).to_numpy().reshape(-1)-35
    thearray=thearray.clip(min=0)
    newarray[i,:thearray.shape[0]]=thearray
    i=i+1
    minlist.append(thearray.min())
    maxlist.append(thearray.max())
 newarrayy=np.zeros([totalchorales,3000])
 newarrayy[:,:2999]=newarray[:,1:3000]

 return newarray, newarrayy, maxlist, minlist

newarray, newarrayy, maxlist, minlist = preparedataset('chorales/train/*')

newarrayvalid, newarrayyvalid, maxlistvalid, minlist = preparedataset('chorales/valid/*')

newarraytest, newarrayytest, maxlisttest, minlist = preparedataset('chorales/test/*')

print(newarrayy[4])
print(max(maxlist))
print(min(minlist))
newdataset = tf.data.Dataset.from_tensor_slices((newarray, newarrayy))
newdatasetvalid = tf.data.Dataset.from_tensor_slices((newarrayvalid, newarrayyvalid))
newdatasettest = tf.data.Dataset.from_tensor_slices((newarraytest, newarrayytest))
#newdataset = newdataset.window(100,50,True)
for window in newdataset:
    print(window[1])
newdataset = newdataset.batch(batch_size=30)
newdatasetvalid = newdatasetvalid.batch(batch_size=30)
newdatasettest = newdatasettest.batch(batch_size=30)

model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=48, output_dim=20),
        tf.keras.layers.LSTM(100, return_sequences=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LSTM(100, return_sequences=True),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(47, activation="softmax"))
        ])

model.compile(
      optimizer=tf.keras.optimizers.Adam(),
      loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
      metrics=[tf.metrics.SparseCategoricalAccuracy()]
      )

fittingdiagram=model.fit(
      newdataset,
      validation_data=newdatasetvalid,
      epochs=70,
      #callbacks=[best_checkpoint_callback, early_stopping_callback, learningratecallbackchange]
      #callbacks=[best_checkpoint_callback, early_stopping_callback]
      )

model.summary()

model.evaluate(newdatasettest)

print(newarraytest[4])

testprediction=model.predict(newarraytest)

print(np.argmax(np.array(testprediction[4][0])))
print(np.argmax(np.array(testprediction[4][1])))
print(np.argmax(np.array(testprediction[4][2])))

i=89
predictionarray=np.zeros([1,3000])
predictionarray[0][:90]=newarraytest[4][:90]
firstprediction=(np.argmax(np.array(model.predict(predictionarray))[0][89]))
print(np.array(model.predict(predictionarray)).shape)
print('First: ')
print(firstprediction)

predictionarray[0][90]=(firstprediction)

while i < (200-2):
    y=i+2
    h=i+1
    predictionarray[0][y]=(np.argmax(np.array(model.predict(predictionarray))[0][h]))
    i=i+1

print('Second: ')
print(predictionarray[0][87:107])
