from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical

from elephas.spark_model import SparkModel
from elephas.utils.rdd_utils import to_simple_rdd

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import *
from pyspark.sql.functions import udf
import numpy as np
import tensorflow as tf
import os
import time

batch_size = 64
nb_classes = 2
epochs = 1
t = 80
i = 0


conf = SparkConf().setAppName('Mnist_Spark_MLP').setMaster('local[4]')
sc = SparkContext(conf=conf)
spark = SparkSession(sc) \
        .builder \
        .appName('WeCloud Spark Training') \
        .getOrCreate()

criteoDir_train = 'criteo_train'

df_train = None
count = 0

for filename in os.listdir(criteoDir_train):
    if filename.endswith(".parquet"):
        i = i + 1
    if not df_train:
        df_train = (spark.read              
            .option("delimiter", "\t")  
            .parquet(os.path.join(criteoDir_train, filename))   
          )
    else:
        df_train = df_train.union((spark.read              
            .option("delimiter", "\t")  
            .parquet(os.path.join(criteoDir_train, filename))
          ))
    if i >= t:
        break



criteoDir_test = 'criteo_test'

df_test = None
count = 0
i = 0

for filename in os.listdir(criteoDir_test):
    if filename.endswith(".parquet"):
        i = i + 1
        if not df_test:
            df_test = (spark.read              
            .option("delimiter", "\t")  
            .parquet(os.path.join(criteoDir_test, filename))   
          )
        else:
            df_test = df_test.union((spark.read              
            .option("delimiter", "\t")  
            .parquet(os.path.join(criteoDir_test, filename)) 
          ))
    if i >= t:
        break

def convert_to_numpy(df):
  vector_udf = udf(lambda vector: vector.toArray().tolist(), ArrayType(DoubleType()))
  df_temp = df.withColumn('expanded_features', vector_udf(col('features')))
  x = np.array(df_temp.select('expanded_features').rdd.map(lambda x: x[0]).collect())
  y = np.array(df_temp.select('label').rdd.map(lambda x: x).collect())
  return x, y

x_train, y_train = convert_to_numpy(df_train)
x_test, y_test = convert_to_numpy(df_test)

y_train = to_categorical(y_train, nb_classes)
y_test = to_categorical(y_test, nb_classes)

rdd = to_simple_rdd(sc, x_train, y_train)

model = Sequential()
model = Sequential()
model.add(Dense(128, input_dim=13))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(2))
model.add(Activation('softmax'))

sgd = Adam(lr=0.01)
model.compile(sgd, 'categorical_crossentropy', [tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

spark_model = SparkModel(model, mode='synchronous')


start_time = time.time()
spark_model.fit(rdd, epochs=epochs, batch_size=batch_size, verbose=2, validation_split=0.1)
print("--- %s seconds ---" % (time.time() - start_time))

score = spark_model.evaluate(x_test, y_test, verbose=2)
print('Test accuracy:', score[1])