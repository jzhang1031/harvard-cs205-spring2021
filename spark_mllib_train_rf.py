from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import FeatureHasher
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import col
import os
import time

conf = SparkConf().setAppName('rf_train')
sc = SparkContext(conf=conf)
spark = SparkSession(sc) \
        .builder \
        .appName('WeCloud Spark Training') \
        .getOrCreate()
print('Session created')

# read in data (edit file name and header)


criteoDir_train = 'criteo_train'

criteo_train = None
count = 0

for filename in os.listdir(criteoDir_train):
  if filename.endswith(".parquet"):
    if not criteo_train:
      criteo_train = (spark.read              
            .option("delimiter", "\t")  
            .parquet(os.path.join(criteoDir_train, filename))   # Creates a DataFrame from Parquet after reading in the file
          )
    else:
      temp = (spark.read              
            .option("delimiter", "\t")  
            .parquet(os.path.join(criteoDir_train, filename))   # Creates a DataFrame from Parquet after reading in the file
          )
      criteo_train = criteo_train.union(temp)
      count += 1
      if count >= 5:
        break

criteoDir_test = 'criteo_test'

criteo_test = None
count = 0

for filename in os.listdir(criteoDir_test):
  if filename.endswith(".parquet"):
    if not criteo_test:
      criteo_test = (spark.read              
            .option("delimiter", "\t")  
            .parquet(os.path.join(criteoDir_test, filename))   # Creates a DataFrame from Parquet after reading in the file
          )
    else:
      temp = (spark.read              
            .option("delimiter", "\t")  
            .parquet(os.path.join(criteoDir_test, filename))   # Creates a DataFrame from Parquet after reading in the file
          )
      criteo_test = criteo_test.union(temp)

# RF
from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=50, maxDepth=25, maxBins = 2500)

start_time = time.time()
Modelrf = rf.fit(criteo_train)
print("--- %s seconds ---" % (time.time() - start_time))

critieoTestPredictions_rf = (Modelrf
                       .transform(criteo_test)
                       .cache())

# model accuracy for random forest model
# from pyspark.sql.functions import col

def modelAccuracy(df):
    return (df
          .select((col('prediction') == col('label')).cast('int').alias('correct'))
          .groupBy()
          .avg('correct')
          .first()[0])

modelAccuracy = modelAccuracy(critieoTestPredictions_rf)
print('modelOneAccuracy for the Random Forest model: {0:.3f}'.format(modelAccuracy))
