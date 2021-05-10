from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import col
import os
import time

conf = SparkConf().setAppName('Spark_LG')
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
      #count += 1
      #if count >= 5:
       # break

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

# train LR
lr = LogisticRegression(maxIter=1000)

start_time = time.time()
lrModel = lr.fit(criteo_train)
print("--- %s seconds ---" % (time.time() - start_time))

def modelAccuracy(df):
    return (df
          .select((col('prediction') == col('label')).cast('int').alias('correct'))
          .groupBy()
          .avg('correct')
          .first()[0])


trainingSummary = lrModel.summary

for i, rate in enumerate(trainingSummary.falsePositiveRateByLabel):
    print("label %d: %s" % (i, rate))

for i, rate in enumerate(trainingSummary.truePositiveRateByLabel):
    print("label %d: %s" % (i, rate))

for i, rec in enumerate(trainingSummary.precisionByLabel):
    print("label %d: %s" % (i, rec))

for i, rec in enumerate(trainingSummary.recallByLabel):
    print("label %d: %s" % (i, rec))

for i, f in enumerate(trainingSummary.fMeasureByLabel()):
    print("label %d: %s" % (i, f))


critieoTestPredictions_lr = (lrModel
                       .transform(criteo_test)
                       .cache())

modelAccuracy = modelAccuracy(critieoTestPredictions_lr)
print('modelOneAccuracy for the Logistic Regression model: {0:.3f}'.format(modelAccuracy))

