from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import col
import time
start_time = time.time()

conf = SparkConf().setAppName('Mnist_Spark_MLP').setMaster('local[4]')
sc = SparkContext(conf=conf)
spark = SparkSession(sc) \
        .builder \
        .appName('WeCloud Spark Training') \
        .getOrCreate()
print('Session created')

# read in data (edit file name and header)
df_train = spark.read.parquet("part-00000-b0ec07ee-7d0a-4260-9e55-c3e38b4ff746-c000.snappy.parquet")
df_test = spark.read.parquet("part-00000-73c54270-188e-4cd1-bfd2-15d259ad444a-c000.snappy.parquet")
df_train.show()

# train LR
lr = LogisticRegression(maxIter=1000)
lrModel = lr.fit(df_train)

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
                       .transform(df_test)
                       .cache())

modelAccuracy = modelAccuracy(critieoTestPredictions_lr)
print('modelOneAccuracy for the Logistic Regression model: {0:.3f}'.format(modelAccuracy))
print("--- %s seconds ---" % (time.time() - start_time))
