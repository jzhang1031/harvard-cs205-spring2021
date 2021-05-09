from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import FeatureHasher
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

# RF
from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=50, maxDepth=25, maxBins = 2500)

Modelrf = rf.fit(df_train)


critieoTestPredictions_rf = (Modelrf
                       .transform(df_test)
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
print("--- %s seconds ---" % (time.time() - start_time))
