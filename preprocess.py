# We follow the instructions provided by Amany in her blog:
# https://medium.com/swlh/preprocessing-criteo-dataset-for-prediction-of-click-through-rate-on-ads-7dee096a2dd9
# This file is the full version of the total preprocess procedure. It mainly contains 
# reading in data, balancing data with different labels, subseting categorical features,
# imputing numerical and categorical data, encoding categorical data with both ordinal encoding and 
# one-hot encoding, spliting the data into training set and testing set, 
# as well as create dataframe that can be used for Spark Mllib.
# Runned with Hadoop Cluster with 12 m4.xlarge working instances with master node storage of 100 Gib.


from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, BooleanType, FloatType
from pyspark.sql.functions import col, count, expr, when
from pyspark.ml.feature import Imputer, StringIndexer, OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import *
from pyspark.sql.functions import udf
import os


import numpy as np
import pandas as pd

conf = SparkConf().setAppName("preprocess")
sc = SparkContext(conf = conf)

spark = SparkSession(sc) \
        .builder \
        .appName('WeCloud Spark Training') \
        .getOrCreate()
print('Session created')

criteoSchema = StructType([
    StructField("label", IntegerType(), True),
    StructField("i_1", FloatType(), True),
    StructField("i_2", FloatType(), True),
    StructField("i_3", FloatType(), True),
    StructField("i_4", FloatType(), True),
    StructField("i_5", FloatType(), True),
    StructField("i_6", FloatType(), True),
    StructField("i_7", FloatType(), True),
    StructField("i_8", FloatType(), True),
    StructField("i_9", FloatType(), True),
    StructField("i_10", FloatType(), True),
    StructField("i_11", FloatType(), True),
    StructField("i_12", FloatType(), True),
    StructField("i_13", FloatType(), True),
    StructField("c_1", StringType(), True),
    StructField("c_2", StringType(), True),
    StructField("c_3", StringType(), True),
    StructField("c_4", StringType(), True),
    StructField("c_5", StringType(), True),
    StructField("c_6", StringType(), True),
    StructField("c_7", StringType(), True),
    StructField("c_8", StringType(), True), 
    StructField("c_9", StringType(), True), 
    StructField("c_10", StringType(), True), 
    StructField("c_11", StringType(), True), 
    StructField("c_12", StringType(), True), 
    StructField("c_13", StringType(), True), 
    StructField("c_14", StringType(), True), 
    StructField("c_15", StringType(), True), 
    StructField("c_16", StringType(), True), 
    StructField("c_17", StringType(), True), 
    StructField("c_18", StringType(), True), 
    StructField("c_19", StringType(), True), 
    StructField("c_20", StringType(), True), 
    StructField("c_21", StringType(), True), 
    StructField("c_22", StringType(), True), 
    StructField("c_23", StringType(), True), 
    StructField("c_24", StringType(), True), 
    StructField("c_25", StringType(), True),
    StructField("c_26", StringType(), True)
]
)


criteoDir = 'criteo_full'

criteoDF = None
count = 0

for filename in os.listdir(criteoDir):
  if filename.endswith(".parquet"):
    if not criteoDF:
      criteoDF = (spark.read              
            .option("delimiter", "\t")  
            .schema(criteoSchema)        # Use the specified schema
            .parquet(os.path.join(criteoDir, filename))   # Creates a DataFrame from Parquet after reading in the file
          )
    else:
      temp = (spark.read              
            .option("delimiter", "\t")  
            .schema(criteoSchema)        # Use the specified schema
            .parquet(os.path.join(criteoDir, filename))   # Creates a DataFrame from Parquet after reading in the file
          )
      criteoDF = criteoDF.union(temp)
    count += 1
    if count >= 5:
      break




criteoDF = criteoDF.repartition(5000)


critieo_clicked = criteoDF.filter(col("label")==1)
critieo_unclicked = criteoDF.filter(col("label")==0)
partition = float(critieo_clicked.count()) / float(critieo_unclicked.count())
critieo_unclicked = critieo_unclicked.sample(False, partition)
criteoDF = critieo_clicked.union(critieo_unclicked)

# filter columns to be cleaned

criteoDF = criteoDF.repartition(5000)

# label column has no missing data
columns = criteoDF.columns[1:]
# whether the feature contains more than 40%, 70%  missing values
columns_70 = []
columns_40 = []
columns_less_40 = []

total_count = float(criteoDF.count())

for i in columns:
    percentage = float(criteoDF.filter(col(i).isNull()).count()) / total_count
    if percentage >= 0.7:
      columns_70.append(i)
    elif percentage >= 0.4 and percentage < 0.7:
      columns_40.append(i)
    else:
      columns_less_40.append(i)

# drop columns with missing value over 70%
criteoDF = criteoDF.drop(*columns_70)

# impute columns has more than 40% missing values (change column to bolean type)
for c in columns_40:
  criteoDF = criteoDF.withColumn(c, when(col(c).isNull(), 0.0)
                .otherwise(1.0))

# impute columns has less than 40% missing values (median for integer column, mode for categorical column)
numeric_cols_impute = [c for c in columns_less_40 if c[0] == 'i']
cate_cols_impute = [c for c in columns_less_40 if c[0] != 'i']

# categorical columns: add class missing?
for c in cate_cols_impute:
  criteoDF = criteoDF.withColumn(c, when(col(c).isNull(), 'missing')
                .otherwise(criteoDF[c]))

# impute numerical columns with mean
numeric_imputer = Imputer(
    inputCols=numeric_cols_impute, 
    outputCols=["{}_imputed".format(c) for c in numeric_cols_impute],
    strategy='mean'
)

criteoDF = numeric_imputer.fit(criteoDF).transform(criteoDF)
criteoDF = criteoDF.drop(*numeric_cols_impute)

categorical_cols = ['c_'+str(i+1) for i in range(26) if ('c_'+str(i+1) not in columns_70) and ('c_'+str(i+1) not in columns_40)]
balanced_count = float(criteoDF.count())

one_hot_cols = []
max_distinct = 0
for k in categorical_cols:
    # now, let's print out the distinct value percentage
    count = criteoDF.select(k).distinct().count()
    if count <= 20:
      one_hot_cols.append(k)
    if count > max_distinct:
      max_distinct = count

categorical_cols_encoded = ["{}_encoded".format(c) for c in categorical_cols]


for i in range(len(categorical_cols)):
  stringindex_vector  = StringIndexer(
    inputCol=categorical_cols[i], 
    outputCol=categorical_cols_encoded[i]
  )
  criteoDF = stringindex_vector.setHandleInvalid("skip").fit(criteoDF).transform(criteoDF)

criteoDF = criteoDF.drop(*categorical_cols)

one_hot_cols_new = ["{}_encoded".format(c) for c in one_hot_cols]
one_hot_cols_encoded = ["{}_one_hot".format(c) for c in one_hot_cols_new]


for i in range(len(one_hot_cols)):
  onehotencoder_vector  = OneHotEncoder(
    inputCol=one_hot_cols_new[i], 
    outputCol=one_hot_cols_encoded[i]
  )
  criteoDF = onehotencoder_vector.transform(criteoDF)

criteoDF = criteoDF.drop(*one_hot_cols_new)




feature_cols = [c for c in criteoDF.columns if c != 'label']
assembler = VectorAssembler(inputCols=feature_cols,outputCol="features")
criteoDF = assembler.transform(criteoDF)
  

criteoDF_for_model = criteoDF.select(["label","features"])

# Splitting the data into training and testing sets
criteoTrain, criteoTest = criteoDF_for_model.randomSplit([0.7, 0.3], seed=2018)





# Save the train data
criteoTrainParquet = "criteo_train"

criteoTrain = criteoTrain.repartition(500)

(criteoTrain.write                       # Our DataFrameWriter
  .option("delimiter", "\t")  
  .option("compression", "snappy")
  .mode("overwrite")                       # Replace existing files
  .parquet(criteoTrainParquet)               # Write DataFrame to parquet files
)


# Save the test data
criteoTestParquet = "criteo_test"

criteoTest = criteoTest.repartition(500)

(criteoTest.write                       # Our DataFrameWriter
  .option("delimiter", "\t")  
  .option("compression", "snappy")
  .mode("overwrite")                       # Replace existing files
  .parquet(criteoTestParquet)               # Write DataFrame to parquet files
)

print('Max distinct: '.format(max_distinct))


