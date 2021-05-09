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

conf = SparkConf().setMaster("local[4]").setAppName("preprocess")
sc = SparkContext(conf = conf)

spark = SparkSession(sc) \
        .builder \
        .appName('WeCloud Spark Training') \
        .getOrCreate()
print('Session created')


# creating my own shema
# criteoFiles = "day_0"
# criteoFiles = "subset/part-00000-bb26ef29-aa66-4261-b605-16e2dc257df1-c000.csv"


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

# criteoDF = (spark.read
#     .option("header", "false")
#     .option("delimiter", "\t")
#     .schema(criteoSchema)
#     .csv(criteoFiles)
# )

# criteoDF = (spark.read
#     .option("header", "false")
#     .option("delimiter", ",")
#     .schema(criteoSchema)
#     .csv(criteoFiles)
# )



criteoDir = "/home/ubuntu/criteo_full/"

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
    # count += 1
    # if count >= 5:
    #   break




criteoDF = criteoDF.repartition(5000)


# criteoSmallDF.printSchema()

# # take small subset
# subset = criteoSmallDF.take(1000)
# rdd = sc.parallelize(subset)
# df=rdd.toDF()

# df.groupBy('label').count().orderBy('label').show()
# criteoDF.sample(False, 0.01)

# Create balanced dataset

critieo_clicked = criteoDF.filter(col("label")==1)
critieo_unclicked = criteoDF.filter(col("label")==0)
partition = float(critieo_clicked.count()) / float(critieo_unclicked.count())
critieo_unclicked = critieo_unclicked.sample(False, partition)
criteoDF = critieo_clicked.union(critieo_unclicked)

# filter columns to be cleaned


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
# output = type(criteoDF)
# print(output)
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

#Fits a model to the input dataset with optional parameters.
# stringindex_vector  = StringIndexer(
#   inputCols=ordinal_cols, 
#   outputCols=["{}_encoded".format(c) for c in ordinal_cols]
# )

for i in range(len(categorical_cols)):
  stringindex_vector  = StringIndexer(
    inputCol=categorical_cols[i], 
    outputCol=categorical_cols_encoded[i]
  )
  criteoDF = stringindex_vector.fit(criteoDF).transform(criteoDF)

criteoDF = criteoDF.drop(*categorical_cols)

one_hot_cols_new = ["{}_encoded".format(c) for c in one_hot_cols]
one_hot_cols_encoded = ["{}_one_hot".format(c) for c in one_hot_cols_new]


for i in range(len(one_hot_cols)):
  onehotencoder_vector  = OneHotEncoder(
    inputCol=one_hot_cols_new[i], 
    outputCol=one_hot_cols_encoded[i]
  )
  # print(dir(stringindex_vector))
  # print(dir(onehotencoder_vector))
  criteoDF = onehotencoder_vector.transform(criteoDF)
  # break

criteoDF = criteoDF.drop(*one_hot_cols_new)


# print(one_hot_cols)


feature_cols = [c for c in criteoDF.columns if c != 'label']
assembler = VectorAssembler(inputCols=feature_cols,outputCol="features")
criteoDF = assembler.transform(criteoDF)

# print('total cols: {0}'.format(criteoDF.columns))
# print('feature cols: {0}'.format(feature_cols))
# criteoDF.show(5)

  

criteoDF_for_model = criteoDF.select(["label","features"])

# Splitting the data into training and testing sets
criteoTrain, criteoTest = criteoDF_for_model.randomSplit([0.7, 0.3], seed=2018)



def convert_to_numpy(df):
  vector_udf = udf(lambda vector: vector.toArray().tolist(), ArrayType(DoubleType()))
  df_temp = df.withColumn('expanded_features', vector_udf(col('features')))
  x = np.array(df_temp.select('expanded_features').rdd.map(lambda x: x[0]).collect())
  y = np.array(df_temp.select('label').rdd.map(lambda x: x).collect())
  return x, y




# X_test, y_test = convert_to_numpy(criteoTest)

# print(X_test.shape)
# print(y_test.shape)





# Save the train data
criteoTrainParquet = "/home/ubuntu/criteoSubset_train"

(criteoTrain.write                       # Our DataFrameWriter
  .option("delimiter", "\t")  
  .option("compression", "snappy")
  .mode("overwrite")                       # Replace existing files
  .parquet(criteoTrainParquet)               # Write DataFrame to parquet files
)


# Save the test data
criteoTestParquet = "/home/ubuntu/criteoSubset_test"

(criteoTest.write                       # Our DataFrameWriter
  .option("delimiter", "\t")  
  .option("compression", "snappy")
  .mode("overwrite")                       # Replace existing files
  .parquet(criteoTestParquet)               # Write DataFrame to parquet files
)



# from pyspark.ml.classification import RandomForestClassifier
# rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=50, maxDepth=25, maxBins = max_distinct)

# Modelrf = rf.fit(criteoTrain)


# critieoTestPredictions_rf = (Modelrf
#                        .transform(criteoTest)
#                        .cache())

# # model accuracy for random forest model
# # from pyspark.sql.functions import col

# def modelAccuracy(df):
#     return (df
#           .select((col('prediction') == col('label')).cast('int').alias('correct'))
#           .groupBy()
#           .avg('correct')
#           .first()[0])

# modelAccuracy = modelAccuracy(critieoTestPredictions_rf)
# print('modelOneAccuracy for the Random Forest model: {0:.3f}'.format(modelAccuracy))

