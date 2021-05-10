# We follow the instructions provided by Amany in her blog:
# https://medium.com/swlh/preprocessing-criteo-dataset-for-prediction-of-click-through-rate-on-ads-7dee096a2dd9
# This file provide several healp function that can be utilzed during data reading and save processes

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, BooleanType, FloatType

import numpy 
import pandas

conf = SparkConf().setMaster("local[4]").setAppName("subset")
sc = SparkContext(conf = conf)

spark = SparkSession(sc) \
        .builder \
        .appName('WeCloud Spark Training') \
        .getOrCreate()
print('Session created')

def read_parquet(path):
    df = (spark.read
    .option("delimiter", "\t")
    .parquet(path)
    )
    return df

def write_parquet(df, path):
    (df.coalesce(1).write 
    .option("delimiter", "\t")  
    .option("compression", "snappy")
    .mode("overwrite")
    .parquet(path)
    )

def convert_to_numpy(df):
  vector_udf = udf(lambda vector: vector.toArray().tolist(), ArrayType(DoubleType()))
  df_temp = df.withColumn('expanded_features', vector_udf(col('features')))
  x = np.array(df_temp.select('expanded_features').rdd.map(lambda x: x[0]).collect())
  y = np.array(df_temp.select('label').rdd.map(lambda x: x).collect())
  return x, y

# example usage
criteoPath = "/home/ubuntu/criteoSubset_train.parquet/part-00000-b0ec07ee-7d0a-4260-9e55-c3e38b4ff746-c000.snappy.parquet"

criteo_DF = (spark.read              
  .option("delimiter", "\t")  
  .parquet(criteoPath)   # Creates a DataFrame from Parquet after reading in the file
)

(criteoDF.write                       # Our DataFrameWriter
.option("delimiter", "\t")  
.option("compression", "snappy")
.mode("overwrite")                       # Replace existing files
.parquet(path)               # Write DataFrame to parquet files
)