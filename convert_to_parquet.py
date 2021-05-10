# We follow the instructions provided by Amany in her blog:
# https://medium.com/swlh/preprocessing-criteo-dataset-for-prediction-of-click-through-rate-on-ads-7dee096a2dd9
# This file convert the original decompressed data to parquet format
# Runned with single m4.xlarge instance with attached 100 Gib EBS volume

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


criteoFiles = "day_0"

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


criteoDF = (spark.read
    .option("header", "false")
    .option("delimiter", "\t")
    .schema(criteoSchema)
    .csv(criteoFiles)
)

criteoOutParquet = "criteo_full"

(criteoDF.write                       # Our DataFrameWriter
  .option("delimiter", "\t")  
  .option("compression", "snappy")
  .mode("overwrite")                       # Replace existing files
  .parquet(criteoOutParquet)               # Write DataFrame to parquet files
)