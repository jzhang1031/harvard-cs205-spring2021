# Harvard CS205 Final Project for Group 2
Contributors: Jerri Zhang, Sean Gao, Yuxin Xu, Junyi Guo

This is the Github repo for Harvard CS205 Final Project. We performed Click-Through Rate Prediction with Parallelism on AWS EC2 instances and EMR Hadoop Cluster. 

## Google site link:
https://sites.google.com/view/ctrpredictionparallelism/overview

## Project Organization
------------
    
    ├── README.md
    ├── archive
    ├── model_test
    ├── convert_file.py
    ├── convert_to_parquet.py
    ├── preprocess.py
    ├── preprocess_simplify.py
    ├── spark_mllib_train_lg.py
    ├── spark_mllib_train_rf.py
    └── test_elephas.py

## Usage
* Convert data to parquet: run convert_to_parquet.py with spark-submit
* Data preprocessing: run preprocess_simplify.py or preprocess.py with spark-submit
* Machine learning module: run spark_mllib_train_lg.py and spark_mllib_train_rf.py with spark-submit
* Deep learning module: run test_elephas.py with spark-submit

## Infrastructure
* Data preprocessing: EMR Hadoop Cluster with 12 m4.xlarge instances, master node config with 100 GiB
* Machine learning module: EMR Hadoop Cluster with m4.xlarge instance, master node config with 100 GiB
* Deep learning module: EC2 m4.xlarge instance with memory configuration w30 GiB