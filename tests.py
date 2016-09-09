from binaryclassificationevaluatorimspa import BinaryClassificationEvaluator_IMSPA


def _Test1():
    file = "s3://emr-rwes-pa-spark-dev-datastore/lichao.test/data/toy_data/task6/labelPred.csv"
    scoreAndLabels = sqlContext.read.load(file, format='com.databricks.spark.csv', header='true',
                                          inferSchema='true')
    scoreAndLabels = scoreAndLabels.select(scoreAndLabels.label.cast('double'), scoreAndLabels.pred) \
        .withColumnRenamed("cast(label as double)", "label")
    evaluator = BinaryClassificationEvaluator_IMSPA(metricName="precisionByRecall", rawPredictionCol="pred",
                                                    labelCol="label", metricValue=0.6)
    precision = evaluator.evaluate(scoreAndLabels)

    if (precision != 0.8):
        raise ValueError("Incorrect precision result!")


def _Test3():
    file = "s3://emr-rwes-pa-spark-dev-datastore/lichao.test/data/toy_data/task6/labelPred.csv"
    scoreAndLabels = sqlContext.read.load(file, format='com.databricks.spark.csv', header='true',
                                          inferSchema='true')
    scoreAndLabels = scoreAndLabels.select(scoreAndLabels.label.cast('double'), scoreAndLabels.pred).withColumnRenamed(
        "cast(label as double)", "label")

    evaluator = BinaryClassificationEvaluator_IMSPA(rawPredictionCol="pred", labelCol="label")
    precision = evaluator.evaluate(scoreAndLabels, {"metricName": "precisionByRecall", "metricValue": 0.6})

    if (precision != 0.8):
        raise ValueError("Incorrect precision result!")
    

if __name__ == "__main__":
    from pyspark.context import SparkContext
    from pyspark import SparkConf, SparkContext
    from pyspark.sql import HiveContext
    from pyspark.sql.types import *
    import decimal
    
    print("started..")

    desiredRecall = decimal.Decimal('0.8')
    app_name = "BinaryClassificationEvaluator_IMSPA"
    sc = SparkContext(appName=app_name)
    sqlContext = HiveContext(sc)

    _Test1()
    _Test3()
    
    sc.stop()
    
    print("finished..")