from binaryclassificationevaluatorimspa import BinaryClassificationEvaluator_IMSPA
from pyspark import SparkContext
from pyspark.sql import HiveContext
import unittest


# def _Test1():
#     file = "s3://emr-rwes-pa-spark-dev-datastore/lichao.test/data/toy_data/task6/labelPred.csv"
#     scoreAndLabels = sqlContext.read.load(file, format='com.databricks.spark.csv', header='true',
#                                           inferSchema='true')
#     scoreAndLabels = scoreAndLabels.select(scoreAndLabels.label.cast('double'), scoreAndLabels.pred) \
#         .withColumnRenamed("cast(label as double)", "label")
#     evaluator = BinaryClassificationEvaluator_IMSPA(metricName="precisionByRecall", rawPredictionCol="pred",
#                                                     labelCol="label", metricValue=0.6)
#     precision = evaluator.evaluate(scoreAndLabels)
#
#     if (precision != 0.8):
#         raise ValueError("Incorrect precision result!")


# def _Test3():
#     file = "s3://emr-rwes-pa-spark-dev-datastore/lichao.test/data/toy_data/task6/labelPred.csv"
#     scoreAndLabels = sqlContext.read.load(file, format='com.databricks.spark.csv', header='true',
#                                           inferSchema='true')
#     scoreAndLabels = scoreAndLabels.select(scoreAndLabels.label.cast('double'), scoreAndLabels.pred).withColumnRenamed(
#         "cast(label as double)", "label")
#
#     evaluator = BinaryClassificationEvaluator_IMSPA(rawPredictionCol="pred", labelCol="label")
#     precision = evaluator.evaluate(scoreAndLabels, {"metricName": "precisionByRecall", "metricValue": 0.6})
#
#     if (precision != 0.8):
#         raise ValueError("Incorrect precision result!")


class PREvaluationMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nFolds = 5
        cls.sc = SparkContext(appName=cls.__name__)
        sqlContext = HiveContext(cls.sc)
        file = "s3://emr-rwes-pa-spark-dev-datastore/lichao.test/data/toy_data/task6/labelPred.csv"
        scoreAndLabels = sqlContext.read.load(file, format='com.databricks.spark.csv', header='true',
                                              inferSchema='true')
        cls.scoreAndLabels = scoreAndLabels.select(scoreAndLabels.label.cast('double'), scoreAndLabels.pred)\
                                           .withColumnRenamed("cast(label as double)", "label")
        cls.scoreAndLabels.cache()

        # path2 = '~/Downloads/Task6__pr_evaluation_metric/toydata/randata_100x20.csv'
        # scoreAndLabels2 = sqlContext.read.load(path2, format='com.databricks.spark.csv', header='true',
        #                                       inferSchema='true')
        # scoreAndLabelscol = scoreAndLabels2.columns
        # # combine features
        # assembler_scoreAndLabels = VectorAssembler(inputCols=scoreAndLabelscol[2:], outputCol="features")
        # data = assembler_scoreAndLabels.transform(scoreAndLabels2) \
        #     .select('matched_positive_id', 'label', 'features')
        # data = data.select(data.matched_positive_id, data.label.cast('double'), data.features)
        # cls.dataWithFoldID = AppendDataMatchingFoldIDs(data=data, nFolds=cls.nFolds)
        # cls.dataWithFoldID.cache()

    @classmethod
    def tearDownClass(cls):
        cls.sc.stop()

    def test_precision_at_given_recall_correct(self):
        evaluator = BinaryClassificationEvaluator_IMSPA(rawPredictionCol="pred", labelCol="label")
        precision = evaluator.evaluate(self.scoreAndLabels, {"metricName": "precisionByRecall", "metricValue": 0.6})

        if (precision != 0.8):
            raise ValueError("Incorrect precision result!")

    # def test_is_ROC_matching(self):
    #     evaluator = BinaryClassificationEvaluator_IMSPA()
    #     ROC = evaluator.evaluate(PREvaluationMetricTests.scoreAndLabelsRaw, {evaluator.metricName: 'areaUnderROC'})
    #     tolerance = 0.0050
    #     self.assertTrue((0.8290 - tolerance) <= ROC<= (0.8290 + tolerance), "ROC value is outside of the specified range")


if __name__ == "__main__":
    unittest.main()
    

# if __name__ == "__main__":
#     from pyspark.context import SparkContext
#     from pyspark import SparkConf, SparkContext
#     from pyspark.sql import HiveContext
#     from pyspark.sql.types import *
#     import decimal
#
#     print("started..")
#
#     desiredRecall = decimal.Decimal('0.8')
#     app_name = "BinaryClassificationEvaluator_IMSPA"
#     sc = SparkContext(appName=app_name)
#     sqlContext = HiveContext(sc)
#
#     _Test1()
#     _Test3()
#
#     sc.stop()
#
#     print("finished..")