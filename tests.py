from binaryclassificationevaluatorimspa import BinaryClassificationEvaluator_IMSPA
from pyspark import SparkContext
from pyspark.sql import HiveContext
import unittest
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import lit, col


def assemble_pred_vector(data, orgLabelCol, orgPositivePredictionCol, newLabelCol, newPredictionCol):
    newdata = data \
        .withColumn('prob_0', lit(1) - data[orgPositivePredictionCol]) \
        .withColumnRenamed(orgPositivePredictionCol, 'prob_1') \
        .withColumnRenamed(orgLabelCol, newLabelCol)
    asmbl = VectorAssembler(inputCols=['prob_0', 'prob_1'],
                            outputCol=newPredictionCol)
    # get the input positive and negative dataframe
    data_asmbl = asmbl.transform(newdata).select(newLabelCol, newPredictionCol)

    return data_asmbl

class PREvaluationMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sc = SparkContext(appName=cls.__name__)
        sqlContext = HiveContext(cls.sc)
        file = "s3://emr-rwes-pa-spark-dev-datastore/lichao.test/data/toy_data/task6/labelPred.csv"
        scoreAndLabels = sqlContext.read.load(file, format='com.databricks.spark.csv', header='true',
                                              inferSchema='true')
        scoreAndLabels = scoreAndLabels.select(scoreAndLabels.label.cast('double'), scoreAndLabels.pred)\
                                       .withColumnRenamed("cast(label as double)", "label")
        cls.rawPredictionCol = "pred"
        cls.labelCol = "label"
        cls.scoreAndLabelsVectorised = assemble_pred_vector(data=scoreAndLabels,
                                                            orgLabelCol="label",
                                                            orgPositivePredictionCol="pred",
                                                            newLabelCol=cls.labelCol,
                                                            newPredictionCol=cls.rawPredictionCol)
        cls.scoreAndLabelsVectorised.cache()
        cls.tolerance = 0.0050

    @classmethod
    def tearDownClass(cls):
        cls.sc.stop()
        

    def test_areaUnderROC(self):
        evaluator = BinaryClassificationEvaluator_IMSPA(rawPredictionCol=self.rawPredictionCol, labelCol=self.labelCol)
        ROC = evaluator.evaluate(PREvaluationMetricTests.scoreAndLabelsVectorised, {evaluator.metricName: 'areaUnderROC'})
        self.assertTrue((0.8290 - self.tolerance) <= ROC <= (0.8290 + self.tolerance), "Area under ROC value is incorrect.")

    def test_ROC_isLargeBetter(self):
        evaluator = BinaryClassificationEvaluator_IMSPA()
        self.assertTrue(evaluator.isLargerBetter(), "method isLargerBetter() returning False.")

    def test_areaUnderPR(self):
        evaluator = BinaryClassificationEvaluator_IMSPA(rawPredictionCol=self.rawPredictionCol, labelCol=self.labelCol)
        PR = evaluator.evaluate(PREvaluationMetricTests.scoreAndLabelsVectorised, {evaluator.metricName: 'areaUnderPR'})
        self.assertTrue((0.8372 - self.tolerance) <= PR <= (0.8372 + self.tolerance), "Area under PR value is incorrect.")

    def test_PR_isLargeBetter(self):
        evaluator = BinaryClassificationEvaluator_IMSPA()
        self.assertTrue(evaluator.isLargerBetter(), "method isLargerBetter() returning false.")

    def test_is_precision_matching_1(self):
        evaluator = BinaryClassificationEvaluator_IMSPA(rawPredictionCol=self.rawPredictionCol, labelCol=self.labelCol)
        desiredRecall = 0.2
        precision = evaluator.evaluate(PREvaluationMetricTests.scoreAndLabelsVectorised,
                                       {"metricName": "precisionByRecall",
                                        "metricValue": desiredRecall})
        self.assertEqual(round(precision, 4), 1.0, "precisionByRecall metric producing incorrect precision: %s" % precision)

    def test_is_precision_matching_2(self):
        evaluator = BinaryClassificationEvaluator_IMSPA(rawPredictionCol=self.rawPredictionCol, labelCol=self.labelCol)
        desiredRecall = 0.4
        precision = evaluator.evaluate(PREvaluationMetricTests.scoreAndLabelsVectorised,
                                       {"metricName": "precisionByRecall",
                                        "metricValue": desiredRecall})
        self.assertEqual(round(precision, 4), 0.9048, "precisionByRecall metric producing incorrect precision: %s" % precision)

    def test_is_precision_matching_3(self):
        evaluator = BinaryClassificationEvaluator_IMSPA(rawPredictionCol=self.rawPredictionCol, labelCol=self.labelCol)
        desiredRecall = 0.6
        precision = evaluator.evaluate(PREvaluationMetricTests.scoreAndLabelsVectorised,
                                       {"metricName": "precisionByRecall",
                                        "metricValue": desiredRecall})
        self.assertEqual(round(precision, 4), 0.8000, "precisionByRecall metric producing incorrect precision: %s" % precision)

    def test_is_precision_isLargeBetter_matching(self):
        evaluator = BinaryClassificationEvaluator_IMSPA(rawPredictionCol=self.rawPredictionCol, labelCol=self.labelCol)
        self.assertTrue(evaluator.isLargerBetter(), "method isLargerBetter() returning false before calculating precision at recall.")
        desiredRecall = 0.2
        precision = evaluator.evaluate(PREvaluationMetricTests.scoreAndLabelsVectorised,
                                       {"metricName": "precisionByRecall",
                                        "metricValue": desiredRecall})
        self.assertTrue(evaluator.isLargerBetter(), "method isLargerBetter() returning false after calculating precision at recall.")

    def test_precision_at_given_recall_correct_with_init(self):
        evaluator = BinaryClassificationEvaluator_IMSPA(metricName="precisionByRecall", rawPredictionCol=self.rawPredictionCol,
                                                        labelCol=self.labelCol, metricValue=0.6)
        precision = evaluator.evaluate(self.scoreAndLabelsVectorised)

        self.assertEqual(round(precision, 4), 0.8, "Incorrect precision result at the given recall using init")

    def test_precision_at_given_recall_correct_with_evaluate(self):
        evaluator = BinaryClassificationEvaluator_IMSPA(rawPredictionCol=self.rawPredictionCol, labelCol=self.labelCol)
        precision = evaluator.evaluate(PREvaluationMetricTests.scoreAndLabelsVectorised, {"metricName": "precisionByRecall", "metricValue": 0.6})

        self.assertEqual(round(precision, 4), 0.8, "Incorrect precision result at the given recall using evaluate")

    def test_is_ROC_matching(self):
        evaluator = BinaryClassificationEvaluator_IMSPA(rawPredictionCol=self.rawPredictionCol, labelCol=self.labelCol)
        ROC = evaluator.evaluate(PREvaluationMetricTests.scoreAndLabelsVectorised, {evaluator.metricName: 'areaUnderROC'})
        self.assertTrue((0.8290 - self.tolerance) <= ROC<= (0.8290 + self.tolerance), "ROC value is outside of the specified range")


if __name__ == "__main__":
    unittest.main()
