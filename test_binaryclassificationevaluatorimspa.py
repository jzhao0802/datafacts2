from abc import abstractmethod, ABCMeta

import os
import time
import datetime
import random
import numpy as np
from pyspark.ml import Estimator
from pyspark.ml.tuning import CrossValidatorModel, ParamGridBuilder
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark import since
from pyspark.ml.wrapper import JavaWrapper
from pyspark.ml.param import Param, Params
from pyspark.ml.param.shared import HasLabelCol, HasPredictionCol, HasRawPredictionCol
from pyspark.ml.util import keyword_only
from pyspark.mllib.common import inherit_doc
from pyspark.sql import DataFrame
from pyspark.sql.functions import *
from pyspark.sql import functions as F
import unittest
from pyspark.sql.types import *
from pyspark import SparkConf, SparkContext
from pyspark.sql import HiveContext
import decimal
from crossvalidator import *
from stratification import *

__all__ = ['Evaluator', 'BinaryClassificationEvaluator_IMSPA']


partition_size = 20

def precision_recall_curve(scoreAndLabels, pos_label=None,
                           sample_weight=None):
    #get the tps, fps and thresholds
    tpsFpsScorethresholds = _binary_clf_curve(scoreAndLabels,
                                             pos_label=pos_label,
                                             sample_weight=sample_weight)
    #total tps
    tpsMax = (tpsFpsScorethresholds.agg(F.max(col("tps"))).collect())[0]["max(tps)"]

    #calculate precision
    tpsFpsScorethresholds = tpsFpsScorethresholds\
        .withColumn("precision", col("tps") / ( col("tps") + col("fps") ) )

    #calculate recall
    tpsFpsScorethresholds = tpsFpsScorethresholds\
        .withColumn("recall", col("tps") / tpsMax)

    return tpsFpsScorethresholds

def _binary_clf_curve(scoreAndLabels, pos_label=None, sample_weight=None):

    #sort the dataframe by pred column in descending order
    sortedScoresAndLabels = scoreAndLabels.sort(desc("pred"))

    #creating rank for pred column
    lookup = (scoreAndLabels.select("pred")
              .distinct()
              .sort(desc("pred"))
              .rdd
              .zipWithIndex()
              .map(lambda x: x[0] + (x[1],))
              .toDF(["pred", "rank"]))

    #join the dataframe with lookup to assign the ranks
    sortedScoresAndLabels = sortedScoresAndLabels.join(lookup, ["pred"])

    #sorting in descending order based on the pred column
    sortedScoresAndLabels = sortedScoresAndLabels.sort(desc("pred"))

    #adding index to the dataframe
    sortedScoresAndLabels = sortedScoresAndLabels.rdd.zipWithIndex()\
        .toDF(['data','index'])\
        .select('data.label','data.pred','data.rank','index')

    #get existing spark context
    sc = SparkContext._active_spark_context

    #get existing HiveContext
    sqlContext = HiveContext.getOrCreate(sc)
	
    #saving the dataframe to temporary table
    sortedScoresAndLabels.registerTempTable("processeddata")

    #TODO: script to avoid partition by warning, and span data across clusters nodes
    #creating the cumulative sum for tps
    sortedScoresAndLabelsCumSum = sqlContext\
        .sql(""" SELECT label, pred, rank, index, sum(label) OVER (ORDER BY index) as tps FROM processeddata """)

    #repartitioning
    sortedScoresAndLabelsCumSum = sortedScoresAndLabelsCumSum.coalesce(partition_size)

    #cache after partitioning
    sortedScoresAndLabelsCumSum.cache()

    #droping the duplicates for thresholds
    sortedScoresAndLabelsCumSumThresholds = sortedScoresAndLabelsCumSum\
        .dropDuplicates(['pred'])

    #creating the fps column based on rank and tps column
    sortedScoresAndLabelsCumSumThresholds = sortedScoresAndLabelsCumSumThresholds\
        .withColumn("fps", 1 + col("rank") - col("tps"))

    return sortedScoresAndLabelsCumSumThresholds


def nearest_values(df, desired_value):
    #subtract all the values in the array by desired value & sort by minimum distance on top and take top 2 records.
    #i.e. get nearest neighbour
    dfWithDiff = df.withColumn("diff", F.abs(col("recall") - desired_value)).sort(asc("diff")).take(2)
    return dfWithDiff


def getPrecisionByRecall(scoreAndLabels, desired_recall):
    #get precision, recall, thresholds
    prcurve = precision_recall_curve(scoreAndLabels)

    prcurve_filtered = prcurve.where(col('recall') == desired_recall)

    #if the recall value exists then get direct precision corresponding to it
    if(prcurve_filtered.count() > 0):
        return (prcurve_filtered.take(1)[0]['precision'])

    #if the recall does not exist in the computed values, do nearest neighbour
    else:
        prcurve_nearest = nearest_values(prcurve, desired_recall)
        #prcurve_nearest[0]['index']
        #prcurve_nearest.cache()
		
        #indices = prcurve_nearest.select('index').flatMap(lambda x: x).collect()
        
        diff_value_near1 = prcurve_nearest[0]['diff']
        diff_value_near2 = prcurve_nearest[1]['diff']

        precision_near1 = prcurve_nearest[0]['precision']
        precision_near2 = prcurve_nearest[1]['precision']

        if(diff_value_near1 > diff_value_near2):
            return precision_near2

        elif(diff_value_near1 < diff_value_near2):
            return precision_near1

        elif(diff_value_near1 == diff_value_near2):
            return (precision_near1 + precision_near2) / 2.0

@inherit_doc
class Evaluator(Params):
    """
    Base class for evaluators that compute metrics from predictions.

    .. versionadded:: 1.4.0
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def _evaluate(self, dataset):
        """
        Evaluates the output.

        :param dataset: a dataset that contains labels/observations and
               predictions
        :return: metric
        """
        raise NotImplementedError()

    @since("1.4.0")
    def evaluate(self, dataset, params=None):
        """
        Evaluates the output with optional parameters.

        :param dataset: a dataset that contains labels/observations and
                        predictions
        :param params: an optional param map that overrides embedded
                       params
        :return: metric
        """
        if params is None:
            params = dict()
        else:
            metricDict = (dict((key.name, value) for key, value in params.iteritems()))
            if(metricDict['metricName'] == "precisionByRecall"):
                return getPrecisionByRecall(dataset, metricDict['metricValue'])
            else:
                if isinstance(params, dict):
                    if params:
                        return self.copy(params)._evaluate(dataset)
                    else:
                        return self._evaluate(dataset)
                else:
                    raise ValueError("Params must be a param map but got %s." % type(params))

    @since("1.5.0")
    def isLargerBetter(self):
        """
        Indicates whether the metric returned by :py:meth:`evaluate` should be maximized
        (True, default) or minimized (False).
        A given evaluator may support multiple metrics which may be maximized or minimized.
        """
        return True


@inherit_doc
class JavaEvaluator(Evaluator, JavaWrapper):
    """
    Base class for :py:class:`Evaluator`s that wrap Java/Scala
    implementations.
    """

    __metaclass__ = ABCMeta

    def _evaluate(self, dataset):
        """
        Evaluates the output.
        :param dataset: a dataset that contains labels/observations and predictions.
        :return: evaluation metric
        """
        self._transfer_params_to_java()
        return self._java_obj.evaluate(dataset._jdf)

    def isLargerBetter(self):
    	evaParamDict = dict((key.name, value) for key, value in self.__dict__['_paramMap'].iteritems())
        metricName = 'areaUnderROC'
        if('metricName' in evaParamDict):
        	metricName = evaParamDict['metricName']
        metricValue = 0.0
        if('metricValue' in evaParamDict):
        	metricValue = evaParamDict['metricValue']
        if(metricName == 'areaUnderROC' or metricName == 'areaUnderPR'):
            self._transfer_params_to_java()
            return self._java_obj.isLargerBetter()
        else:
            return True


@inherit_doc
class BinaryClassificationEvaluator_IMSPA(JavaEvaluator, HasLabelCol, HasRawPredictionCol):
    """
    Evaluator for binary classification, which expects two input
    columns: rawPrediction and label.

    >>> from pyspark.mllib.linalg import Vectors
    >>> scoreAndLabels = map(lambda x: (Vectors.dense([1.0 - x[0], x[0]]), x[1]),
    ...    [(0.1, 0.0), (0.1, 1.0), (0.4, 0.0), (0.6, 0.0), (0.6, 1.0), (0.6, 1.0), (0.8, 1.0)])
    >>> dataset = sqlContext.createDataFrame(scoreAndLabels, ["raw", "label"])
    ...
    >>> evaluator = BinaryClassificationEvaluator(rawPredictionCol="raw")
    >>> evaluator.evaluate(dataset)
    0.70...
    >>> evaluator.evaluate(dataset, {evaluator.metricName: "areaUnderPR"})
    0.83...

    .. versionadded:: 1.4.0
    """

    # a placeholder to make it appear in the generated doc
    metricName = Param(Params._dummy(), "metricName",
                       "metric name in evaluation (areaUnderROC|areaUnderPR)")
    #metricValue = Param(Params._dummy(), "metricValue", "metric recall value in getPrecisionByRecall")


    @keyword_only
    def __init__(self, rawPredictionCol="rawPrediction", labelCol="label",
                 metricName="areaUnderROC", metricValue=0.6):
        """
        __init__(self, rawPredictionCol="rawPrediction", labelCol="label", \
                 metricName="areaUnderROC")
        """
        super(BinaryClassificationEvaluator_IMSPA, self).__init__()
        self._java_obj = self._new_java_obj(
            "org.apache.spark.ml.evaluation.BinaryClassificationEvaluator", self.uid)
        #: param for metric name in evaluation (areaUnderROC|areaUnderPR)
        self.metricName = Param(self, "metricName",
                                "metric name in evaluation (areaUnderROC|areaUnderPR)")
        self.metricValue = Param(self, "metricValue", "metric recall value in getPrecisionByRecall" )
        self._setDefault(rawPredictionCol="rawPrediction", labelCol="label",
                         metricName="areaUnderROC")
        kwargs = self.__init__._input_kwargs
        self._set(**kwargs)

    @since("1.4.0")
    def setMetricName(self, value):
        """
        Sets the value of :py:attr:`metricName`.
        """
        self._paramMap[self.metricName] = value
        return self

    @since("1.4.0")
    def getMetricName(self):
        """
        Gets the value of metricName or its default value.
        """
        return self.getOrDefault(self.metricName)

    @since("1.4.0")
    def setMetricValue(self, value):
        """
        Sets the value of :py:attr:`metricValue`.
        """
        self._paramMap[self.metricValue] = value
        return self

    @since("1.4.0")
    def getMetricValue(self):
        """
        Gets the value of metricValue or its default value.
        """
        return self.getOrDefault(self.metricValue)

    @keyword_only
    @since("1.4.0")
    def setParams(self, rawPredictionCol="rawPrediction", labelCol="label",
                  metricName="areaUnderROC"):
        """
        setParams(self, rawPredictionCol="rawPrediction", labelCol="label", \
                  metricName="areaUnderROC")
        Sets params for binary classification evaluator.
        """
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)

def avg_asmb(data):
    newdata = data \
        .withColumn('avg_prob_0', lit(1) - data.pred) \
        .withColumnRenamed('pred', 'avg_prob_1')
    asmbl = VectorAssembler(inputCols=['avg_prob_0', 'avg_prob_1'],
                            outputCol="rawPrediction")
    # get the input positive and negative dataframe
    data_asmbl = asmbl.transform(newdata) \
        .select('label', 'rawPrediction')
    return data_asmbl

class PREvaluationMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nFolds = 5
        app_name = "BinaryClassificationEvaluator_IMSPA"
        sc = SparkContext(appName=app_name)
        sqlContext = HiveContext(sc)
        path = '~/Downloads/Task6__pr_evaluation_metric/toydata/labelPred.csv'
        scoreAndLabels = sqlContext.read.load(path, format='com.databricks.spark.csv', header='true',
                                              inferSchema='true')
        cls.scoreAndLabels = scoreAndLabels.select(scoreAndLabels.label.cast('double'), scoreAndLabels.pred)
        cls.scoreAndLabelsRaw = avg_asmb(cls.scoreAndLabels)

        path2 = '~/Downloads/Task6__pr_evaluation_metric/toydata/randata_100x20.csv'
        scoreAndLabels2 = sqlContext.read.load(path2, format='com.databricks.spark.csv', header='true',
                                              inferSchema='true')
        scoreAndLabelscol = scoreAndLabels2.columns
        # combine features
        assembler_scoreAndLabels = VectorAssembler(inputCols=scoreAndLabelscol[2:], outputCol="features")
        data = assembler_scoreAndLabels.transform(scoreAndLabels2) \
            .select('matched_positive_id', 'label', 'features')
        data = data.select(data.matched_positive_id, data.label.cast('double'), data.features)
        cls.dataWithFoldID = AppendDataMatchingFoldIDs(data=data, nFolds=cls.nFolds)
        cls.dataWithFoldID.cache()

    def test_is_ROC_matching(self):
        evaluator = BinaryClassificationEvaluator_IMSPA()
        ROC = evaluator.evaluate(PREvaluationMetricTests.scoreAndLabelsRaw, {evaluator.metricName: 'areaUnderROC'})
        tolerance = 0.0050
        self.assertTrue((0.8290 - tolerance) <= ROC<= (0.8290 + tolerance), "ROC value is outside of the specified range")

    def test_is_ROC_isLargeBetter_matching(self):
        evaluator = BinaryClassificationEvaluator_IMSPA()
        ROC = evaluator.evaluate(PREvaluationMetricTests.scoreAndLabelsRaw, {evaluator.metricName: 'areaUnderROC'})
        self.assertTrue(evaluator.isLargerBetter(), "method isLargerBetter() returning false.")

    def test_is_PR_matching(self):
        evaluator = BinaryClassificationEvaluator_IMSPA()
        tolerance = 0.0050
        PR = evaluator.evaluate(PREvaluationMetricTests.scoreAndLabelsRaw, {evaluator.metricName: 'areaUnderPR'})
        self.assertTrue((0.8372 - tolerance) <= PR <= (0.8372 + tolerance), "PR value is outside of the specified range")

    def test_is_PR_isLargeBetter_matching(self):
        evaluator = BinaryClassificationEvaluator_IMSPA()
        PR = evaluator.evaluate(PREvaluationMetricTests.scoreAndLabelsRaw, {evaluator.metricName: 'areaUnderPR'})
        self.assertTrue(evaluator.isLargerBetter(), "method isLargerBetter() returning false.")

    def test_is_precision_matching_1(self):
        evaluator = BinaryClassificationEvaluator_IMSPA()
        desiredRecall = decimal.Decimal('0.2')
        precision = evaluator.evaluate(PREvaluationMetricTests.scoreAndLabels,
                                       {evaluator.metricName: "precisionByRecall",
                                        evaluator.metricValue: desiredRecall})
        self.assertEqual(precision,1.0, "precisionByRecall metric producing incorrect precision: %s" % precision)

    def test_is_precision_matching_2(self):
        evaluator = BinaryClassificationEvaluator_IMSPA()
        desiredRecall = decimal.Decimal('0.4')
        precision = evaluator.evaluate(PREvaluationMetricTests.scoreAndLabels,
                                       {evaluator.metricName: "precisionByRecall",
                                        evaluator.metricValue: desiredRecall})
        self.assertEqual(precision, 0.9048, "precisionByRecall metric producing incorrect precision: %s" % precision)
        print("%s \n" % precision)

    def test_is_precision_matching_3(self):
        evaluator = BinaryClassificationEvaluator_IMSPA()
        desiredRecall = decimal.Decimal('0.6')
        precision = evaluator.evaluate(PREvaluationMetricTests.scoreAndLabels,
                                       {evaluator.metricName: "precisionByRecall",
                                        evaluator.metricValue: desiredRecall})
        self.assertEqual(precision, 0.8003, "precisionByRecall metric producing incorrect precision: %s" % precision)
        print("%s \n" % precision)

    def test_is_precision_isLargeBetter_matching(self):
        evaluator = BinaryClassificationEvaluator_IMSPA()
        desiredRecall = decimal.Decimal('0.2')
        precision = evaluator.evaluate(PREvaluationMetricTests.scoreAndLabels,
                                       {evaluator.metricName: "precisionByRecall",
                                        evaluator.metricValue: desiredRecall})
        self.assertTrue(evaluator.isLargerBetter(), "method isLargerBetter() returning false.")


    def test_is_best_metric_correct(self):
        lambdastart = 0.0001
        lambdastop = 0.001
        lambdanum = 2
        for iFold in range(PREvaluationMetricTests.nFolds):
            # stratified sampling
            ts = PREvaluationMetricTests.dataWithFoldID.filter(PREvaluationMetricTests.dataWithFoldID.foldID == iFold)
            tr = PREvaluationMetricTests.dataWithFoldID.filter(PREvaluationMetricTests.dataWithFoldID.foldID != iFold)

            # remove the fold id column
            ts = ts.drop('foldID')
            tr = tr.drop('foldID')

            # transfer to RF invalid label column
            stringIndexer = StringIndexer(inputCol="label", outputCol="indexed")
            si_model = stringIndexer.fit(tr)
            tr_td = si_model.transform(tr)
            ts_td = si_model.transform(ts)

            # Build the model
            """
            Set the ElasticNet mixing parameter.
            For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.
            For 0 < alpha < 1, the penalty is a combination of L1 and L2.
            Default is 0.0 which is an L2 penalty.
            """
            lr = LogisticRegression(featuresCol="features",
                                    labelCol="label",
                                    fitIntercept=True,
                                    elasticNetParam=0.0)

            # Create the parameter grid builder
            paramGrid = ParamGridBuilder() \
                .addGrid(lr.regParam, list(np.linspace(lambdastart, lambdastop, lambdanum))) \
                .build()

            # Create the evaluator
            evaluator = BinaryClassificationEvaluator_IMSPA(labelCol="indexed", metricName="precisionByRecall")

            # Create the cross validator
            crossval = CrossValidatorWithStratification(estimator=lr,
                                                        estimatorParamMaps=paramGrid,
                                                        evaluator=evaluator,
                                                        numFolds=PREvaluationMetricTests.nFolds,
                                                        metricValue=0.4)

            # run cross-validation and choose the best parameters
            cvModel = crossval.fit(tr_td)

            self.assertEqual(crossval.getBestMetric(), (crossval.getAllMetrics()).max(),"best metric does not correspond to the maximum.")

if __name__ == "__main__":
    unittest.main()
