#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import time
import datetime
import random
import numpy as np

from abc import abstractmethod, ABCMeta

from pyspark.ml.wrapper import JavaWrapper
from pyspark.ml.param import Param, Params
from pyspark.ml.param.shared import HasLabelCol, HasPredictionCol, HasRawPredictionCol
from pyspark.ml.util import keyword_only
from pyspark.mllib.common import inherit_doc


from pyspark.ml.tuning import CrossValidatorModel, ParamGridBuilder
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier

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
    #sc = SparkContext._active_spark_context

    #get existing HiveContext
    #sqlContext = HiveContext.getOrCreate(sc)
	
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
        if isinstance(params, dict):
            if params:
                return self.copy(params)._evaluate(dataset)
            else:
                return self._evaluate(dataset)
        else:
            raise ValueError("Params must be a param map but got %s." % type(params))

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
        self._transfer_params_to_java()
        return self._java_obj.isLargerBetter()


@inherit_doc
class BinaryClassificationEvaluator(JavaEvaluator, HasLabelCol, HasRawPredictionCol):
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
    """

    # a placeholder to make it appear in the generated doc
    metricName = Param(Params._dummy(), "metricName",
                       "metric name in evaluation (areaUnderROC|areaUnderPR)")

    @keyword_only
    def __init__(self, rawPredictionCol="rawPrediction", labelCol="label",
                 metricName="areaUnderROC"):
        """
        __init__(self, rawPredictionCol="rawPrediction", labelCol="label", \
                 metricName="areaUnderROC")
        """
        super(BinaryClassificationEvaluator, self).__init__()
        self._java_obj = self._new_java_obj(
            "org.apache.spark.ml.evaluation.BinaryClassificationEvaluator", self.uid)
        #: param for metric name in evaluation (areaUnderROC|areaUnderPR)
        self.metricName = Param(self, "metricName",
                                "metric name in evaluation (areaUnderROC|areaUnderPR)")
        self._setDefault(rawPredictionCol="rawPrediction", labelCol="label",
                         metricName="areaUnderROC")
        kwargs = self.__init__._input_kwargs
        self._set(**kwargs)

    def setMetricName(self, value):
        """
        Sets the value of :py:attr:`metricName`.
        """
        self._paramMap[self.metricName] = value
        return self

    def getMetricName(self):
        """
        Gets the value of metricName or its default value.
        """
        return self.getOrDefault(self.metricName)

    @keyword_only
    def setParams(self, rawPredictionCol="rawPrediction", labelCol="label",
                  metricName="areaUnderROC"):
        """
        setParams(self, rawPredictionCol="rawPrediction", labelCol="label", \
                  metricName="areaUnderROC")
        Sets params for binary classification evaluator.
        """
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)


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

    def setMetricName(self, value):
        """
        Sets the value of :py:attr:`metricName`.
        """
        self._paramMap[self.metricName] = value
        return self

    def getMetricName(self):
        """
        Gets the value of metricName or its default value.
        """
        return self.getOrDefault(self.metricName)

    def setMetricValue(self, value):
        """
        Sets the value of :py:attr:`metricValue`.
        """
        self._paramMap[self.metricValue] = value
        return self

    def getMetricValue(self):
        """
        Gets the value of metricValue or its default value.
        """
        return self.getOrDefault(self.metricValue)

    @keyword_only
    def setParams(self, rawPredictionCol="rawPrediction", labelCol="label",
                  metricName="areaUnderROC"):
        """
        setParams(self, rawPredictionCol="rawPrediction", labelCol="label", \
                  metricName="areaUnderROC")
        Sets params for binary classification evaluator.
        """
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)

def _Test1():
    evaluator = BinaryClassificationEvaluator_IMSPA(labelCol="indexed", metricName="precisionByRecall")


def _Test2():
    # input parameters
    pos_file = "dat_hae.csv"
    neg_file = "dat_nonhae.csv"
    data_file = "dat_results.csv"
    start_tree = 5
    stop_tree = 10
    num_tree = 2
    start_depth = 2
    stop_depth = 3
    num_depth = 2
    nFolds = 3

    s3_path = "s3://emr-rwes-pa-spark-dev-datastore/lichao.test/"
    data_path = "s3://emr-rwes-pa-spark-dev-datastore/lichao.test/data/HAE/data_973/"
    start_time = time.time()
    st = datetime.datetime.fromtimestamp(start_time).strftime('%Y%m%d_%H%M%S')
    resultDir_s3 = s3_path + "Results/" + st + "/"
    if not os.path.exists(resultDir_s3):
        os.makedirs(resultDir_s3)
    resultDir_master = "home/lichao.wang/code/lichao/test/Results/" + st + "/"
    if not os.path.exists(resultDir_master):
        os.makedirs(resultDir_master)

    # seed
    seed = 42
    random.seed(seed)

    # reading in the data from S3
    pos = sqlContext.read.load((data_path + pos_file),
                               format='com.databricks.spark.csv',
                               header='true',
                               inferSchema='true')

    neg = sqlContext.read.load((data_path + neg_file),
                               format='com.databricks.spark.csv',
                               header='true',
                               inferSchema='true')
    # get the column names
    pos_col = pos.columns
    neg_col = neg.columns

    # combine features
    assembler_pos = VectorAssembler(inputCols=pos_col[2:], outputCol="features")
    assembler_neg = VectorAssembler(inputCols=neg_col[2:-1], outputCol="features")

    # get the input positive and negative dataframe
    pos_asmbl = assembler_pos.transform(pos) \
        .select('PATIENT_ID', 'HAE', 'features') \
        .withColumnRenamed('PATIENT_ID', 'matched_positive_id') \
        .withColumnRenamed('HAE', 'label')

    pos_ori = pos_asmbl.withColumn('label', pos_asmbl['label'].cast('double')) \
        .select('matched_positive_id', 'label', 'features')

    neg_asmbl = assembler_neg.transform(neg) \
        .select('HAE', 'HAE_PATIENT_ID', 'features') \
        .withColumnRenamed('HAE', 'label') \
        .withColumnRenamed('HAE_PATIENT_ID', 'matched_positive_id')

    neg_ori = neg_asmbl.withColumn('label', neg_asmbl['label'].cast('double')) \
        .select('matched_positive_id', 'label', 'features')

    data = pos_ori.unionAll(neg_ori)
	
    dataWithFoldID = AppendDataMatchingFoldIDs(data=data, nFolds=nFolds)
    dataWithFoldID.cache()

    # iteration through all folds
    for iFold in range(nFolds):
        # stratified sampling
        ts = dataWithFoldID.filter(dataWithFoldID.foldID == iFold)
        tr = dataWithFoldID.filter(dataWithFoldID.foldID != iFold)

        # remove the fold id column
        ts = ts.drop('foldID')
        tr = tr.drop('foldID')

        # transfer to RF invalid label column
        stringIndexer = StringIndexer(inputCol="label", outputCol="indexed")
        si_model = stringIndexer.fit(tr)
        tr_td = si_model.transform(tr)
        ts_td = si_model.transform(ts)

        # Build the model
        rf = RandomForestClassifier(labelCol="indexed", featuresCol="features")

        # Create the parameter grid builder
        paramGrid = ParamGridBuilder() \
            .addGrid(rf.numTrees, list(np.linspace(start_tree, stop_tree,
                                                   num_tree).astype('int'))) \
            .addGrid(rf.maxDepth, list(np.linspace(start_depth, stop_depth,
                                                   num_depth).astype('int'))) \
            .build()

        # Create the evaluator
        evaluator = BinaryClassificationEvaluator_IMSPA(labelCol="indexed")
        #evaluator = BinaryClassificationEvaluator()

        # Create the cross validator
        crossval = CrossValidatorWithStratification(estimator=rf,
                                                    estimatorParamMaps=paramGrid,
                                                    evaluator=evaluator,
                                                    numFolds=nFolds)

        # run cross-validation and choose the best parameters
        cvModel = crossval.fit(tr_td)

        # Predict on training data
        prediction_tr = cvModel.transform(tr_td)
        pred_score_tr = prediction_tr.select('label', 'indexed', 'probability')

        # predict on test data
        prediction_ts = cvModel.transform(ts_td)
        pred_score_ts = prediction_ts.select('label', 'indexed', 'probability')

        # AUC
        #prediction_tr.show(truncate=False)
        AUC_tr = evaluator.evaluate(prediction_tr, {evaluator.metricName: 'areaUnderROC'})
        AUC_ts = evaluator.evaluate(prediction_ts, {evaluator.metricName: 'areaUnderROC'})

        # print out results
        # fAUC = open(resultDir_master + "AUC_fold" + str(iFold) + ".txt", "a")
        # fAUC.write("{}: Traing AUC = {} \n".format(data_file[:(len(data_file) - 4)], AUC_tr))
        # fAUC.write("{}: Test AUC = {} \n".format(data_file[:(len(data_file) - 4)], AUC_ts))
        # fAUC.close()

        pred_score_tr.coalesce(1) \
            .write.format('com.databricks.spark.csv') \
            .save(resultDir_s3 + data_file[:(len(data_file) - 4)] + "_pred_tr_fold" + str(iFold) + ".csv")
        pred_score_ts.coalesce(1) \
            .write.format('com.databricks.spark.csv') \
            .save(resultDir_s3 + data_file[:(len(data_file) - 4)] + "_pred_ts_fold" + str(iFold) + ".csv")

        # fFinished = open(resultDir_master + "finished.txt", "a")
        # fFinished.write("Test for {} finished. Please manually check the result.. \n".format(data_file))
        # fFinished.close()

if __name__ == "__main__":
    from pyspark.context import SparkContext
    from pyspark import SparkConf, SparkContext
    from pyspark.sql import HiveContext
    from pyspark.sql.types import *
    import decimal

    desiredRecall = decimal.Decimal('0.8')
    app_name = "BinaryClassificationEvaluator_IMSPA"
    sc = SparkContext(appName=app_name)
    sqlContext = HiveContext(sc)
	
    _Test1()
	
    _Test2()
