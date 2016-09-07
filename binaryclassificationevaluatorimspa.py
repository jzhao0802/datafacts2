from abc import abstractmethod, ABCMeta
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.util import keyword_only
from pyspark.mllib.common import inherit_doc
from pyspark.ml.param import Param, Params
from pyspark.ml.param.shared import HasLabelCol, HasRawPredictionCol
import pyspark.sql.functions as F
# a temporary solution for SQLContext
from pyspark import SparkContext
from pyspark.sql import SQLContext
#

partition_size = 20


def precision_recall_curve(scoreAndLabels, rawPredictionCol, labelCol):
    # get the tps, fps and thresholds
    tpsFpsScorethresholds = _binary_clf_curve(scoreAndLabels,
                                              rawPredictionCol, labelCol)
    # total tps
    tpsMax = ((tpsFpsScorethresholds.agg(F.max(F.col("tps"))).collect())[0].asDict())["max(tps)"]

    # calculate precision
    tpsFpsScorethresholds = tpsFpsScorethresholds \
        .withColumn("precision", F.col("tps") / (F.col("tps") + F.col("fps")))

    # calculate recall
    tpsFpsScorethresholds = tpsFpsScorethresholds \
        .withColumn("recall", F.col("tps") / tpsMax)

    return tpsFpsScorethresholds


def _binary_clf_curve(scoreAndLabels, rawPredictionCol, labelCol):
    # sort the dataframe by pred column in descending order
    sortedScoresAndLabels = scoreAndLabels.sort(F.desc(rawPredictionCol))

    # creating rank for pred column
    lookup = (scoreAndLabels.select(rawPredictionCol)
              .distinct()
              .sort(F.desc(rawPredictionCol))
              .rdd
              .zipWithIndex()
              .map(lambda x: x[0] + (x[1],))
              .toDF([rawPredictionCol, "rank"]))

    # join the dataframe with lookup to assign the ranks
    sortedScoresAndLabels = sortedScoresAndLabels.join(lookup, [rawPredictionCol])

    # sorting in descending order based on the pred column
    sortedScoresAndLabels = sortedScoresAndLabels.sort(F.desc(rawPredictionCol))

    # adding index to the dataframe
    sortedScoresAndLabels = sortedScoresAndLabels.rdd.zipWithIndex() \
        .toDF(['data', 'index']) \
        .select('data.' + labelCol, 'data.' + rawPredictionCol, 'data.rank', 'index')

    # get existing spark context
    # sc = SparkContext._active_spark_context

    # get existing HiveContext
    # sqlContext = HiveContext.getOrCreate(sc)

    # saving the dataframe to temporary table
    sortedScoresAndLabels.registerTempTable("processeddata")

    # TODO: script to avoid partition by warning, and span data across clusters nodes
    # creating the cumulative sum for tps
    # A temporary solution for Spark 1.5.2
    sqlContext = SQLContext(SparkContext._active_spark_context)
    sortedScoresAndLabelsCumSum = sqlContext \
        .sql(
        "SELECT " + labelCol + ", " + rawPredictionCol + ", rank, index, sum(" + labelCol + ") OVER (ORDER BY index) as tps FROM processeddata ")

    # repartitioning
    sortedScoresAndLabelsCumSum = sortedScoresAndLabelsCumSum.coalesce(partition_size)

    # cache after partitioning
    sortedScoresAndLabelsCumSum.cache()

    # droping the duplicates for thresholds
    sortedScoresAndLabelsCumSumThresholds = sortedScoresAndLabelsCumSum \
        .dropDuplicates([rawPredictionCol])

    # creating the fps column based on rank and tps column
    sortedScoresAndLabelsCumSumThresholds = sortedScoresAndLabelsCumSumThresholds \
        .withColumn("fps", 1 + F.col("rank") - F.col("tps"))

    return sortedScoresAndLabelsCumSumThresholds


def nearest_values(df, desired_value):
    # subtract all the values in the array by desired value & sort by minimum distance on top and take top 2 records.
    # i.e. get nearest neighbour
    dfWithDiff = df.withColumn("diff", F.abs(F.col("recall") - desired_value)).sort(F.asc("diff")).take(2)
    return dfWithDiff


def getPrecisionByRecall(scoreAndLabels,
                         rawPredictionCol,
                         labelCol,
                         desired_recall):
    # get precision, recall, thresholds
    prcurve = precision_recall_curve(scoreAndLabels, rawPredictionCol, labelCol)

    prcurve_filtered = prcurve.where(F.col('recall') == desired_recall)

    # if the recall value exists then get direct precision corresponding to it
    if (prcurve_filtered.count() > 0):
        return ((prcurve_filtered.take(1)[0].asDict())['precision'])

    # if the recall does not exist in the computed values, do nearest neighbour
    else:
        prcurve_nearest = nearest_values(prcurve, desired_recall)
        # prcurve_nearest[0]['index']
        # prcurve_nearest.cache()

        # indices = prcurve_nearest.select('index').flatMap(lambda x: x).collect()

        diff_value_near1 = (prcurve_nearest[0].asDict())['diff']
        diff_value_near2 = (prcurve_nearest[1].asDict())['diff']

        precision_near1 = (prcurve_nearest[0].asDict())['precision']
        precision_near2 = (prcurve_nearest[1].asDict())['precision']

        if (diff_value_near1 > diff_value_near2):
            return precision_near2

        elif (diff_value_near1 < diff_value_near2):
            return precision_near1

        elif (diff_value_near1 == diff_value_near2):
            return (precision_near1 + precision_near2) / 2.0


@inherit_doc
class BinaryClassificationEvaluator_IMSPA(BinaryClassificationEvaluator, HasLabelCol, HasRawPredictionCol):
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

    # metricValue = Param(Params._dummy(), "metricValue", "metric recall value in getPrecisionByRecall")


    @keyword_only
    def __init__(self, rawPredictionCol="rawPrediction", labelCol="label",
                 metricName="areaUnderROC", metricValue=0.6):
        """
        __init__(self, rawPredictionCol="rawPrediction", labelCol="label", \
                 metricName="areaUnderROC")
        """
        super(BinaryClassificationEvaluator_IMSPA, self).__init__()
        if (metricName == "areaUnderROC") | (metricName == "areaUnderPR"):
            self._java_obj = self._new_java_obj(
                "org.apache.spark.ml.evaluation.BinaryClassificationEvaluator", self.uid)
            #: param for metric name in evaluation (areaUnderROC|areaUnderPR)
            self.metricName = Param(self, "metricName",
                                    "metric name in evaluation (areaUnderROC|areaUnderPR)")
            self._setDefault(rawPredictionCol="rawPrediction", labelCol="label",
                             metricName="areaUnderROC")
            kwargs = self.__init__._input_kwargs
            if "metricValue" in kwargs.keys():
                kwargs.pop("metricValue")

        elif (metricName == "precisionByRecall"):
            self.metricValue = Param(self, "metricValue", "metric recall value in getPrecisionByRecall")
            self.metricName = Param(self, "metricName",
                                    "metric name in evaluation (areaUnderROC|areaUnderPR)")
            self._setDefault(rawPredictionCol="rawPrediction", labelCol="label",
                             metricName="areaUnderROC", metricValue=0.6)
            kwargs = self.__init__._input_kwargs

        else:
            raise ValueError("Invalid input metricName: {}".format(self.metricNameValue))

        self._set(**kwargs)

        # for the computing precision at given recall in PySpark (in case it's only requested in calling evaluate())
        self.initMetricValueValue = metricValue
        self.initMetricNameValue = metricName
        self.rawPredictionColValue = rawPredictionCol
        self.labelColValue = labelCol

    def evaluate(self, dataset, params=None):
        if params is None:
            if (self.initMetricNameValue == "areaUnderROC") | (self.initMetricNameValue == "areaUnderPR"):
                return super(BinaryClassificationEvaluator_IMSPA, self).evaluate(dataset)
            else:
                return getPrecisionByRecall(dataset,
                                            self.rawPredictionColValue,
                                            self.labelColValue,
                                            self.initMetricValueValue)
        elif (isinstance(params, dict)):
            if ("precisionByRecall" in params.keys()):
                if "metricValue" in params.keys():
                    return getPrecisionByRecall(dataset,
                                                self.rawPredictionColValue,
                                                self.labelColValue,
                                                params["metricValue"])
                else:
                    raise ValueError("When 'precisionByRecall' is specified calling the evaluate() method, " + \
                                     "'metricValue' must also be specified")
            else:
                return super(BinaryClassificationEvaluator_IMSPA, self).evaluate(dataset, params)
        else:
            raise ValueError("Params must be a param map but got %s." % type(params))

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