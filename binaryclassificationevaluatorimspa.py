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

from abc import abstractmethod, ABCMeta

from pyspark import since
from pyspark.ml.wrapper import JavaWrapper
from pyspark.ml.param import Param, Params
from pyspark.ml.param.shared import HasLabelCol, HasPredictionCol, HasRawPredictionCol
from pyspark.ml.util import keyword_only
from pyspark.mllib.common import inherit_doc

from getPrecisionByRecall_spark import *

__all__ = ['Evaluator', 'BinaryClassificationEvaluator_IMSPA']


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
        metricDict = (dict((key.name, value) for key, value in params.iteritems()))
        if params is None:
            params = dict()
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
        self._transfer_params_to_java()
        return self._java_obj.isLargerBetter()


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
    metricValue = Param(Params._dummy(), "metricValue", "metric recall value in getPrecisionByRecall")


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
                  metricName="areaUnderROC", metricValue=0.6):
        """
        setParams(self, rawPredictionCol="rawPrediction", labelCol="label", \
                  metricName="areaUnderROC")
        Sets params for binary classification evaluator.
        """
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)


if __name__ == "__main__":
    from pyspark.context import SparkContext
    from pyspark.sql import SQLContext
    from pyspark import SparkConf, SparkContext
    from pyspark.sql import SQLContext, HiveContext
    from pyspark.sql.types import *
    import decimal

    app_name = "BinaryClassificationEvaluator_IMSPA"

    sc = SparkContext(appName=app_name)
    sqlContext = HiveContext(sc)

    path = '~/Downloads/Task6__pr_evaluation_metric/toydata/labelPred.csv'
    scoreAndLabels = sqlContext.read.load(path, format='com.databricks.spark.csv', header='true', inferSchema='true')
    scoreAndLabels = scoreAndLabels.select(scoreAndLabels.label.cast('double'), scoreAndLabels.pred)

    evaluator = BinaryClassificationEvaluator_IMSPA()

    desiredRecall = decimal.Decimal('0.8')
    precision = evaluator.evaluate(scoreAndLabels,
                                   {evaluator.metricName: "precisionByRecall", evaluator.metricValue: desiredRecall})

    print("Precision to given recall is %s" % precision)
