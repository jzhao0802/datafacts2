import os
import time
import datetime
import random
import numpy as np
from pyspark import SparkContext
from pyspark.ml.param import Params, Param
from pyspark.ml import Estimator
from pyspark.ml.util import keyword_only
from pyspark.ml.tuning import CrossValidatorModel
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import *
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder
import stratification

__all__ = ['CrossValidatorWithStratification']


class CrossValidatorWithStratification(Estimator):
    """
    K-fold cross validation.

    >>> from pyspark.ml.classification import LogisticRegression
    >>> from pyspark.ml.evaluation import BinaryClassificationEvaluator
    >>> from pyspark.mllib.linalg import Vectors
    >>> dataset = sqlContext.createDataFrame(
    ...     [(Vectors.dense([0.0]), 0.0),
    ...      (Vectors.dense([0.4]), 1.0),
    ...      (Vectors.dense([0.5]), 0.0),
    ...      (Vectors.dense([0.6]), 1.0),
    ...      (Vectors.dense([1.0]), 1.0)] * 10,
    ...     ["features", "label"])
    >>> lr = LogisticRegression()
    >>> grid = ParamGridBuilder().addGrid(lr.maxIter, [0, 1]).build()
    >>> evaluator = BinaryClassificationEvaluator()
    >>> cv = CrossValidator(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator)
    >>> cvModel = cv.fit(dataset)
    >>> evaluator.evaluate(cvModel.transform(dataset))
    0.8333...
    """

    # a placeholder to make it appear in the generated doc
    estimator = Param(Params._dummy(), "estimator", "estimator to be cross-validated")

    # a placeholder to make it appear in the generated doc
    estimatorParamMaps = Param(Params._dummy(), "estimatorParamMaps", "estimator param maps")

    # a placeholder to make it appear in the generated doc
    evaluator = Param(
        Params._dummy(), "evaluator",
        "evaluator used to select hyper-parameters that maximize the cross-validated metric")

    # a placeholder to make it appear in the generated doc
    numFolds = Param(Params._dummy(), "numFolds", "number of folds for cross validation")

    @keyword_only
    def __init__(self, estimator=None, estimatorParamMaps=None, evaluator=None, stratificationMethod=None, numFolds=3, metricValue=0.6):
        """
        __init__(self, estimator=None, estimatorParamMaps=None, evaluator=None, numFolds=3)
        """
        super(CrossValidatorWithStratification, self).__init__()
        #: param for estimator to be cross-validated
        self.estimator = Param(self, "estimator", "estimator to be cross-validated")
        #: param for estimator param maps
        self.estimatorParamMaps = Param(self, "estimatorParamMaps", "estimator param maps")
        #: param for the evaluator used to select hyper-parameters that
        #: maximize the cross-validated metric

        self.evaluator = Param(
            self, "evaluator",
            "evaluator used to select hyper-parameters that maximize the cross-validated metric")
        #: param for number of folds for cross validation
        self.numFolds = Param(self, "numFolds", "number of folds for cross validation")
        self._setDefault(numFolds=3)

        self.stratificationMethod = Param(self, "stratificationMethod", "stratification method to stratify the data")
        self._setDefault(stratificationMethod="AppendDataMatchingFoldIDs")

        self.bestIndex = None
        self.bestMetric = None
        self.allMetrics = None
        self.metricValue=Param(self, "metricValue", "metricValue")
        
        kwargs = self.__init__._input_kwargs
        self._set(**kwargs)

    @keyword_only
    def setParams(self, estimator=None, estimatorParamMaps=None, evaluator=None, stratificationMethod=None, numFolds=3, metricValue=0.6):
        """
        setParams(self, estimator=None, estimatorParamMaps=None, evaluator=None, numFolds=3):
        Sets params for cross validator.
        """
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)

    def setEstimator(self, value):
        """
        Sets the value of :py:attr:`estimator`.
        """
        self._paramMap[self.estimator] = value
        return self

    def getEstimator(self):
        """
        Gets the value of estimator or its default value.
        """
        return self.getOrDefault(self.estimator)

    def setEstimatorParamMaps(self, value):
        """
        Sets the value of :py:attr:`estimatorParamMaps`.
        """
        self._paramMap[self.estimatorParamMaps] = value
        return self

    def setStratificationMethod(self, value):
        """
        Sets the value of :py:attr:`stratificationMethod`.
        """
        self._paramMap[self.stratificationMethod] = value
        return self

    def getEstimatorParamMaps(self):
        """
        Gets the value of estimatorParamMaps or its default value.
        """
        return self.getOrDefault(self.estimatorParamMaps)

    def getStratificationMethod(self):
        """
        Gets the value of stratificationMethod or its default value.
        """
        return self.getOrDefault(self.stratificationMethod)

    def setEvaluator(self, value):
        """
        Sets the value of :py:attr:`evaluator`.
        """
        self._paramMap[self.evaluator] = value
        return self

    def getEvaluator(self):
        """
        Gets the value of evaluator or its default value.
        """
        return self.getOrDefault(self.evaluator)

    def setNumFolds(self, value):
        """
        Sets the value of :py:attr:`numFolds`.
        """
        self._paramMap[self.numFolds] = value
        return self

    def getNumFolds(self):
        """
        Gets the value of numFolds or its default value.
        """
        return self.getOrDefault(self.numFolds)
    
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

    # returns the hyperparameters of the best model chosen by the cross validator
    def getBestModelParms(self):
        epm = self.getOrDefault(self.estimatorParamMaps)

        if (self.bestIndex != None):
            bestModelParms = dict((key.name, value) for key, value in epm[self.bestIndex].iteritems())
        else:
            bestModelParms = "\nCrossvalidation has not run yet.\n"
        return bestModelParms

    def getBestMetric(self):
        if (self.bestMetric != None):
            bestMetric = self.bestMetric
        else:
            bestMetric = "\nCrossvalidation has not run yet.\n"
        return bestMetric

    def getAllMetrics(self):
        if(self.allMetrics != None):
            allMetrics = self.allMetrics
        else:
            allMetrics = "\nCrossvalidation has not run yet.\n"
        return allMetrics
            

    def _fit(self, dataset):
        est = self.getOrDefault(self.estimator)
        epm = self.getOrDefault(self.estimatorParamMaps)
        numModels = len(epm)
        eva = self.getOrDefault(self.evaluator)
        metricValue=0.0
        if(self.isSet(self.metricValue)):
            metricValue = self.getOrDefault(self.metricValue)
        evaParamDict = dict((key.name, value) for key, value in eva.__dict__['_paramMap'].iteritems())
        metricName = 'areaUnderROC'
        if('metricName' in evaParamDict):
        	metricName = evaParamDict['metricName']
        #metricValue = 0.0
        #if('metricValue' in evaParamDict):
        #	metricValue = evaParamDict['metricValue']
        nFolds = self.getOrDefault(self.numFolds)

        stratificationMethod = self.getOrDefault(self.stratificationMethod)

        idColName = "foldID"

        # Direct call for testing purpose. Do not delete
        # dataWithFoldID = AppendDataMatchingFoldIDs(data=dataset, nFolds=nFolds)

        # Code to call the method which is an attribute to cross validator class
        dataWithFoldID = getattr(stratification, stratificationMethod)(data=dataset, nFolds=nFolds)

        dataWithFoldID.cache()
        metrics = np.zeros(numModels)

        # select features, label and foldID in order
        dataWithFoldID = dataWithFoldID.select('features', 'label', 'foldID')

        for i in range(nFolds):
            condition = (dataWithFoldID[idColName] == i)
            validation = dataWithFoldID.filter(condition).select('features', 'label')
            train = dataWithFoldID.filter(~condition).select('features', 'label')
            stringIndexer = StringIndexer(inputCol="label", outputCol="indexed")
            si_model = stringIndexer.fit(train)
            train_td = si_model.transform(train)
            validation_td = si_model.transform(validation)

            for j in range(numModels):
                model = est.fit(train_td, epm[j])
                # TODO: duplicate evaluator to take extra params from input
                if(metricName == "precisionByRecall"):
                    scoresAndLabels = model.transform(validation_td, epm[j])
                    scoresAndLabels = scoresAndLabels.rdd.map(lambda x : (float(x['probability'][0]),float(x['label']))).toDF(['pred','label'])
                    metric = eva.evaluate(scoresAndLabels, {eva.metricName: metricName, eva.metricValue: metricValue})
                    metrics[j] += metric
                else:
                    metric = eva.evaluate(model.transform(validation_td, epm[j]))
                    if(metric is not None):
                        metrics[j] += metric

        if eva.isLargerBetter():
            self.bestIndex = np.argmax(metrics)
        else:
            self.bestIndex = np.argmin(metrics)
        
        self.bestMetric = metrics[self.bestIndex]
        self.allMetrics = metrics
        
        # return the best model
        self.bestModel = est.fit(dataset, epm[self.bestIndex])
        # bestModelParms = dict((key.name, value) for key, value in epm[self.bestIndex].iteritems())
        return CrossValidatorModel(self.bestModel)


def copy(self, extra=None):
    if extra is None:
        extra = dict()
    newCV = Params.copy(self, extra)
    if self.isSet(self.estimator):
        newCV.setEstimator(self.getEstimator().copy(extra))
    # estimatorParamMaps remain the same
    if self.isSet(self.evaluator):
        newCV.setEvaluator(self.getEvaluator().copy(extra))
    return newCV