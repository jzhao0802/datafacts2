import numpy as np
from pyspark import SparkContext
from pyspark.sql import HiveContext
from pyspark.sql.functions import explode

__all__ = ['SplitRow','AppendDataMatchingFoldIDs']



def SplitRow(x):
    """
    Split the row into tuple with (key, rest of row values)

    rdd Row is row is given as input
    Output is a tuple of 1st column and rest of the column
    e.g. if input is 1,2,3,4,5
         output will be 1, [2,3,4,5]
    """
    return (x[0], x[1:])

def AppendDataMatchingFoldIDs(data, nFolds, nDesiredPartitions="None"):
    """
    Append a column of stratified fold IDs to the input DataFrame.

    Parameters
    ----------
    data: pyspark.sql.dataframe.DataFrame
        The input DataFrame.
    labelCol: string
        The column name of the label. Default: "label".
    nFolds: integer
        The number of foldds to stratify.
    nDesiredPartitions: integer or "None".
        The number of partitions in the returned DataFrame. If "None",
        the result has the same number of partitions as the input data.
        Default: "None".

    Returns
    ----------
    A pyspark.sql.DataFrame with a column appended to the input data
    as the fold ID. The column name of the fold ID is "foldID".
    """
    sc = SparkContext._active_spark_context
    #sqlContext = SQLContext(sc)
    sqlContext = HiveContext.getOrCreate(sc)

    inputColumns = data.columns

    if nDesiredPartitions == "None":
        nDesiredPartitions = data.rdd.getNumPartitions()

    # Group by key, where key is matched_positive_id
    data_rdd = data.rdd.map(lambda x: SplitRow(x))\
        .groupByKey()\
        .map(lambda x : (x[0], list(x[1])))


    # getting the count of positive after grouping
    nPoses = data_rdd.count()
    npFoldIDsPos = np.array(list(range(nFolds)) * np.ceil(float(nPoses) / nFolds))

    # select the actual numbers of FoldIds matching the count of positive data points
    npFoldIDs = npFoldIDsPos[:nPoses]

    # Shuffle the foldIDs to give randomness
    np.random.shuffle(npFoldIDs)

    rddFoldIDs = sc.parallelize(npFoldIDs, nDesiredPartitions).map(int)
    dfDataWithIndex = data_rdd.zipWithIndex() \
        .toDF() \
        .withColumnRenamed("_1", "orgData")
    dfNewKeyWithIndex = rddFoldIDs.zipWithIndex() \
        .toDF() \
        .withColumnRenamed("_1", "key")
    dfJoined = dfDataWithIndex.join(dfNewKeyWithIndex, "_2") \
        .select('orgData._1', 'orgData._2', 'key') \
        .withColumnRenamed('key', 'foldID') \
        .withColumnRenamed('_1','matched_positive_id') \
        .coalesce(nDesiredPartitions)

    """exploding the features and label column,
     which means grouped data of labels and features will be expanded.
     In short, grouped data by matched_positive_id will be expanded."""
    dfExpanded = dfJoined.select(dfJoined.matched_positive_id, explode(dfJoined._2).alias('data'), dfJoined.foldID)

    """expression is the columns names to be selected finally
    aliasing system generated column names such as _1 to actual column names."""
    expression=[]
    expression.append('matched_positive_id')
    for item in range(len(inputColumns) - 1):
        expression.append('data._' + str( item+1 ) + ' as ' + inputColumns[ item+1 ])
    expression.append('foldID')

    """selecting the data using the expression using 'dataframe.selectExpr' method."""
    dfWithFoldID = dfExpanded.selectExpr(expression)

    return dfWithFoldID