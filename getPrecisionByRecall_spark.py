from pyspark import SparkConf, SparkContext
from pyspark.sql import HiveContext, DataFrame
from pyspark.sql.functions import *
from pyspark.sql import functions as F
import decimal

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
    dfWithDiff = df.withColumn("diff", F.abs(col("recall") - desired_value)).sort(asc("diff")).limit(2)

    return dfWithDiff


def getPrecisionByRecall(scoreAndLabels, desired_recall):
    #get precision, recall, thresholds
    prcurve = precision_recall_curve(scoreAndLabels)

    prcurve_filtered = prcurve.where(col('recall') == desired_recall)

    #if the recall value exists then get direct precision corresponding to it
    if(prcurve_filtered.count() > 0):
        return (prcurve_filtered.limit(1).select('precision').head()[0])

    #if the recall does not exist in the computed values, do nearest neighbour
    else:
        prcurve_nearest = nearest_values(prcurve, desired_recall)

        indices = prcurve_nearest.select('index').flatMap(lambda x: x).collect()

        diff_value_near1 = prcurve_nearest.where(col('index') == indices[0]).head()['diff']
        diff_value_near2 = prcurve_nearest.where(col('index') == indices[1]).head()['diff']

        precision_near1 = prcurve_nearest.where(col('index') == indices[0]).head()['precision']
        precision_near2 = prcurve_nearest.where(col('index') == indices[1]).head()['precision']

        if(diff_value_near1 > diff_value_near2):
            return precision_near2

        elif(diff_value_near1 < diff_value_near2):
            return precision_near1

        elif(diff_value_near1 == diff_value_near2):
            return (precision_near1 + precision_near2) / 2.0


def Tests(sc, sqlContext):
    import sys

    path = '~/Downloads/Task6__pr_evaluation_metric/toydata/labelPred.csv'

    scoreAndLabels = sqlContext.read.load(path ,format='com.databricks.spark.csv', header='true', inferSchema='true')
    scoreAndLabels = scoreAndLabels.select(scoreAndLabels.label.cast('double'), scoreAndLabels.pred)
    precision = getPrecisionByRecall(scoreAndLabels, decimal.Decimal(sys.argv[1]))

    print("Precision to given recall is %s" % precision)	

if __name__ == "__main__":
    sc = SparkContext(appName = "Test")
    sqlContext = HiveContext(sc)
    Tests(sc, sqlContext)

