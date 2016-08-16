import numpy as np

def precision_recall_curve(y_true, probas_pred, pos_label=None,
                           sample_weight=None):
    fps, tps, thresholds = _binary_clf_curve(y_true, probas_pred,
                                             pos_label=pos_label,
                                             sample_weight=sample_weight)

    precision = tps / (tps + fps)
    recall = tps / tps[-1]

    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]


def _binary_clf_curve(y_true, y_score, pos_label=None, sample_weight=None):
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)

    # ensure binary classification if pos_label is not specified
    classes = np.unique(y_true)
    if (pos_label is None and
        not (np.array_equal(classes, [0, 1]) or
             np.array_equal(classes, [-1, 1]) or
             np.array_equal(classes, [0]) or
             np.array_equal(classes, [-1]) or
             np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)
    
    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        weight = sample_weight[desc_score_indices]
    else:
        weight = 1.


    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    # We need to use isclose to avoid spurious repeated thresholds
    # stemming from floating point roundoff errors.
    distinct_value_indices = np.where(np.logical_not(np.isclose(
        np.diff(y_score), 0)))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = (y_true * weight).cumsum()[threshold_idxs]
    if sample_weight is not None:
        fps = weight.cumsum()[threshold_idxs] - tps
    else:
        fps = 1 + threshold_idxs - tps
    return fps, tps, y_score[threshold_idxs]
        
        
def column_or_1d(y, warn=False):
    """ Ravel column or 1d numpy array, else raises an error
    Parameters
    ----------
    y : array-like
    warn : boolean, default False
       To control display of warnings.
    Returns
    -------
    y : array
    """
    shape = np.shape(y)
    if len(shape) == 1:
        return np.ravel(y)
    if len(shape) == 2 and shape[1] == 1:
        if warn:
            warnings.warn("A column-vector y was passed when a 1d array was"
                          " expected. Please change the shape of y to "
                          "(n_samples, ), for example using ravel().",
                          DataConversionWarning, stacklevel=2)
        return np.ravel(y)

    raise ValueError("bad input shape {0}".format(shape))

    
def nearest_values(array, value):
	#subtract all the values in the array by desired value
	diff_array = array-value
	
	#get the difference value closer to zero
	diff_value_upper, position_upper = min((b,a) for a,b in enumerate(diff_array) if b>0)
	diff_value_lower, position_lower = max((b,a) for a,b in enumerate(diff_array) if b<0)
	
	return diff_value_upper, position_upper, diff_value_lower, position_lower
	

def getPrecisionByRecall(scoreAndLabels, desired_recall):
	#get precision, recall, thresholds
    precision, recall, thresholds = precision_recall_curve(scoreAndLabels[0],scoreAndLabels[1])
    
    #desired recall is already existing then return the corresponding precision
    if(desired_recall in recall):
    	idx = recall.index(desired_recall)
    	return recall[idx]
    	
    #if the recall does not exist in the computed values, do nearest neighbour
    else:
    	diff_value_upper, position_upper, diff_value_lower, position_lower = nearest_values(recall, desired_recall)
    	
    	diff_value_lower = np.abs(diff_value_lower)
    	
    	if(diff_value_upper >  diff_value_lower):
    		return precision[position_lower]
    		
    	elif(diff_value_upper <  diff_value_lower):
    		return precision[position_lower]
    		
    	elif(diff_value_upper == diff_value_lower):
    		return (precision[position_upper] + precision[position_lower]) / 2.0


def Tests():
    import csv
    import sys

    path = '~/Downloads/Task6__pr_evaluation_metric/toydata/labelPred.csv'
    
    with open(path, 'rb') as f:
        reader = csv.reader(f)
        #skipping the header
        next(reader, None) 
        file_list = list(reader)

    scoreAndLabels = []
    labels = []
    scores = []
    for x in file_list:
	    labels.append(float(x[0]))
	    scores.append(float(x[1]))

    scoreAndLabels.append(labels)
    scoreAndLabels.append(scores)

    precision = getPrecisionByRecall(scoreAndLabels, float(sys.argv[1]))

    print("Precision to given recall is %s" % precision)	

if __name__ == "__main__":
	Tests()

