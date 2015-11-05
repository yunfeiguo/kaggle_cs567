import matplotlib.pyplot as plt
def manyScatter(df,xcols,ycol):
    '''
    plot scatter plot
    with ycol as y axis
    each one of xcols as
    x axis
    '''
    yname = df.columns[ycol]
    for i in xcols:
	plt.figure()
	plt.scatter(df.iloc[:,i], df.iloc[:,ycol],s=1)
    	plt.ylabel(yname)
    	plt.xlabel(df.columns[i])
    	#plt.savefig(df.columns[i]+'.png')
def getMAE(prediction,reality):
    MAE=sum(abs(prediction-reality))/len(reality)
    print('MAE:'+str(MAE))
    return(MAE)
def marshall_palmer(ref, minutes_past):
    #print "Estimating rainfall from {0} observations".format(len(minutes_past))
    # how long is each observation valid?
    valid_time = np.zeros_like(minutes_past)
    valid_time[0] = minutes_past.iloc[0]
    for n in xrange(1, len(minutes_past)):
        valid_time[n] = minutes_past.iloc[n] - minutes_past.iloc[n-1]
    valid_time[-1] = valid_time[-1] + 60 - np.sum(valid_time)
    valid_time = valid_time / 60.0

    # sum up rainrate * validtime
    sum = 0
    for dbz, hours in zip(ref, valid_time):
        # See: https://en.wikipedia.org/wiki/DBZ_(meteorology)
        if np.isfinite(dbz):
            mmperhr = pow(pow(10, dbz/10)/200, 0.625)
            sum = sum + mmperhr * hours
    return sum
# each unique Id is an hour of data at some gauge
def getBenchMark(hour):
    #rowid = hour['Id'].iloc[0]
    # sort hour by minutes_past
    hour = hour.sort('minutes_past', ascending=True)
    est = marshall_palmer(hour['Ref'], hour['minutes_past'])
    return est    