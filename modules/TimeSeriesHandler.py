# -*- encoding: utf-8 -*-
# -------------------------------------------------------------------------------
'''
@Copyright :  Copyright(C) 2021, KeepSoft. All rights reserved. 

@File      :  TimeSeriesHandler.py  
@Desc      :  To accumulate or interpolate rainfall timeseries.  

Change History:
---------------------------------------------------------------------------------
version 1.0.0, Qin ZhaoYu, 2021-08-27,
Init model.
---------------------------------------------------------------------------------
version 1.0.1, Qin ZhaoYu, 2021-08-28,
Fix potential bugs in resampling timeseries methods.
---------------------------------------------------------------------------------
version 1.0.2, Qin ZhaoYu, 2021-09-02,
Provide more custom read settings; 
fix resampleByAccumulateContinue() bug.
---------------------------------------------------------------------------------
version 1.1.0, Qin ZhaoYu, 2021-09-03,
Fix zero divisior bug in resample-methods; add checkRepeated().
---------------------------------------------------------------------------------
version 1.1.1, Qin ZhaoYu, 2021-10-14,
Add timeseries sorting in reading data.
---------------------------------------------------------------------------------
'''
# -------------------------------------------------------------------------------
import datetime, os


class TimeSeriesHandler():

    @staticmethod
    def __searchTimeSeries(valArr:list, valX) ->int:
        '''to search the left boundary index of `valX` in `valArr`.

        Notice that, returning index < 0, means no element bigger than `valX`. 
        '''
        for i in range(0, len(valArr)):
            if valArr[i][0] >= valX:
                return i
        return -9999

    @staticmethod
    def readTimeSeries(tsFile, idCol=1, dateCol=2, valCol=4, fmt="%Y/%m/%d %H:%M:%S") -> map:
        '''to read input timeseries from file
        ''' 
        stationMap = {}
        ts = []
        station = ""
        with open(tsFile, "r", encoding="utf8") as reader:          
            for line in reader:
                arr = line.strip().split()
                if len(arr) < valCol: continue
                if arr[idCol - 1] != station:
                    if ts:
                        ts.sort()
                        stationMap[station] = ts
                        ts = []
                    station = arr[idCol - 1]                    
                dateStr = "{} {}".format(arr[dateCol - 1], arr[dateCol])
                currDate = datetime.datetime.strptime(dateStr, fmt)    
                ts.append([currDate, float(arr[valCol-1])])
        if ts:
            ts.sort()
            stationMap[station] = ts
        return stationMap

    @staticmethod
    def checkRepeatedElem(valArr:list):
        '''to check and average repeated datetime elements.
        '''
        newTs = [valArr[0]]
        for i in range(1, len(valArr)):
            if valArr[i][0] == valArr[i-1][0]:
                newTs[-1][1] = 0.5*(newTs[-1][1] + valArr[i][1])
                print("{} repeated".format(valArr[i]))
            else:
                newTs.append(valArr[i])
        valArr = newTs

    @staticmethod
    def resampleByInterpolateLinear(startDateStr:str, stopDateStr:str, valArr:list, step:"sec" = 3600) -> list:
        '''to resample timeseries according given step.
        '''
        newTs = []
        startDate = datetime.datetime.strptime(startDateStr, "%Y/%m/%d %H:%M:%S") 
        stopDate = datetime.datetime.strptime(stopDateStr, "%Y/%m/%d %H:%M:%S") 
        startIndex = TimeSeriesHandler.__searchTimeSeries(valArr, startDate)
        if startIndex < 0:
            print("ERROR: start datetime is beyond input timeseries.")
            return newTs
        valArr = valArr[startIndex:]

        currDate = startDate
        currYear = startDate.year
        print("resample timeseries by interpolation, current year: {} ......".format(currYear))
        while currDate <= stopDate:
            if currDate.year != currYear:
                currYear = currDate.year
                print("interpolation, current year: {} ......".format(currYear))
                
            nextValInd = TimeSeriesHandler.__searchTimeSeries(valArr, currDate)
            nextDate, nextVal = valArr[nextValInd]
            if nextValInd <= 0: # < 0 means no latter datetime; = 0 means no earlier datetime.
                newTs.append([currDate, round(nextVal, 4)])
            else:
                lastDate, lastVal = valArr[nextValInd - 1]
                deltaTime = (nextDate - lastDate).seconds
                if deltaTime <= 0:
                    ratio = 0
                else:
                    ratio = (currDate - lastDate).seconds/deltaTime
                currVal = ratio * nextVal + (1 - ratio) * lastVal
                newTs.append([currDate, round(currVal, 4)])
            currDate += datetime.timedelta(seconds=step)
        return newTs

    @staticmethod
    def resampleByAccumulateIsolate(startDateStr:str, stopDateStr:str, valArr:list, step:"sec" = 3600) -> list:
        '''to resample isolated timeseries according given step.
        '''
        newTs = []
        startDate = datetime.datetime.strptime(startDateStr, "%Y/%m/%d %H:%M:%S") 
        stopDate = datetime.datetime.strptime(stopDateStr, "%Y/%m/%d %H:%M:%S") 
        startIndex = TimeSeriesHandler.__searchTimeSeries(valArr, startDate)
        if startIndex < 0:
            print("ERROR: start datetime is beyond input timeseries.")
            return newTs
        valArr = valArr[startIndex:]

        currDate = startDate
        currIndex, lastIndex = 0, 0
        currYear = currDate.year
        print("resample isolate timeseries by accumulation, current year: {} ......".format(currYear))
        while currDate <= stopDate:  
            if currDate.year != currYear:
                currYear = currDate.year
                print("accumulation, current year: {}".format(currYear))
                
            currIndex = TimeSeriesHandler.__searchTimeSeries(valArr, currDate)
            if currIndex == 0:  # no earlier datetime.
                newTs.append([currDate, 0.0])
            elif currIndex < 0: # no latter datetime.
                newTs.append([currDate, 0.0])
            else:
                currVal = sum([elem[1] for elem in valArr[lastIndex:currIndex]])
                newTs.append([currDate, round(currVal, 4)])
            currDate += datetime.timedelta(seconds=step)
            lastIndex = currIndex
        return newTs

    @staticmethod
    def resampleByAccumulateContinue(startDateStr:str, stopDateStr:str, valArr:list, step:"sec" = 3600) -> list:
        '''to resample continues timeseries according given step.
		'''
        newTs = []
        startDate = datetime.datetime.strptime(startDateStr, "%Y/%m/%d %H:%M:%S") 
        stopDate = datetime.datetime.strptime(stopDateStr, "%Y/%m/%d %H:%M:%S") 
        startIndex = TimeSeriesHandler.__searchTimeSeries(valArr, startDate)
        if startIndex < 0:
            print("ERROR: start datetime is beyond input timeseries.")
            return newTs
        valArr = valArr[startIndex:]

        currDate, lastDate = startDate, startDate
        currIndex, lastIndex = 0, 0
        currYear = currDate.year
        print("resample continue timeseries by accumulation, current year: {} ......".format(currYear))
        while currDate <= stopDate:  
            if currDate.year != currYear:
                currYear = currDate.year
                print("accumulation, current year: {}".format(currYear))

            currIndex = TimeSeriesHandler.__searchTimeSeries(valArr, currDate)
            if currIndex == 0:  # no earlier datetime.
                newTs.append([currDate, 0.0])
            elif currIndex < 0: # no latter datetime.
                newTs.append([currDate, 0.0])
            else:
                currVal = 0.0
                currTimeDelta = (currDate - valArr[currIndex - 1][0]).seconds
                currOrigTimeDelta = (valArr[currIndex][0] - valArr[currIndex - 1][0]).seconds
                if currOrigTimeDelta <= 0:
                    currRatio = 0
                else:
                    currRatio = currTimeDelta / currOrigTimeDelta
                currVal += currRatio * valArr[currIndex][1] # split current step rainfall.
                if lastIndex > 0:
                    lastTimeDelta = (valArr[lastIndex][0] - lastDate).seconds
                    lastOrigTimeDelta = (valArr[lastIndex][0] - valArr[lastIndex - 1][0]).seconds
                    if lastOrigTimeDelta <= 0:
                        lastRatio = 0
                    else:
                        lastRatio = lastTimeDelta / lastOrigTimeDelta
                    currVal += lastRatio * valArr[lastIndex][1]  # split last step rainfall.
                if currIndex - lastIndex > 1:
                    currVal += sum([elem[1] for elem in valArr[lastIndex + 1:currIndex]]) # accmulate rainfall in between.
                newTs.append([currDate, round(currVal, 4)])
            lastDate = currDate
            lastIndex = currIndex            
            currDate += datetime.timedelta(seconds=step)
        return newTs        
   
    @staticmethod
    def output(outFile:str, valArr:list, station:str, fmt="%Y-%m-%d %H:%M:%S"):
        '''to write `valArr` to file.
        '''   
        with open(outFile, "w", encoding="utf8") as writer:
            for elem in valArr:
                dateStr = elem[0].strftime(fmt)
                dateStr = dateStr.split(".")[0]
                writer.write("{}\t{}\t".format(station, dateStr))
                for i in range(1, len(elem)):
                    writer.write("{}\t".format(elem[i]))
                writer.write("\n")



if __name__ == "__main__":
    print("----------------------------------------------------------------------")
    print("------------------------- TimeSeriesHandler --------------------------")
    print("----------------------------------------------------------------------")

    tsFile = raw_input("Please input timeseries file:")
    outFile = raw_input("Please input output file:")
    ID = raw_input("Please input ID of timeseries:")
    step = input("Please input output timestep(s):")

    if input("If ID located at column 1 [y/n]:").upper() == "Y":
        idCol = 1
    else:
        idCol = input("Please input ID column:")

    if input("If date located at column 2 [y/n]:").upper() == "Y":
        dateCol = 1
    else:
        dateCol = input("Please input date column:")

    if input("If value located at column 4 [y/n]:").upper() == "Y":
        valCol = 4
    else:
        valCol = input("Please input value column:")

    if input("If input datetime fmt is '%Y/%m/%d %H:%M:%S' [y/n]:").upper() == "Y":
        inputFmt = "%Y/%m/%d %H:%M:%S"
    else:
        inputFmt = raw_input("Please input datetime fmt:")

    if input("If output datetime fmt is '%Y/%m/%d %H:%M:%S' [y/n]:").upper() == "Y":
        outFmt = "%Y/%m/%d %H:%M:%S"
    else:
        outFmt = raw_input("Please input datetime fmt:")

    startDate = raw_input("Please input start datetime:")
    stopDate = raw_input("Please input stop datetime:")

    print("Please choice resampling mode:\n \
        1, for resampling continues timeseries by accumulating; \
        2, for resampling isolated timeseries by accumulating; \
        3, for resampling conitnues timeseries by iterpolating.")
    mode = input("Mode:")
    while (mode not in [1, 2, 3])
        mode = input("Please input valid mode: ")
	
    stationTS = TimeSeriesHandler.readTimeSeries(tsFile, idCol, dateCol, valCol, inputFmt)

    if mode == 1:
        newTs = TimeSeriesHandler.resampleByAccumulateContinue(startDate, stopDate, stationTS[ID], step)
    elif mode == 2:
        newTs = TimeSeriesHandler.resampleByAccumulateIsolate(startDate, stopDate, stationTS[ID], step)
    elif mode == 3:
        newTs = TimeSeriesHandler.resampleByInterpolateLinear(startDate, stopDate, stationTS[ID], step)

    TimeSeriesHandler.output(outFile, newTs, ID, outFmt)

