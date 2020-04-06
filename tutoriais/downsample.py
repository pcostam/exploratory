# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 18:37:45 2020

@author: anama
"""
import pandas as pd
import random
import datetime

#NOTA MUITO IMPORTANTE
#PANDAS TEM UMA COISA CHAMADA MASK PARA SERIES
#https://stackoverflow.com/questions/45694396/how-to-cast-time-columns-and-find-timedelta-with-condition-in-python-pandas
def randomdates(stime, etime, n):
    frmt = '%d-%m-%Y %H:%M:%S'
    stime = datetime.datetime.strptime(stime, frmt)
    etime = datetime.datetime.strptime(etime, frmt)
    td = etime - stime
    return [random.random() * td + stime for _ in range(n)]

def timerange(stime, n):
    actual = 0
    nowtime = stime
    res = list()
    res.append(nowtime)
    while actual < n:
        nowtime += datetime.timedelta(minutes=1)  
        res.append(nowtime)
        actual += 1
    return res
        
timetry = timerange(datetime.datetime(2016, 9, 1, 7), 50)

for dt in timetry:
    print("timetry", dt.strftime('%H:%M:%S'))
    
start ="26-12-2018 09:27:53"
end ="27-12-2018 09:27:53"

timeData = randomdates(start, end, 50)
timelist = []
for time in timeData:
    time = time.time()
    timelist.append(time)
    print("time", time)


value =[random.random() for _ in range(50)]

data = {'value': value,
        'date': timelist}

df = pd.DataFrame(data)

df['date'] = pd.to_timedelta(df['date'].astype(str))
aux_1 = df['date'].copy()
aux = df['date'].copy()
df.index = aux
df = df.resample('30min').mean()