import pandas as pd
import numpy as np
import datetime
import random
from impyute.imputation.cs import fast_knn
def randomtimes(stime, etime, n):
    frmt = '%d-%m-%Y %H:%M:%S'
    stime = datetime.datetime.strptime(stime, frmt)
    etime = datetime.datetime.strptime(etime, frmt)
    td = etime - stime
    return [random.random() * td + stime for _ in range(n)]

start ="26-12-2018 09:27:53"
end ="27-12-2018 09:27:53"

timeData = randomtimes(start, end, 50)

flowData = [random.random() for _ in range(50)]
pressureData = [random.random() for _ in range(50)]
# intialise data of lists.
data = {'Flow': flowData,
        'Pressure': pressureData }
        #'Time': timeData}

# Create DataFrame
df = pd.DataFrame(data)

df = df.mask(np.random.choice([True, False], size=df.shape, p=[.2,.8]))

print(df)

df = df.interpolate(method='slinear')
print(df)
