import matplotlib.pyplot as plt
import pandas as pd 
import datetime
import matplotlib.dates as mdates

#see https://stackoverflow.com/questions/32972371/how-to-show-date-and-time-on-x-axis-in-matplotlib
#https://stackoverflow.com/questions/43641757/matplotlib-vlines-between-dates-on-x-axis
#https://stackoverflow.com/questions/29760821/python-matplotlib-show-tick-marks-associated-to-plotted-points
#https://matplotlib.org/3.1.0/gallery/ticks_and_spines/major_minor_demo.html
dates = ["Tue  2 Jun 16:55:51 CEST 2015",
"Wed  3 Jun 14:51:49 CEST 2015",
"Fri  5 Jun 10:31:59 CEST 2015",
"Sat  6 Jun 20:47:31 CEST 2015",
"Sun  7 Jun 13:58:23 CEST 2015",
"Mon  8 Jun 14:56:49 CEST 2015",
"Tue  9 Jun 23:39:11 CEST 2015",
"Sat 13 Jun 16:55:26 CEST 2015",
"Sun 14 Jun 15:52:34 CEST 2015",
"Sun 14 Jun 16:17:24 CEST 2015",
"Mon 15 Jun 13:23:18 CEST 2015"]

X = pd.to_datetime(dates)
print(X)
fig, ax = plt.subplots(figsize=(6,1))
fig.autofmt_xdate()

# everything after this is turning off stuff that's plotted by default
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
begin = datetime.datetime(2015, 6, 9)
end = datetime.datetime(2015, 6, 13)

ax.scatter([datetime.datetime(2015, 6, 10)], [0], zorder=3)
ax.hlines(0, begin, end, lw=10)
ax.hlines(1, X[1], datetime.datetime(2015, 6, 15), lw=10)

plt.axvspan(X[1], X[3], facecolor='#2ca02c', alpha=0.5)
ax.axvline(pd.to_datetime(begin), linewidth=2)

ax.get_yaxis().set_ticklabels(["B", "C"])
ax.get_yaxis().set_ticks([0, 1])
#ax.set_xticks(X)

#round to nearest dates
day = pd.to_timedelta("1", unit='D')
plt.xlim(X[0] - day, X[-1] + day)
values = [0,0,0,0,1,0,0,0,0,0,0]
ax.scatter(X, values, marker='s', zorder=1)

xfmt = mdates.DateFormatter('%d-%m-%y %H:%M')
ax.xaxis.set_major_formatter(xfmt)



plt.show()