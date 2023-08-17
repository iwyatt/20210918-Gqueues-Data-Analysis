# -*- coding: utf-8 -*-
"""2021-09-18 gQueues Data Analysis.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17SJOXFeBgTgnP0p60NKRPGbGG80UYehY
"""

# Data Analysis of Tasks Completed
# Task Manager that collects data is Gqueues https://gqueues.com
# Export of tasks from Gqueues is in csv format and imported into google drive
# Script reads the csv from Google Drive, searches for the portion of the csv containing the task data, then loads that data to be analyzed and charted

# Import Libraries
from pandas import pandas
import csv

# Set path to latest gqueues export csv
path = "/content/drive/MyDrive/gqueues_backup.csv"


# open csv and read
file = open(path)
reader = csv.reader(file)
lines = list(reader)

# look for task data in section of the export that contains the task items start and end rows
for line in lines:
  if "*GQ* Items" in line:
      lineNumStart = lines.index(line) +1
      break

for line in lines:
  if "*GQ* Assignments" in line:
      lineNumEnd = lines.index(line) -2
      break

# subset the file for the data we need
data = pandas.read_csv(path,skiprows = lineNumStart, skipfooter = len(lines) - lineNumEnd)

# Describe Data Frame to confirm we have proper parsing
data.info()

# Convert the data type of column 'Date' from string (YYYY/MM/DD) to datetime64
data["dateCompleted"] = pandas.to_datetime(data["dateCompleted"], format="%Y-%m-%d")
testData = data[data['dateCompleted'] >= pandas.to_datetime('2021-01-01')]['dateCompleted']

# Chart output
testData.groupby(data["dateCompleted"].dt.date).size().plot()
testData = data[data['dateCompleted'] >= pandas.to_datetime('2023-01-01')]['dateCompleted']
testData.groupby(data["dateCompleted"].dt.strftime('%W')).size().plot()

df = data.copy()
unique_tags = df["tags"].unique();
for value in unique_tags:
  df[value] = df["tags"].eq(value).astype(int)

import matplotlib.pyplot as plt
# MAKE A BAR CHART of TAGS - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.bar.html
df2 = df.copy()
df2["dateCompleted"] = df2["dateCompleted"].dt.date

df2 = df2[df2['dateCompleted'] >= pandas.to_datetime('2023-01-01')]

# optional date to week
## Convert the date string to a datetime object
df2["dateCompleted"] = pandas.to_datetime(df2["dateCompleted"])

## Get the week of year from the datetime object
df2["dateCompleted"] = df2["dateCompleted"].dt.isocalendar().week

# build aggregated data frame by date
df3 = df2.groupby(pandas.Grouper(key="dateCompleted")).sum()
df3 = df3.sort_values("dateCompleted")
df3 = df3.reset_index()

# build graph data
graph = df3[unique_tags]
graph["dateCompleted"] = df3["dateCompleted"]
graph = graph.sort_index(axis=1)

# plt bar graph with color map
plt = graph.plot.bar(x="dateCompleted",rot=45,stacked=True,figsize=(15, 10),fontsize=6, cmap=plt.cm.tab20, legend=True,xlabel="Week Number of Year", ylabel="Num Tasks Completed").legend(loc="upper left")