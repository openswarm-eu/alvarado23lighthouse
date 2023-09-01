import pandas as pd
from datetime import datetime
import json
import re
import numpy as np

# Define a regular expression pattern to extract timestamp and source from log lines
log_pattern = re.compile(r'timestamp=(?P<timestamp>.*?) .*? x=(?P<x>[-\d.]+) y=(?P<y>[-\d.]+) c0=(?P<c0>\d+) c1=(?P<c1>\d+)')

# Create an empty list to store the extracted data
data = []

# Open the log file and iterate over each line
with open("./pydotbot.log", "r") as log_file:
    for line in log_file:
        # Extract timestamp and source from the line
        match = log_pattern.search(line)
        if match and "event=lh2" in line:
            # Append the extracted data to the list
            data.append({
                "timestamp": datetime.strptime(match.group("timestamp"), "%Y-%m-%dT%H:%M:%S.%fZ"),
                "x":  match.group("x"),
                "y":  match.group("y"),
                "c0": int(match.group("c0")),
                "c1": int(match.group("c1")),
            })
# Create a pandas DataFrame from the extracted data
df = pd.DataFrame(data)

# Save the files witht he times that correspond to the blender experiments

# 0_blender.csv
start = '2023-07-24T17:11:36.069403Z'
end   = '2023-07-24T17:11:43.410077Z'
df_0 = df.loc[(df['timestamp'] > datetime.strptime(start, "%Y-%m-%dT%H:%M:%S.%fZ")) & (df['timestamp'] < datetime.strptime(end, "%Y-%m-%dT%H:%M:%S.%fZ"))]
df_0.to_csv('./0_lh2.csv', index=True)

# 1_blender.csv
start = '2023-07-24T17:11:49.416083Z'
end   = '2023-07-24T17:12:09.602936Z'
df_1 = df.loc[(df['timestamp'] > datetime.strptime(start, "%Y-%m-%dT%H:%M:%S.%fZ")) & (df['timestamp'] < datetime.strptime(end, "%Y-%m-%dT%H:%M:%S.%fZ"))]
df_1.to_csv('./1_lh2.csv', index=True)

# 2_blender.csv
start = '2023-07-24T17:12:35.578912Z'
end   = '2023-07-24T17:12:48.641975Z'
df_2 = df.loc[(df['timestamp'] > datetime.strptime(start, "%Y-%m-%dT%H:%M:%S.%fZ")) & (df['timestamp'] < datetime.strptime(end, "%Y-%m-%dT%H:%M:%S.%fZ"))]
df_2.to_csv('./2_lh2.csv', index=True)

# 3_blender.csv
start = '2023-07-24T17:13:13.166500Z'
end   = '2023-07-24T17:13:35.522189Z'
df_3 = df.loc[(df['timestamp'] > datetime.strptime(start, "%Y-%m-%dT%H:%M:%S.%fZ")) & (df['timestamp'] < datetime.strptime(end, "%Y-%m-%dT%H:%M:%S.%fZ"))]
df_3.to_csv('./3_lh2.csv', index=True)

# 4_blender.csv
start = '2023-07-24T17:13:58.044711Z'
end   = '2023-07-24T17:14:13.893894Z'
df_4 = df.loc[(df['timestamp'] > datetime.strptime(start, "%Y-%m-%dT%H:%M:%S.%fZ")) & (df['timestamp'] < datetime.strptime(end, "%Y-%m-%dT%H:%M:%S.%fZ"))]
df_4.to_csv('./4_lh2.csv', index=True)

# 5_blender.csv
start = '2023-07-24T17:14:34.414414Z'
end   = '2023-07-24T17:14:55.602269Z'
df_5 = df.loc[(df['timestamp'] > datetime.strptime(start, "%Y-%m-%dT%H:%M:%S.%fZ")) & (df['timestamp'] < datetime.strptime(end, "%Y-%m-%dT%H:%M:%S.%fZ"))]
df_5.to_csv('./5_lh2.csv', index=True)

# 6_blender.csv
start = '2023-07-24T17:15:23.296630Z'
end   = '2023-07-24T17:19:15.028362Z'
df_6 = df.loc[(df['timestamp'] > datetime.strptime(start, "%Y-%m-%dT%H:%M:%S.%fZ")) & (df['timestamp'] < datetime.strptime(end, "%Y-%m-%dT%H:%M:%S.%fZ"))]
df_6.to_csv('./6_lh2.csv', index=True)
