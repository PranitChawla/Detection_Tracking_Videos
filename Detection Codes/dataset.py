import os
import shutil
import pandas as pd 
import json

list_files = os.listdir("Ladi_Video_Annotations")

dict1 = {}

for file in list_files:
	path = "Ladi_Video_Annotations"+"/"+file
	data = pd.read_csv(path)
	list_1 = []
	# print (path)
	for index, row in data.iterrows():
		# print (row[0],row[1],path)
		list_1.append([row[0],row[1]])
		list_1.append([row[3],row[4]])
	dict1[file.split('.')[0]] = list_1
print (dict1)
json_object = json.dumps(dict1, indent = 4) 
  
# Writing to sample.json 
with open("intervals.json", "w") as outfile: 
    outfile.write(json_object)


