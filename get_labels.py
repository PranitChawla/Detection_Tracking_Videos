import json
import os

list_images = os.listdir("data/AVI")

with open('intervals.json', 'r') as openfile: 
  
    # Reading from json file 
    json_object = json.load(openfile) 

dictionary_images = {}

for item in json_object:
	for i in range(item[0],item[1]):
		dictionary_images[i] = 1

for i in range (len(list_images)):
	if i not in dictionary_images.keys():
		dictionary_images[i] = 0

json_object = json.dumps(dictionary_images, indent = 4) 
  
# Writing to sample.json 
with open("dataset_dict.json", "w") as outfile: 
    outfile.write(json_object)