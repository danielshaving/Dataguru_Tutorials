import csv
import json
import sys
import collections
import numpy as np

datapath = 'instances_train2019.json'
jsonfile = open(datapath)
jsondic = json.load(jsonfile)

jsondic_annotations = jsondic['annotations']
jsondic_categories = jsondic['categories']
jsondic_images = jsondic['images']

annotations = []



for i, category in enumerate(jsondic_categories):
    print(category)