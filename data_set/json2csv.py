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

for item in jsondic_annotations:
    bbox = item['bbox']
    image_id = item['image_id']
    annotations.append([image_id,bbox[0],bbox[1],bbox[2],bbox[3],item['category_id']])

for i,image in enumerate(jsondic_images):
    #print(image)
    sys.stdout.write('\r>> Converting image %f %%'  % (i/(len(jsondic_images)+len(jsondic_categories))*100))
    sys.stdout.flush()
    for annotation in annotations :
        if annotation[0] == image['id']:
            annotation[0] = image['file_name']

for i, category in enumerate(jsondic_categories):
    #print(category)
    sys.stdout.write('\r>> Converting image %f %%' % (len(jsondic_images) + i / (len(jsondic_images) + len(jsondic_categories)) * 100))
    sys.stdout.flush()
    for annotation in annotations:
        if annotation[5] == category['id']:
            annotation[5] = category['name']
    #print(annotation)

ann_path = './annotations.csv'
with open(ann_path, 'w', newline='') as fp:
    csv_writer = csv.writer(fp, dialect='excel')
    csv_writer.writerows(annotations)



#print(annotations)
# for image in jsondic_images:
#     print(image)
# print(annotation)

#for item in jsondic_images:


# for keys,value in jsondic.items():
#     print(keys)



