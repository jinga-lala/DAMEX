import json
import os
from collections import defaultdict

# import torchvision.datasets as dset
from .voc import VOCDetection
from PIL import Image

# Load the VOC dataset
dataset_name="dota"
data_root = '/coc/pskynet4/yashjain/data/uodb/'  # Change this path to where you stored the VOC dataset
voc_root = f'/coc/pskynet4/yashjain/data/uodb/{dataset_name}/'  # Change this path to where you stored the VOC dataset
image_set = 'test' # Change this to 'train' if you want to convert the training set
train_img_id = 19219 #0 
train_annotation_id = 236236 #1 
voc_dataset = VOCDetection(voc_root, year='2023', image_set=image_set, download=False)

# Initialize COCO dataset format dictionaries
coco_dataset = {
    "info": {
        "year": 2019,
        "version": "1.0",
        "description": f"{dataset_name} to COCO format",
    },
    "images": [],
    "annotations": [],
    "categories": [],
}

voc07_CLASSES = ('__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

voc_mapping = {'1': 5, '2': 2, '3': 16, '4': 9, '5': 44, '6': 6, '7': 3, '8': 17, '9': 62, '10': 21, '11': 67, '12': 18, '13': 19, '14': 4, '15': 1, '16': 64, '17': 20, '18': 63, '19': 7, '20': 72}


voc12_CLASSES = ('__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

clipart_CLASSES = ('__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                    'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                    'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                    'tvmonitor')

watercolor_CLASSES = ('__background__', 'bicycle', 'bird', 'car', 'cat', 'dog', 'person')
watercolor_mapping = {'1': 2, '2': 16, '3': 3, '4': 17, '5': 18, '6': 1}

comic_CLASSES = ('__background__', 'bicycle', 'bird', 'car', 'cat', 'dog', 'person')
comic_mapping = {'1': 2, '2': 16, '3': 3, '4': 17, '5': 18, '6': 1}

deeplesion_CLASSES = ('__background__', 'lesion')
deeplesion_mapping = {'1':91}

dota_CLASSES = ('__background__', 'plane', 'baseball-diamond', 'bridge', 'ground-track-field', 
                'small-vehicle', 'large-vehicle', 'ship', 'tennis-court', 'basketball-court', 
                'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter')

dota_mapping = {0 : 0, 1: 92, 2: 93, 3: 94, 4: 95, 5: 96, 6: 97, 7: 98, 8: 99, 9: 100, 10: 101, 11: 102, 12: 103, 13: 104, 14: 105, 15: 106}

kitchen_CLASSES = ('__background__', 'coca_cola_glass_bottle', 'coffee_mate_french_vanilla', 
                   'honey_bunches_of_oats_honey_roasted', 'hunts_sauce', 'mahatma_rice',
                   'nature_valley_soft_baked_oatmeal_squares_cinnamon_brown_sugar', 
                   'nature_valley_sweet_and_salty_nut_almond', 'palmolive_orange',
                   'pop_secret_light_butter', 'pringles_bbq', 'red_bull')

kitchen_mapping = {0:0, 1:107, 2:108, 3:109, 4:110, 5:111, 6:112, 7:113, 8:114, 9:115, 10:116, 11:117}


kitti_CLASSES = ('__background__', 'pedestrian', 'cyclist', 'vehicle')
kitti_mapping = {0: 0, 1:118, 2:119, 3:120 }

lisa_CLASSES = ('__background__', 'stop', 'speedlimit', 'warning', 'noturn')
lisa_mapping = {0:0, 1:121, 2:122, 3:123, 4:124}

widerface_CLASSES = ('__background__', 'face')
widerface_mapping = {0:0, 1:125}

category_map = defaultdict(lambda: len(category_map))
category_map = {eval(f'{dataset_name}_CLASSES')[i]: i for i in range(len(eval(f'{dataset_name}_CLASSES')))}

annotation_id = train_annotation_id # Change this to 1 if you want to convert the training set

# Iterate through the VOC dataset and convert to COCO format
for image_id, (image, target) in enumerate(voc_dataset, start=1):
    # Extract image information
    image_id = image_id+train_img_id # Change this to 1 if you want to convert the training set
    if dataset_name =='kitchen':
        file_name = os.path.basename(target["annotation"]["folder"]) + '_' + os.path.basename(target["annotation"]["filename"])
    else:
        file_name = os.path.basename(target["annotation"]["filename"])
    if not file_name.endswith(".jpg") and not file_name.endswith(".png"):
        file_name = file_name + ".jpg"

    if not os.path.exists(os.path.join(voc_root, "JPEGImages", file_name)):
        print(file_name)
        continue

    image_size = Image.open(os.path.join(voc_root, "JPEGImages", file_name)).size

    # Add image entry to COCO dataset
    coco_dataset["images"].append({
        "id": image_id,
        "file_name": file_name,
        "width": image_size[0],
        "height": image_size[1],
    })

    # Extract annotations information
    for obj in target["annotation"]["object"]:
        bbox = obj["bndbox"]
        category_name = obj["name"]

        if dataset_name == 'kitti':
            if category_name == 'car':
                category_name = 'vehicle'
            elif category_name == 'van':
                category_name = 'vehicle'
            elif category_name == 'truck':
                category_name = 'vehicle'
            elif category_name == 'tricycle':
                category_name = 'vehicle'
            elif category_name == 'person_sitting':
                category_name = 'pedestrian'
            # elif category_name == 'misc':
                # category_name = 'vehicle'
        if category_name == 'dontcare':
            continue
        # Map category name to an integer ID
        category_id = category_map[category_name]

        # Calculate bounding box [x, y, width, height]
        x = float(bbox["xmin"])
        y = float(bbox["ymin"])
        width = float(bbox["xmax"]) - x
        height = float(bbox["ymax"]) - y

        # Add annotation entry to COCO dataset
        coco_dataset["annotations"].append({
            # "keypoints": 0,
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [x, y, width, height],
            "area": width * height,
            "iscrowd": 0,
        })

        annotation_id += 1
# Create COCO categories
coco_dataset["categories"] = [{"id": idx, "name": name} for name, idx in category_map.items()]

print(image_id, annotation_id)
# Save COCO dataset to a JSON file
coco_output_file = f'/coc/pskynet4/yashjain/data/uodb/{dataset_name}/Annotations/{dataset_name}_{image_set}_in_coco_format.json'  # Change this path to where you want to save the COCO JSON file
with open(coco_output_file, 'w') as f:
    json.dump(coco_dataset, f, indent=2)




## COCO mapping
'''
{'supercategory': 'person', 'id': 1, 'name': 'person'}
{'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'}
{'supercategory': 'vehicle', 'id': 3, 'name': 'car'}
{'supercategory': 'vehicle', 'id': 4, 'name': 'motorcycle'}
{'supercategory': 'vehicle', 'id': 5, 'name': 'airplane'}
{'supercategory': 'vehicle', 'id': 6, 'name': 'bus'}
{'supercategory': 'vehicle', 'id': 7, 'name': 'train'}
{'supercategory': 'vehicle', 'id': 8, 'name': 'truck'}
{'supercategory': 'vehicle', 'id': 9, 'name': 'boat'}
{'supercategory': 'outdoor', 'id': 10, 'name': 'traffic light'}
{'supercategory': 'outdoor', 'id': 11, 'name': 'fire hydrant'}
{'supercategory': 'outdoor', 'id': 13, 'name': 'stop sign'}
{'supercategory': 'outdoor', 'id': 14, 'name': 'parking meter'}
{'supercategory': 'outdoor', 'id': 15, 'name': 'bench'}
{'supercategory': 'animal', 'id': 16, 'name': 'bird'}
{'supercategory': 'animal', 'id': 17, 'name': 'cat'}
{'supercategory': 'animal', 'id': 18, 'name': 'dog'}
{'supercategory': 'animal', 'id': 19, 'name': 'horse'}
{'supercategory': 'animal', 'id': 20, 'name': 'sheep'}
{'supercategory': 'animal', 'id': 21, 'name': 'cow'}
{'supercategory': 'animal', 'id': 22, 'name': 'elephant'}
{'supercategory': 'animal', 'id': 23, 'name': 'bear'}
{'supercategory': 'animal', 'id': 24, 'name': 'zebra'}
{'supercategory': 'animal', 'id': 25, 'name': 'giraffe'}
{'supercategory': 'accessory', 'id': 27, 'name': 'backpack'}
{'supercategory': 'accessory', 'id': 28, 'name': 'umbrella'}
{'supercategory': 'accessory', 'id': 31, 'name': 'handbag'}
{'supercategory': 'accessory', 'id': 32, 'name': 'tie'}
{'supercategory': 'accessory', 'id': 33, 'name': 'suitcase'}
{'supercategory': 'sports', 'id': 34, 'name': 'frisbee'}
{'supercategory': 'sports', 'id': 35, 'name': 'skis'}
{'supercategory': 'sports', 'id': 36, 'name': 'snowboard'}
{'supercategory': 'sports', 'id': 37, 'name': 'sports ball'}
{'supercategory': 'sports', 'id': 38, 'name': 'kite'}
{'supercategory': 'sports', 'id': 39, 'name': 'baseball bat'}
{'supercategory': 'sports', 'id': 40, 'name': 'baseball glove'}
{'supercategory': 'sports', 'id': 41, 'name': 'skateboard'}
{'supercategory': 'sports', 'id': 42, 'name': 'surfboard'}
{'supercategory': 'sports', 'id': 43, 'name': 'tennis racket'}
{'supercategory': 'kitchen', 'id': 44, 'name': 'bottle'}
{'supercategory': 'kitchen', 'id': 46, 'name': 'wine glass'}
{'supercategory': 'kitchen', 'id': 47, 'name': 'cup'}
{'supercategory': 'kitchen', 'id': 48, 'name': 'fork'}
{'supercategory': 'kitchen', 'id': 49, 'name': 'knife'}
{'supercategory': 'kitchen', 'id': 50, 'name': 'spoon'}
{'supercategory': 'kitchen', 'id': 51, 'name': 'bowl'}
{'supercategory': 'food', 'id': 52, 'name': 'banana'}
{'supercategory': 'food', 'id': 53, 'name': 'apple'}
{'supercategory': 'food', 'id': 54, 'name': 'sandwich'}
{'supercategory': 'food', 'id': 55, 'name': 'orange'}
{'supercategory': 'food', 'id': 56, 'name': 'broccoli'}
{'supercategory': 'food', 'id': 57, 'name': 'carrot'}
{'supercategory': 'food', 'id': 58, 'name': 'hot dog'}
{'supercategory': 'food', 'id': 59, 'name': 'pizza'}
{'supercategory': 'food', 'id': 60, 'name': 'donut'}
{'supercategory': 'food', 'id': 61, 'name': 'cake'}
{'supercategory': 'furniture', 'id': 62, 'name': 'chair'}
{'supercategory': 'furniture', 'id': 63, 'name': 'couch'}
{'supercategory': 'furniture', 'id': 64, 'name': 'potted plant'}
{'supercategory': 'furniture', 'id': 65, 'name': 'bed'}
{'supercategory': 'furniture', 'id': 67, 'name': 'dining table'}
{'supercategory': 'furniture', 'id': 70, 'name': 'toilet'}
{'supercategory': 'electronic', 'id': 72, 'name': 'tv'}
{'supercategory': 'electronic', 'id': 73, 'name': 'laptop'}
{'supercategory': 'electronic', 'id': 74, 'name': 'mouse'}
{'supercategory': 'electronic', 'id': 75, 'name': 'remote'}
{'supercategory': 'electronic', 'id': 76, 'name': 'keyboard'}
{'supercategory': 'electronic', 'id': 77, 'name': 'cell phone'}
{'supercategory': 'appliance', 'id': 78, 'name': 'microwave'}
{'supercategory': 'appliance', 'id': 79, 'name': 'oven'}
{'supercategory': 'appliance', 'id': 80, 'name': 'toaster'}
{'supercategory': 'appliance', 'id': 81, 'name': 'sink'}
{'supercategory': 'appliance', 'id': 82, 'name': 'refrigerator'}
{'supercategory': 'indoor', 'id': 84, 'name': 'book'}
{'supercategory': 'indoor', 'id': 85, 'name': 'clock'}
{'supercategory': 'indoor', 'id': 86, 'name': 'vase'}
{'supercategory': 'indoor', 'id': 87, 'name': 'scissors'}
{'supercategory': 'indoor', 'id': 88, 'name': 'teddy bear'}
{'supercategory': 'indoor', 'id': 89, 'name': 'hair drier'}
{'supercategory': 'indoor', 'id': 90, 'name': 'toothbrush'}
'''