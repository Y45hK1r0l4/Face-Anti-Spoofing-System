import os
import random
import shutil
from enum import unique
from fileinput import filename
from itertools import islice
from os.path import exists

outputFolderPath = "Dataset/SplitData"
inputFolderPath = "Dataset/all"
splitRatio = {"train": 0.7, "val": 0.2, "test": 0.1}
classes = ["fake", "real"]

try:
    shutil.rmtree(outputFolderPath)
    print("Removed Directory")
except OSError as e:
    os.mkdir(outputFolderPath)


# ---------- Directory to Create --------- #
os.makedirs(f"{outputFolderPath}/train/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/train/labels", exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/labels", exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/labels", exist_ok=True)


# ---------- Get The Names ----------- #
listNames = os.listdir(inputFolderPath)
print(listNames)
print(len(listNames))
uniqueNames = []
for name in listNames:
    uniqueNames.append(name.split('.')[0])

uniqueNames = list(set(uniqueNames))

# ----------- Shuffle -------------- #
random.shuffle(uniqueNames)
print(uniqueNames)

# --------- Find the number of images for each folder --------- #
lenData = len(uniqueNames)
lenTrain = int(lenData * splitRatio['train'])
lenVal = int(lenData * splitRatio['val'])
lenTest = int(lenData * splitRatio['test'])


# -------- Put remaining images in Training -------- #
if lenData != lenTrain + lenTest + lenVal:
    remaining = lenData - (lenTrain + lenTest + lenVal)
    lenTrain += remaining


# ------- Split the list ------- #
lengthToSplit = [lenTrain, lenVal, lenTest]
Input = iter(uniqueNames)
Output = [list(islice(Input, elem)) for elem in lengthToSplit]
print(f'Total Images: {lenData} \nSplit: {lenTrain} {lenVal} {lenTest}')


# -------------------- Copy the files ---------------------#
sequence = ['train', 'val', 'test']

for i, out in enumerate(Output):
    for fileName in out:
        shutil.copy(f'{inputFolderPath}/{fileName}.jpg',f'{outputFolderPath}/{sequence[i]}/images/{fileName}.jpg')
        shutil.copy(f'{inputFolderPath}/{fileName}.txt',f'{outputFolderPath}/{sequence[i]}/labels/{fileName}.txt')

print("Split Process Completed...")


# -------------- Creating Data.yml file ------------- #

dataYaml = (f'path: C:\\Users\\USER\\PycharmProjects\\AntiSpoofing\\DataSet\n'
f'train: train/images\n'
f'val: val/images\n'
f'test: test/images\n'
f'\n'
f'nc: {len(classes)}\n'
f'names: {classes}')


# Writing dataYaml to the file
with open(f"{outputFolderPath}/data.yml", 'w') as f:
    f.write(dataYaml)





