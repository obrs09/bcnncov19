from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import skimage
from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split
#import opencv
from skimage.io import imread
from skimage.transform import resize
import time
import sys

start = time.time()


def load_image_files(container_path, dimension=(256, 256, 3)):
    """
    Load image files with categories as subfolder names
    which performs like scikit-learn sample dataset

    Parameters
    ----------
    container_path : string or unicode
        Path to the main folder holding one subfolder per category
    dimension : tuple
        size to which image are adjusted to

    Returns
    -------
    Bunch
    """
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]

    descr = "A image classification dataset"
    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            img = skimage.io.imread(file)

            img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')

            flat_data.append(img_resized.flatten())
            images.append(img_resized)
            target.append(i)
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)
    #print(images)
    return Bunch(data=flat_data,
                 target=target,
                 target_names=categories,
                 images=images,
                 DESCR=descr),folders

image_dataset_train,folders_train = load_image_files("train/")
image_dataset_test,folders_test = load_image_files("test/")
#image_dataset = load_image_files("images/")

X_train = image_dataset_train.data
y_train = image_dataset_train.target

X_test = image_dataset_test.data
y_test = image_dataset_test.target
#     image_dataset.data, image_dataset.target, test_size=0.3,random_state=109)


# param_grid = [
#   {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
#   {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
#  ]
#svc = svm.SVC()
clf = svm.SVC()
#clf = GridSearchCV(svc, param_grid)
clf.fit(X_train, y_train)

print(folders_train)

y_pred = clf.predict(X_test)


print(y_pred)
print(y_test)
len_of_y = len(y_pred)
predict_correct_covid = 0
predict_wrong_covid = 0
predict_correct_noncovid = 0
predict_wrong_noncovid = 0

for i in range(len_of_y):
    if y_pred[i] == y_test[i] and y_pred[i] == 0:
        predict_correct_covid += 1
    elif y_pred[i] == y_test[i] and y_pred[i] == 1:
        predict_correct_noncovid += 1
    elif y_pred[i] != y_test[i] and y_pred[i] == 0:
        predict_wrong_covid += 1
    elif y_pred[i] != y_test[i] and y_pred[i] == 1:
        predict_wrong_noncovid += 1

print("predict_correct_covid", predict_correct_covid)
print("predict_wrong_covid", predict_wrong_covid)
print("predict_correct_noncovid", predict_correct_noncovid)
print("predict_wrong_noncovid", predict_wrong_noncovid)

print("percen of correct covid", predict_correct_covid/(predict_correct_covid + predict_wrong_covid))
print("percen of correct noncovid", predict_correct_noncovid/(predict_correct_noncovid + predict_wrong_noncovid))

print("precent over all", (predict_correct_covid + predict_correct_noncovid)/len_of_y)



end = time.time()
print("time", end - start)