import torchxrayvision as xrv
import torchvision
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import os
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms

d = xrv.datasets.COVID19_Dataset(imgpath="covid-chestxray-dataset/images/",
                                 csvpath="covid-chestxray-dataset/metadata.csv")
sample = d[10]
# print(d)
# print(pd.Series(dict(zip(d.pathologies, sample["lab"]))))

# plt.imshow(sample["img"][0], cmap="gray")
# plt.show()

virus = "Pneumonia/Viral/COVID-19"  # Virus to look for
x_ray_view = "PA"  # View of X-Ray
modality = "CT"

metadata = "./covid-chestxray-dataset/metadata.csv"  # Meta info
imageDir = "./covid-chestxray-dataset/images"  # Directory of images
noncovid19outputDir = "./covid-chestxray-dataset/output/CT/noncovidCT"  # Output directory to store selected images
covid19outputDir = "./covid-chestxray-dataset/output/CT/covidCT"  # Output directory to store selected images
otheroutputDir = "./covid-chestxray-dataset/output/xray/otherxray"

metadata_csv = pd.read_csv(metadata)

#modality=list(set(metadata_csv["modality"]))
view = list(set(metadata_csv["view"]))


print(view)


for (i, row) in metadata_csv.iterrows():

    if row["finding"] != virus and row["finding"] != "No Finding" and row["finding"] != "todo" and row["finding"] != "Tuberculosis" and row["finding"] != "Unknown" and row["view"] == x_ray_view:

        filename = row["filename"].split(os.path.sep)[-1]
        filePath = os.path.sep.join([imageDir, filename])
        shutil.copy2(filePath, otheroutputDir)



# transform_cifar = transforms.Compose([
#         transforms.Resize((32, 32)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         ])
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar)
#
# data_transform = transforms.Compose([transforms.RandomResizedCrop(224),
#                                       transforms.ToTensor()])
#
# train_data = datasets.ImageFolder(train_dir, transform=data_transform)
#
# print(trainset)
