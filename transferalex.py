import torchvision
from torchvision import transforms
import torch.optim as optim
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import data
import config_frequentist as cfg



# check if CUDA is available
train_on_gpu = torch.cuda.is_available()



if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')


# define training and test data directories
train_dir = "./covid-chestxray-dataset/output/train"
test_dir = "./covid-chestxray-dataset/output/test"


transform_covid19 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

trainset = torchvision.datasets.ImageFolder(root="./covid-chestxray-dataset/output/train", transform=transform_covid19)
testset = torchvision.datasets.ImageFolder(root="./covid-chestxray-dataset/output/test", transform=transform_covid19)

# print out some data stats
print('Num training images: ', len(trainset))
print('Num test images: ', len(testset))

n_epochs = cfg.n_epochs
lr = cfg.lr
num_workers = cfg.num_workers
valid_size = cfg.valid_size
batch_size = cfg.batch_size


train_loader, valid_loader, test_loader = data.getDataloader(
        trainset, testset, valid_size, batch_size, num_workers)

# Load the pretrained model from pytorch
AlexNet_model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)

# Freeze training for all "features" layers
AlexNet_model.classifier[4] = nn.Linear(4096,1024)

#Updating the third and the last classifier that is the output layer of the network. Make sure to have 10 output nodes if we are going to get 10 class labels through our model.
AlexNet_model.classifier[6] = nn.Linear(1024,2)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
AlexNet_model.to(device)


# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()

# specify optimizer (stochastic gradient descent) and learning rate = 0.001
optimizer = optim.Adam(AlexNet_model.parameters(), lr=lr)


loss_list=[]

# number of epochs to train the model


for epoch in range(10):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = AlexNet_model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        loss_list.append(running_loss)


plt.plot(loss_list)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("loss vs epoch")
plt.show()
torch.cuda.empty_cache()

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = AlexNet_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = AlexNet_model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

