import torch
import numpy as np
import torch.nn as nn
import sys

import numpy as np
from models.NonBayesianModels.AlexNet import AlexNet
from main_frequentist import getModel
import data
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import config_frequentist as cfg
def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax

if __name__ == '__main__':
    inputs = 3
    outputs = 2
    train_on_gpu = torch.cuda.is_available()
    dataset = 'covid19'
    net_type = 'alexnet'
    classes = ["covid19", "non"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = getModel(net_type, inputs, outputs)
    model.load_state_dict(torch.load("./checkpoints/covid19/frequentist/model_alexnet.pt"))
    model.train()
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    #see image
    # transform_see = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    # ])
    # testset_see = torchvision.datasets.ImageFolder(root="./covid-chestxray-dataset/output/test",
    #                                            transform=transform_see)
    #

    # test_img_see = testset_see[num_test]
    # test_loader_see = torch.utils.data.DataLoader(test_img_see)
    #
    #
    # imshow(test_img_see[0], normalize=False)
    # print('the img is', test_img_see[1])
    # print('from', testset_see.class_to_idx)
    # #plt.show()


    #predict img
    transform_covid19 = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    testset = torchvision.datasets.ImageFolder(root="./covid-chestxray-dataset/output/test",
                                               transform=transform_covid19)

    num_test = np.random.randint(0, len(testset))

    test_img = testset[num_test]


    test_loader = torch.utils.data.DataLoader(testset)
    i=1
    output = []
    for data, target in test_loader:
    #     # move tensors to GPU if CUDA is available
        data, target = data.cuda(), target.cuda()
        output.append(model(data))
    print('the img is', test_img[1])
    print('from', testset.class_to_idx)
    print(output[num_test])
    print(torch.max(output[num_test], 1))
    # for data, target in test_loader:
    #     # move tensors to GPU if CUDA is available
    #     data, target = data.cuda(), target.cuda()
    #     # forward pass: compute predicted outputs by passing inputs to the model
    #     output = model(data)
    #     # calculate the batch loss
    #     loss = criterion(output, target)
    #     # update  test loss
    #     test_loss += loss.item()*data.size(0)
    #     # convert output probabilities to predicted class
    #     _, pred = torch.max(output, 1)
    #     # compare predictions to true label
    #     correct_tensor = pred.eq(target.data.view_as(pred))
    #     correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    #     # calculate test accuracy for each object class
    #     for i in range(len(target)):
    #         label = target.data[i]
    #         class_correct[label] += correct[i].item()
    #         class_total[label] += 1
    #
    # # calculate avg test loss
    # test_loss = test_loss/len(test_loader.dataset)
    # print('Test Loss: {:.6f}\n'.format(test_loss))
    #
    # for i in range(Num_fruits):
    #     if class_total[i] > 0:
    #         print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
    #             classes[i], 100 * class_correct[i] / class_total[i],
    #             np.sum(class_correct[i]), np.sum(class_total[i])))
    #     else:
    #         print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))
    #
    # print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    #     100. * np.sum(class_correct) / np.sum(class_total),
    #     np.sum(class_correct), np.sum(class_total)))


