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
import argparse
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import torchvision
from torch.nn import functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import data as dat
from main_bayesian import getModel
import config_bayesian as cfg

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def softplus(x):
    return np.log(1 + np.exp(x))

# def imshow(image, ax=None, title=None, normalize=True):
#     """Imshow for Tensor."""
#     if ax is None:
#         fig, ax = plt.subplots()
#     image = image.numpy().transpose((1, 2, 0))
#
#     if normalize:
#         mean = np.array([0.5, 0.5, 0.5])
#         std = np.array([0.5, 0.5, 0.5])
#         image = std * image + mean
#         image = np.clip(image, 0, 1)
#
#     ax.imshow(image)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
#     ax.tick_params(axis='both', length=0)
#     ax.set_xticklabels('')
#     ax.set_yticklabels('')
#
#     return ax

def gaussian(x, mu, sig):
    y = 1 / (sig * np.sqrt(2 * np.pi)) * np.exp(-1/2 * np.power(((x - mu) / sig), 2))
    return y

def get_uncertainty_per_image(model, input_image, T=15, normalized=False):
    input_image = input_image.unsqueeze(0)
    input_images = input_image.repeat(T, 1, 1, 1)

    net_out, k = model(input_images)

    pred = torch.mean(net_out, dim=0).cpu().detach().numpy()
    if normalized:
        prediction = F.softplus(net_out)
        p_hat = prediction / torch.sum(prediction, dim=1).unsqueeze(1)

    else:
        p_hat = F.softmax(net_out, dim=1)
    p_hat = p_hat.detach().cpu().numpy()
    p_bar = np.mean(p_hat, axis=0)

    temp = p_hat - np.expand_dims(p_bar, 0)
    epistemic = np.dot(temp.T, temp) / T
    epistemic = np.diag(epistemic)

    aleatoric = np.diag(p_bar) - (np.dot(p_hat.T, p_hat) / T)
    aleatoric = np.diag(aleatoric)

    return pred, epistemic, aleatoric, net_out

if __name__ == '__main__':

    train_on_gpu = torch.cuda.is_available()

    net_type = 'alexnet'
    classes = ["covid19", "non"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    layer_type = cfg.layer_type
    activation_type = cfg.activation_type

    trainset, testset, inputs, outputs = dat.getDataset('covid19')

    model = getModel(net_type, inputs, outputs, priors=None, layer_type=layer_type, activation_type=activation_type)
    model.load_state_dict(torch.load('/fridge/bcnncov19/checkpoints/covid19/bayesian/model_alexnet_lrt_softplus.pt'))
    model.train()
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    #see image
    transform_see = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    testset_see = torchvision.datasets.ImageFolder(root="./covid-chestxray-dataset/output/test",
                                               transform=transform_see)


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

    test_img_see = testset[num_test]
    test_loader_see = torch.utils.data.DataLoader(test_img_see)


    test_img = testset[num_test]
    test_loader = torch.utils.data.DataLoader(test_img)


    print('the img is', test_img[1])
    print('from', testset.class_to_idx)
    test_img = test_img[0].to(device)

    test_img_see = test_img.cpu().detach().numpy().reshape(224, 224, 3)

    pred_covid19, epistemic, aleatoric, net_out = get_uncertainty_per_image(model, test_img, T=100, normalized=True)

    op_final_pred = 1
    final_pred = int(np.argmax(pred_covid19))

    if final_pred:
        op_final_pred = 0

    print(pred_covid19)
    print("prediction", final_pred)
    print(epistemic, aleatoric)

    net_out = net_out.cpu().detach().numpy()

    pdf_mu_1 = np.mean(model.classifier.act_mu.cpu().detach().numpy(), axis=0)[0]
    pdf_mu_2 = np.mean(model.classifier.act_mu.cpu().detach().numpy(), axis=0)[1]
    pdf_var_1 = np.mean(model.classifier.act_var.cpu().detach().numpy(), axis=0)[0]
    pdf_var_2 = np.mean(model.classifier.act_var.cpu().detach().numpy(), axis=0)[1]



    x_values = np.linspace(-2, 4, 1000)
    var_0 = epistemic[0] + aleatoric[0]
    var_1 = epistemic[1] + aleatoric[1]
    std_0 = 1e-16 + np.sqrt(var_0)
    std_1 = 1e-16 + np.sqrt(var_1)
    mu_0 = pred_covid19[0]
    mu_1 = pred_covid19[1]

    plt.figure(0)
    plt.plot(x_values, gaussian(x_values, final_pred, std_0), "-")
    #plt.plot(x_values, gaussian(x_values, op_final_pred, std_1), "-")
    plt.plot(x_values, gaussian(x_values, op_final_pred, 1-std_0), "--")
    plt.xlabel("P(Covid-19)")
    plt.ylabel("density")
    plt.title("prediction of posterior")

    plt.figure(1)
    plt.imshow(test_img_see)

    plt.show()

    # for data, target in test_loader:
    # #     # move tensors to GPU if CUDA is available
    #     data, target = data.cuda(), target.cuda()
    #     output.append(model(data))
    # print('the img is', test_img[1])
    # print('from', testset.class_to_idx)
    # print(output[num_test])
    # print(torch.max(output[num_test], 1))

