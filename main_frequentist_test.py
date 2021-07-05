
import torch
import numpy as np
import torch.nn as nn
from models.NonBayesianModels.AlexNet import AlexNet
from main_frequentist import getModel
import data

import config_frequentist as cfg
if __name__ == '__main__':

    train_on_gpu = torch.cuda.is_available()
    dataset='covid19'
    net_type = 'alexnet'
    classes = ["covid19", "non"]

    # CUDA settings
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Hyper Parameter settings
    n_epochs = cfg.n_epochs
    lr = cfg.lr
    num_workers = cfg.num_workers
    valid_size = cfg.valid_size
    batch_size = cfg.batch_size
    num_classes = len(classes)

    trainset, testset, inputs, outputs = data.getDataset(dataset)
    train_loader, valid_loader, test_loader = data.getDataloader(
        trainset, testset, valid_size, batch_size, num_workers)

    model = getModel(net_type, inputs, outputs)
    #model = getModel(net_type, 1, 10, priors=None, layer_type=layer_type, activation_type=activation_type)
    model.load_state_dict(torch.load("./checkpoints/covid19/frequentist/model_alexnet.pt"))
    model.train()
    model.to(device)


    criterion = nn.CrossEntropyLoss()

    Num_fruits = len(classes)

    test_loss = 0.0
    class_correct = list(0. for i in range(Num_fruits))
    class_total = list(0. for i in range(Num_fruits))


    # iterate over test data
    for data, target in test_loader:
        print(np.shape(data))
        # move tensors to GPU if CUDA is available
        data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update  test loss
        test_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        # calculate test accuracy for each object class
        for i in range(len(target)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # calculate avg test loss
    test_loss = test_loss/len(test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(Num_fruits):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))


