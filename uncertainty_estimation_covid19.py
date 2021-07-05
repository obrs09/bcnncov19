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


# CUDA settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

covid19_set = None
notcovid19_set = None

transform_covid19 = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def init_dataset(notcovid19_dir):
    transform_covid19 = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    global covid19_set
    global notcovid19_set
    covid19_set, _, _, _ = dat.getDataset('covid19')
    #covid19_set = torchvision.datasets(covid19_set)
    notcovid19_set = torchvision.datasets.ImageFolder(root="./covid-chestxray-dataset/output/fruit/", transform=transform_covid19)

    # covid19_set = [x[0] for x in covid19_set]
    # covid19_set = covid19_set[:]
    # covid19_set = torch.FloatTensor(covid19_set)
    #
    # notcovid19_set = [x[0] for x in notcovid19_set]
    # notcovid19_set = notcovid19_set[:]
    # notcovid19_set = torch.FloatTensor(notcovid19_set)
    # print(covid19_set)
    # print(notcovid19_set)
    #
    # torch.reshape(covid19_set, (224, 224, 3))
    # torch.reshape(notcovid19_set, (224, 224, 3))



def get_uncertainty_per_image(model, input_image, T=15, normalized=False):
    #input_image = input_image.unsqueeze(0)
    input_images = input_image.repeat(T, 1, 1, 1)

    net_out, ap = model(input_images)
    #print(net_out,ap)
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

    return pred, epistemic, aleatoric


def get_uncertainty_per_batch(model, batch, T=15, normalized=False):
    batch_predictions = []
    net_outs = []
    batches = batch.unsqueeze(0).repeat(T, 1, 1, 1, 1)

    preds = []
    epistemics = []
    aleatorics = []
    
    for i in range(T):  # for T batches
        net_out, _ = model(batches[i].cuda())
        net_outs.append(net_out)
        if normalized:
            prediction = F.softplus(net_out)
            prediction = prediction / torch.sum(prediction, dim=1).unsqueeze(1)
        else:
            prediction = F.softmax(net_out, dim=1)
        batch_predictions.append(prediction)
    
    for sample in range(batch.shape[0]):
        # for each sample in a batch
        pred = torch.cat([a_batch[sample].unsqueeze(0) for a_batch in net_outs], dim=0)
        pred = torch.mean(pred, dim=0)
        preds.append(pred)

        p_hat = torch.cat([a_batch[sample].unsqueeze(0) for a_batch in batch_predictions], dim=0).detach().cpu().numpy()
        p_bar = np.mean(p_hat, axis=0)

        temp = p_hat - np.expand_dims(p_bar, 0)
        epistemic = np.dot(temp.T, temp) / T
        epistemic = np.diag(epistemic)
        epistemics.append(epistemic)

        aleatoric = np.diag(p_bar) - (np.dot(p_hat.T, p_hat) / T)
        aleatoric = np.diag(aleatoric)
        aleatorics.append(aleatoric)

    epistemic = np.vstack(epistemics)  # (batch_size, categories)
    aleatoric = np.vstack(aleatorics)  # (batch_size, categories)
    preds = torch.cat([i.unsqueeze(0) for i in preds]).cpu().detach().numpy()  # (batch_size, categories)

    return preds, epistemic, aleatoric


def get_sample(dataset, sample_type='covid19'):
    idx = np.random.randint(len(dataset.targets))
    if sample_type=='covid19':
        datasetlist = [x[0] for x in dataset]
        sample = datasetlist[idx]
        #sample = dataset[idx]
        truth = dataset.targets[idx]
    else:
        path, truth = dataset.samples[idx]
        sample = torch.from_numpy(np.array(Image.open(path)))


    sample = sample.unsqueeze(0)
    #sample = transform_covid19(sample)
    return sample.to(device), truth, idx



def run(net_type, weight_path, notcovid19_dir):



    init_dataset(notcovid19_dir)

    layer_type = cfg.layer_type
    activation_type = cfg.activation_type

    trainset, testset, inputs, outputs = dat.getDataset('covid19')

    net = getModel(net_type, inputs, outputs, priors=None, layer_type=layer_type, activation_type=activation_type)
    net.load_state_dict(torch.load(weight_path))
    net.train()
    net.to(device)

    fig = plt.figure()
    fig.suptitle('Uncertainty Estimation', fontsize='x-large')
    covid19_img = fig.add_subplot(321)
    notcovid19_img = fig.add_subplot(322)
    epi_stats_norm = fig.add_subplot(323)
    ale_stats_norm = fig.add_subplot(324)
    epi_stats_soft = fig.add_subplot(325)
    ale_stats_soft = fig.add_subplot(326)

    sample_covid19, truth_covid19, idx = get_sample(covid19_set)
    sample_covid19_see = sample_covid19

    pred_covid19, epi_covid19_norm, ale_covid19_norm = get_uncertainty_per_image(net, sample_covid19, T=25, normalized=True)
    pred_covid19, epi_covid19_soft, ale_covid19_soft = get_uncertainty_per_image(net, sample_covid19, T=25, normalized=False)

    sample_covid19.squeeze().cpu()

    image19 = np.array(sample_covid19_see.squeeze().cpu())
    shape_image19 = np.shape(image19)
    #image19 = np.reshape(image19, [shape_image19[1], shape_image19[2], shape_image19[0]])
    image19 = np.reshape(image19, [224, 224, 3])

    image19 = [image19[0]*0.229 + 0.485, image19[1]*0.229 + 0.456, image19[2]*0.225 + 0.406]


    covid19_img.imshow(image19)
    #covid19_img.imshow(sample_covid19_see.squeeze().cpu())
    covid19_img.axis('off')
    covid19_img.set_title('covid19 Truth: {} Prediction: {}'.format(int(truth_covid19), int(np.argmax(pred_covid19))))

    sample_notcovid19, truth_notcovid19, idx = get_sample(notcovid19_set, sample_type='covid19')
    sample_notcovid19_see = sample_notcovid19

    pred_notcovid19, epi_notcovid19_norm, ale_notcovid19_norm = get_uncertainty_per_image(net, sample_notcovid19, T=25, normalized=True)
    pred_notcovid19, epi_notcovid19_soft, ale_notcovid19_soft = get_uncertainty_per_image(net, sample_notcovid19, T=25, normalized=False)

    sample_notcovid19.squeeze().cpu()
    imagen19 = np.array(sample_notcovid19_see.squeeze().cpu())
    shape_imagen19 = np.shape(imagen19)
    #imagen19 = np.reshape(imagen19, [shape_imagen19[1], shape_imagen19[2], shape_imagen19[0]])
    imagen19 = np.reshape(imagen19, [224, 224, 3])

    imagen19 = [imagen19[0] * 0.229 + 0.485, imagen19[1] * 0.229 + 0.456, imagen19[2] * 0.225 + 0.406]

    notcovid19_img.imshow(imagen19)
    #notcovid19_img.imshow(sample_notcovid19.squeeze().cpu())
    notcovid19_img.axis('off')
    notcovid19_img.set_title('notcovid19 Truth: {}({}) Prediction: {}({})'.format(
        int(truth_notcovid19), chr(65 + truth_notcovid19), int(np.argmax(pred_notcovid19)), chr(65 + np.argmax(pred_notcovid19))))

    x = list(range(2))
    data = pd.DataFrame({
        'epistemic_norm': np.hstack([epi_covid19_norm, epi_notcovid19_norm]),
        'aleatoric_norm': np.hstack([ale_covid19_norm, ale_notcovid19_norm]),
        'epistemic_soft': np.hstack([epi_covid19_soft, epi_notcovid19_soft]),
        'aleatoric_soft': np.hstack([ale_covid19_soft, ale_notcovid19_soft]),
        'category': np.hstack([x, x]),
        'dataset': np.hstack([['covid19']*2, ['notcovid19']*2])
    })
    print(data)
    sns.barplot(x='category', y='epistemic_norm', hue='dataset', data=data, ax=epi_stats_norm)
    sns.barplot(x='category', y='aleatoric_norm', hue='dataset', data=data, ax=ale_stats_norm)
    epi_stats_norm.set_title('Epistemic Uncertainty (Normalized)')
    ale_stats_norm.set_title('Aleatoric Uncertainty (Normalized)')

    sns.barplot(x='category', y='epistemic_soft', hue='dataset', data=data, ax=epi_stats_soft)
    sns.barplot(x='category', y='aleatoric_soft', hue='dataset', data=data, ax=ale_stats_soft)
    epi_stats_soft.set_title('Epistemic Uncertainty (Softmax)')
    ale_stats_soft.set_title('Aleatoric Uncertainty (Softmax)')

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "PyTorch Uncertainty Estimation b/w covid19 and notcovid19")
    parser.add_argument('--net_type', default='alexnet', type=str, help='model')
    parser.add_argument('--weights_path', default='/fridge/bcnncov19/checkpoints/covid19/bayesian/model_alexnet_lrt_softplus.pt', type=str, help='weights for model')
    parser.add_argument('--notcovid19_dir', default="./covid-chestxray-dataset/output/xray/noncovidxray/", type=str, help='weights for model')
    args = parser.parse_args()

    run(args.net_type, args.weights_path, args.notcovid19_dir)
