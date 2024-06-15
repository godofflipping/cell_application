from train import TrainNeuralNetwork
from get_data import get_data
import warnings
import json
import random
import numpy as np
import torch
from torch import nn, optim
from torchvision import models
from PIL import ImageFile

warnings.filterwarnings("ignore")
torch.cuda.empty_cache()
ImageFile.LOAD_TRUNCATED_IMAGES = True


def model_run():

    RANDOM_STATE = 123
    torch.manual_seed(RANDOM_STATE)
    torch.cuda.manual_seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    train_loader, test_loader, reverse_dict, categories = get_data(RANDOM_STATE)
    class_count = len(categories)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    def classifier_2layers(num_features):
        return nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, class_count)
        )

    def classifier_3layers(num_features):
        return nn.Sequential(
            nn.Linear(num_features, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, class_count)
        )

    torch.cuda.empty_cache()

    #EfficientNet B5

    model_effnet = models.efficientnet_b5(pretrained=False)
    num_features = model_effnet.classifier[1].in_features
    model_effnet.classifier = nn.Linear(num_features, class_count)
    optimizer_effnet = optim.Adam(model_effnet.parameters(), lr=0.001)
    model_effnet = model_effnet.to(device)

    link_effnet = 'EfficientNetB5/'

    #ResNeXt50

    model_resnext = models.resnext50_32x4d(pretrained=False)
    num_features = model_resnext.fc.in_features
    model_resnext.fc = nn.Linear(num_features, class_count)
    optimizer_resnext = optim.Adam(model_resnext.parameters(), lr=0.001)
    model_resnext = model_resnext.to(device)

    link_resnext = 'ResNeXt50/'

    #DenseNet121

    model_densenet = models.densenet121(pretrained=False)
    num_features = model_densenet.classifier.in_features
    model_densenet.classifier = nn.Linear(num_features, class_count)
    optimizer_densenet = optim.Adam(model_densenet.parameters(), lr=0.001)
    model_densenet = model_densenet.to(device)

    link_densenet = 'DenseNet121/'

    #EfficientNetV2
    """
    model_effnetv2 = models.efficientnet_v2_s(pretrained=False)
    num_features = model_effnetv2.classifier[1].in_features
    model_effnetv2.classifier = nn.Linear(num_features, class_count)

    #https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout

    #model_effnetv2.classifier = classifier_2layers(num_features)

    optimizer_effnetv2 = optim.Adam(model_effnetv2.parameters(), lr=0.001)
    model_effnetv2 = model_effnetv2.to(device)

    link_effnetv2 = 'EfficientNetV2/Full/'
    """

    learning_rate = 1e-5

    # Variant 1

    model_effnetv2_var1 = models.efficientnet_v2_s(pretrained=True)

    for param in model_effnetv2_var1.features[:-2].parameters():
        param.require_grad = False

    num_features = model_effnetv2_var1.classifier[1].in_features
    model_effnetv2_var1.classifier = nn.Linear(num_features, class_count)

    optimizer_effnetv2_var1 = optim.Adam(model_effnetv2_var1.parameters(), lr=learning_rate)
    model_effnetv2_var1 = model_effnetv2_var1.to(device)

    link_effnetv2_var1 = 'EfficientNetV2/TransferLearning/2layer1linear/'

    # Variant 2

    model_effnetv2_var2 = models.efficientnet_v2_s(pretrained=True)

    for param in model_effnetv2_var2.features[:-2].parameters():
        param.require_grad = False

    num_features = model_effnetv2_var2.classifier[1].in_features
    model_effnetv2_var2.classifier = classifier_2layers(num_features)

    optimizer_effnetv2_var2 = optim.Adam(model_effnetv2_var2.parameters(), lr=learning_rate)
    model_effnetv2_var2 = model_effnetv2_var2.to(device)

    link_effnetv2_var2 = 'EfficientNetV2/TransferLearning/2layer2linear/'

    # Variant 3

    model_effnetv2_var3 = models.efficientnet_v2_s(pretrained=True)

    for param in model_effnetv2_var3.features[:-1].parameters():
        param.require_grad = False

    num_features = model_effnetv2_var3.classifier[1].in_features
    model_effnetv2_var3.classifier = classifier_2layers(num_features)

    optimizer_effnetv2_var3 = optim.Adam(model_effnetv2_var3.parameters(), lr=learning_rate)
    model_effnetv2_var3 = model_effnetv2_var3.to(device)

    link_effnetv2_var3 = 'EfficientNetV2/TransferLearning/1layer2linear/'

    # Variant 4

    model_effnetv2_var4 = models.efficientnet_v2_s(pretrained=True)

    for param in model_effnetv2_var4.features[:-3].parameters():
        param.require_grad = False

    num_features = model_effnetv2_var4.classifier[1].in_features
    model_effnetv2_var4.classifier = nn.Linear(num_features, class_count)

    optimizer_effnetv2_var4 = optim.Adam(model_effnetv2_var4.parameters(), lr=learning_rate)
    model_effnetv2_var4 = model_effnetv2_var4.to(device)

    link_effnetv2_var4 = 'EfficientNetV2/TransferLearning/3layer1linear/'

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    comparison = [
        #(model_effnet, optimizer_effnet, link_effnet),
        #(model_resnext, optimizer_resnext, link_resnext),
        #(model_effnetv2_var1, optimizer_effnetv2_var1, link_effnetv2_var1),
        #(model_effnetv2_var2, optimizer_effnetv2_var2, link_effnetv2_var2),
        #(model_effnetv2_var3, optimizer_effnetv2_var3, link_effnetv2_var3),
        #(model_densenet, optimizer_densenet, link_densenet)
        (model_effnetv2_var4, optimizer_effnetv2_var4, link_effnetv2_var4)
    ]

    IS_FILE = False
    num_epochs = 40

    for model, optimizer, link in comparison:
        print('--------------------------------------')
        print('MODEL:\t ' + link.split('/')[0])
        print('MODE:\t ' + link.split('/')[1] + ' ' + link.split('/')[2])
        print('--------------------------------------')

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        training_history = {'accuracy': [], 'loss':[] }
        validation_history = {'accuracy': [], 'loss':[] }
        conf_matrix_history = {'list': []}
        confusion_matrixes = conf_matrix_history['list']

        if IS_FILE:
            FILE_PATH_TRAIN = link + 'train_acc.json'
            with open(FILE_PATH_TRAIN, 'r') as file:
                training_history = json.load(file)

            FILE_PATH_VALID = link + 'valid_acc.json'
            with open(FILE_PATH_VALID, 'r') as file:
                validation_history = json.load(file)

            FILE_PATH_MATRIX = link + 'conf_matrix.json'
            with open(FILE_PATH_MATRIX, 'r') as file:
                conf_matrix_history = json.load(file)

                for i in range(len(conf_matrix_history['list'])):
                    confusion_matrixes.append(np.array([np.array(row) for row in conf_matrix_history['list'][i]]))

        train_network = TrainNeuralNetwork(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train_loader=train_loader,
            test_loader=test_loader,
            class_count=class_count,
            confusion_matrixes=confusion_matrixes,
            reverse_dict=reverse_dict,
            categories=categories,
            random_state=RANDOM_STATE,
            scheduler=scheduler
        )

        train_network.run(
            link=link,
            num_epochs=num_epochs,
            validation_history=validation_history,
            training_history=training_history,
            conf_matrix_history=conf_matrix_history,
            save_weights=True
        )


if __name__ == "__main__":
    model_run()
