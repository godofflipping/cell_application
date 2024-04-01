import json
import numpy as np

import torch
from torch import nn, optim
from torchvision import models

torch.cuda.empty_cache()

from train import TrainNeuralNetwork
from get_data import get_data
from visualisation import plot_accuracy_loss, plot_conf_matrix, show_images

import warnings
warnings.filterwarnings("ignore")


def main():
    train_loader, test_loader, reverse_dict, categories = get_data()
    class_count = len(categories)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    torch.cuda.empty_cache()

    model_effnet = models.efficientnet_b5(pretrained=False)
    num_features = model_effnet.classifier[1].in_features
    model_effnet.classifier = nn.Linear(num_features, class_count)

    optimizer = optim.Adam(model_effnet.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model_effnet = model_effnet.to(device)
    criterion = criterion.to(device)

    #print(summary(model_effnet, (channels, image_side, image_side), device='cuda'))

    training_history = {'accuracy': [], 'loss':[] }
    validation_history = {'accuracy': [], 'loss':[] }
    conf_matrix_history = {'list': []}
    confusion_matrixes = conf_matrix_history['list']

    IS_FILE = True

    LINK = 'efficient_net_b5/'

    if IS_FILE:
        FILE_PATH_TRAIN = LINK + 'effnet_train_acc.json'
        with open(FILE_PATH_TRAIN, 'r') as file:
            training_history = json.load(file)

        FILE_PATH_VALID = LINK + 'effnet_valid_acc.json'
        with open(FILE_PATH_VALID, 'r') as file:
            validation_history = json.load(file)

        FILE_PATH_MATRIX = LINK + 'effnet_conf_matrix.json'
        with open(FILE_PATH_MATRIX, 'r') as file:
            conf_matrix_history = json.load(file)

            for i in range(len(conf_matrix_history['list'])):
                confusion_matrixes.append(np.array([np.array(row) for row in conf_matrix_history['list'][i]]))

    num_epochs = 4

    train_effnet = TrainNeuralNetwork(
        model=model_effnet,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        train_loader=train_loader,
        test_loader=test_loader,
        class_count=class_count,
        confusion_matrixes=confusion_matrixes,
        reverse_dict=reverse_dict,
        categories=categories
    )

    train_effnet.run(
        link=LINK,
        num_epochs=num_epochs,
        validation_history=validation_history,
        training_history=training_history,
        conf_matrix_history=conf_matrix_history,
        save_weights=True
    )
    
    
if __name__ == "__main__":
    main()