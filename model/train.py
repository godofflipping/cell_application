import os
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix


class TrainNeuralNetwork():
    def __init__(self,
                 model, criterion,
                 optimizer, device,
                 train_loader, test_loader,
                 class_count, confusion_matrixes,
                 reverse_dict, categories,
                 random_state, scheduler=None
                 ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.class_count = class_count
        self.confusion_matrixes = confusion_matrixes
        self.reverse_dict = reverse_dict
        self.categories = categories
        self.scheduler = scheduler
        torch.manual_seed(random_state)
        torch.cuda.manual_seed(random_state)
        random.seed(random_state)
        np.random.seed(random_state)

    def train(self, num_epochs, validation_history, training_history):
        for epoch in range(1, num_epochs + 1):
            self.model.train()
            running_loss = 0.0
            epoch_accuracy = 0.0

            for inputs, labels in tqdm(self.train_loader):
                inputs = inputs.to(self.device).float()
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                predicted_outputs = self.model(inputs)

                _, predictions = torch.max(predicted_outputs.data, 1)

                loss = self.criterion(predicted_outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                epoch_accuracy += accuracy_score(
                                    y_true=labels.cpu().numpy(),
                                    y_pred=predictions.cpu().numpy()
                                )

            epoch_loss = running_loss / len(self.train_loader)
            epoch_accuracy /= len(self.train_loader)

            validation_accuracy, validation_loss = self.evaluate()

            validation_history['loss'].append(validation_loss)
            validation_history['accuracy'].append(validation_accuracy)
            training_history['loss'].append(epoch_loss)
            training_history['accuracy'].append(epoch_accuracy)

            if self.scheduler is not None:
                self.scheduler.step()

            print()
            print("Epoch {}/{}:".format(epoch, num_epochs))
            print('Train Accuracy: {:.4f}   Train Loss: {:.4f}'.format(
                epoch_accuracy, epoch_loss
            ))
            print('Valid Accuracy: {:.4f}   Valid Loss: {:.4f}'.format(
                validation_accuracy, validation_loss
            ))
            print()

    def evaluate(self):
        validation_loss = 0.0
        validation_accuracy = 0
        matrix = np.zeros((self.class_count, self.class_count))

        self.model.eval()

        for inputs, labels in tqdm(self.test_loader):
            inputs = inputs.to(self.device).float()
            labels = labels.to(self.device)

            with torch.no_grad():
                predicted_outputs = self.model(inputs)
                loss = self.criterion(predicted_outputs, labels)

            _, predictions = torch.max(predicted_outputs.data, 1)
            validation_loss += loss.item()

            validation_accuracy += accuracy_score(
                y_true=labels.cpu().numpy(),
                y_pred=predictions.cpu().numpy()
            )

            matrix += confusion_matrix(
                y_true=[self.reverse_dict[i] for i in labels.cpu().numpy()],
                y_pred=[self.reverse_dict[i] for i in predictions.cpu().numpy()],
                labels=self.categories
            )

        validation_loss /= len(self.test_loader)
        validation_accuracy /= len(self.test_loader)

        self.confusion_matrixes.append(matrix)

        return validation_accuracy, validation_loss

    def run(self, link, num_epochs, validation_history, training_history, conf_matrix_history, save_weights):
        weights_effnet_path = link + 'weights.pth'
        if save_weights:
            if os.path.isfile(weights_effnet_path):
                self.model.load_state_dict(torch.load(weights_effnet_path, map_location=torch.device(self.device)))
            self.train(
                num_epochs=num_epochs,
                validation_history=validation_history,
                training_history=training_history
            )
            torch.save(self.model.state_dict(), weights_effnet_path)
        else:
            self.model.load_state_dict(torch.load(weights_effnet_path, map_location=torch.device(self.device)))

        conf_matrix_history['list'] = [conf_matrix.tolist() for conf_matrix in self.confusion_matrixes]

        FILE_PATH_TRAIN = link + 'train_acc.json'
        with open(FILE_PATH_TRAIN, 'w') as file_train:
            json.dump(training_history, file_train)

        FILE_PATH_VALID = link + 'valid_acc.json'
        with open(FILE_PATH_VALID, 'w') as file_valid:
            json.dump(validation_history, file_valid)

        FILE_PATH_MATRIX = link + 'conf_matrix.json'
        with open(FILE_PATH_MATRIX, 'w') as file_matrix:
            json.dump(conf_matrix_history, file_matrix)

        print(max(training_history['accuracy']), max(validation_history['accuracy']))