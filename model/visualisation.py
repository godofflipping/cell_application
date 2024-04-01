import torch
import matplotlib.pyplot as plt
import seaborn as sns

def plot_accuracy_loss(training_history, validation_history, LINK):
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    axs[0].plot(training_history['loss'], label='Train Loss')
    axs[0].plot(validation_history['loss'], label='Validation Loss')
    axs[0].set_title("Losses over Epochs")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    axs[1].plot(training_history['accuracy'], label='Train Accuracy')
    axs[1].plot(validation_history['accuracy'], label='Vaidation Accuracy')
    axs[1].set_title("Accuracies over Epochs")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()

    plt.show()

    fig.savefig(LINK + 'plot.png')
    
def show_images(data_loader, device, reverse_dict, model=None):
    images, labels = next(iter(data_loader))
    plt.figure(figsize = (20, 20))
    length = len(labels)
    sample = min(length, 16)

    if model != None:
        images = images.to(device).float()
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)

    for i in range(sample):
        plt.subplot(4, 4, i + 1)
        image = images[i] / 255
        image = image.permute(1, 2, 0)
        plt.imshow(image.cpu())

        if model == None:
            plt.title(
                reverse_dict[labels[i].item()],
                color='blue',
                fontsize=12
            )
        else:
            if predictions[i] == labels[i]:
                title_color = 'green'
            else:
                title_color = 'red'
            plt.title(
                f'Predicted: {reverse_dict[predictions[i].item()]} | True: {reverse_dict[labels[i].item()]}',
                color=title_color,
                fontsize=12
            )
        plt.axis('off')
    plt.show()
    
    
def plot_conf_matrix(confusion_matrixes, categories, LINK):
    fig, ax = plt.subplots(figsize=(12, 9))
    #ConfusionMatrixDisplay(confusion_matrixes[-1], display_labels=categories, ).plot(ax=ax);

    conf_matrix = confusion_matrixes[-1]
    conf_matrix_norn = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    sns.heatmap(conf_matrix_norn, annot=False, xticklabels=categories, yticklabels=categories)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    fig.savefig(LINK + 'matrix.png')
