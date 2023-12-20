from model import *
from train import *
from classifier import *


def Validation(model, val_dl):
    correct_prediction = 0
    total_prediction = 0

    # disable gradient update
    with torch.no_grad():
        for data in val_dl:
            # get the input features and target labels and put them on the GPU 
            inputs, labels = data[0].to(device), data[1].to(device)
            
            # normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # get predictions
            outputs = model(inputs)

            # get the predicted class with the highest score
            _, predicted_class = torch.max(outputs, 1)
            # count of predicted classes that matched the target label
            correct_prediction += (predicted_class == labels).sum().item()
            total_prediction += predicted_class.shape[0]

    acc = correct_prediction / total_prediction
    print(f'Accuracy: {acc:2f}, Total Items: {total_prediction}')

inference(audioclf, val_dl)