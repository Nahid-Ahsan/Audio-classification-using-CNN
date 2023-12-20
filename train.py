from classifier import *
from model import *

def training(model, train_dl, num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr = 0.001,
                                                    steps_per_epoch = int(len(train_dl)),
                                                    epochs = num_epochs,
                                                    anneal_strategy = 'linear',
                                                    )

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

        for idx, data in enumerate(train_dl):
            # get the input features and target labels and put them on the GPU 
            inputs, labels = data[0].to(device), data[1].to(device)
            
            # normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # keep stats for loss and accuracy
            running_loss += loss.item()


            # get the predicted class with the highest score
            _, predicted_class = torch.max(outputs, 1)
            # count of predicted classes that matched the target label
            correct_prediction += (predicted_class == labels).sum().item()
            total_prediction += predicted_class.shape[0]

        # print the stats at the end of the training
        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        acc = correct_prediction / total_prediction
        print(f'Epoch: {epoch}, Loss: {avg_loss: 2f}, Accuracy: {acc: 2f}')

    print("Finished Training")

num_epochs = 10
training(audioclf, train_dl, num_epochs)