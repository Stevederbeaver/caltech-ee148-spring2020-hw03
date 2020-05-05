from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import seaborn as sns

import os

'''
This code is adapted from two sources:
(i) The official PyTorch MNIST example (https://github.com/pytorch/examples/blob/master/mnist/main.py)
(ii) Starter code from Yisong Yue's CS 155 Course (http://www.yisongyue.com/courses/cs155/2020_winter/)
'''

class fcNet(nn.Module):
    '''
    Design your model with fully connected layers (convolutional layers are not
    allowed here). Initial model is designed to have a poor performance. These
    are the sample units you can try:
        Linear, Dropout, activation layers (ReLU, softmax)
    '''
    def __init__(self):
        # Define the units that you will use in your model
        # Note that this has nothing to do with the order in which operations
        # are applied - that is defined in the forward function below.
        super(fcNet, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=20)
        self.fc2 = nn.Linear(20, 10)
        self.dropout1 = nn.Dropout(p=0.5)

    def forward(self, x):
        # Define the sequence of operations your model will apply to an input x
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = F.relu(x)

        output = F.log_softmax(x, dim=1)
        return output


class ConvNet(nn.Module):
    '''
    Design your model with convolutional layers.
    '''
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=1)
        self.conv2 = nn.Conv2d(8, 8, 3, 1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(200, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output


class Net(nn.Module):
    '''
    Build the best MNIST classifier.
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5,5), stride=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(5,5), stride=1)
        self.maxpool1 = nn.MaxPool2d((2, 2))

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3,3), stride=1)
        self.maxpool2 = nn.MaxPool2d((2, 2), stride = (2, 2))

        self.dropout = nn.Dropout(0.25)

        self.fc1 = nn.Linear(576, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool1(x)

        x = self.dropout(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = self.dropout(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output

def train(args, model, device, train_loader, optimizer, epoch):
    '''
    This is your training function. When you call this function, the model is
    trained for 1 epoch.
    '''
    model.train()   # Set the model to training mode
    correct = 0
    train_num = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()               # Clear the gradient
        output = model(data)                # Make predictions
        loss = F.nll_loss(output, target)   # Compute loss
        loss.backward()                     # Gradient computation
        optimizer.step()                    # Perform a single optimization step

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        train_num += len(data)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                100. * batch_idx / len(train_loader), loss.item()))

    print('\nTraining set Accuracy: {}/{} ({:.2f}%)\n'.format(
    correct, train_num, 100. * correct / train_num))

# Calculate the loss of any set (training set)
def calculate_train(model, device, loader):
    model.eval()
    loader_loss = 0
    correct = 0
    loader_num = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loader_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            loader_num += len(data)

    loader_loss /= loader_num
    accuracy = 100. * correct / loader_num

    return loader_loss, accuracy


def test(model, device, test_loader):
    model.eval()    # Set the model to inference mode
    test_loss = 0
    correct = 0
    test_num = 0
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_num += len(data)

    test_loss /= test_num

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, test_num,
        100. * correct / test_num))

# Visualizing wrongly classified images in the test set
def test_with_image_output(model, device, test_loader):
    wrong_images = []
    correct_labels = []
    predicted_labels = []

    model.eval()    # Set the model to inference mode
    test_loss = 0
    correct = 0
    test_num = 0
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            for ind in range(len(pred)):
                if pred[ind] != target[ind]:
                    wrong_images.append(data[ind])
                    correct_labels.append(target[ind])
                    predicted_labels.append(pred[ind])
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_num += len(data)

    test_loss /= test_num
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, test_num,
        100. * correct / test_num))

    for i in range(9):
        image = wrong_images[i]
        plt.imshow(image[0, :, :])
        plt.title("Correct Label = "+ str(correct_labels[i].item()) +
             ", Predicted Label = "+ str(predicted_labels[i].item()))
        plt.savefig('Wrong Image' + ' '+ str(i+1))
        plt.show()

# The following code is selected from scikit-learn.org, which take a matrix as the input and plots it
# Reference: https://deeplizard.com/learn/video/0LhiS6yu2qQ
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(title + ' of test set')

# Visualizing the confusion matrix for the test set
def test_with_confusion_matrix(model, device, test_loader, correct_label):
    model.eval()    # Set the model to inference mode
    test_loss = 0
    correct = 0
    test_num = 0
    with torch.no_grad():   # For the inference step, gradient is not computed
        all_pred = torch.tensor([]).float()
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            all_pred = torch.cat((all_pred, pred.squeeze().float()), 0)
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_num += len(data)

    test_loss /= test_num

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, test_num,
        100. * correct / test_num))

    print(len(all_pred))
    cm = confusion_matrix(correct_label, all_pred)
    label_class = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    plot_confusion_matrix(cm, label_class)


def extract_feature_vector_Net(trained_model, data):
    x = trained_model.conv1(data)
    x = F.relu(x)
    x = trained_model.conv2(x)
    x = F.relu(x)
    x = trained_model.maxpool1(x)

    x = trained_model.dropout(x)
    x = trained_model.conv3(x)
    x = F.relu(x)
    x = trained_model.conv4(x)
    x = F.relu(x)
    x = trained_model.maxpool2(x)
    x = trained_model.dropout(x)

    x = torch.flatten(x, 1)
    x = trained_model.fc1(x)
    x = F.relu(x)
    x = trained_model.dropout(x)

    return x

# Obtaining the embedded vectors
def test_with_embedding(model, device, test_loader):
    embedding_vectors = [[] for i in range(10)]

    model.eval()    # Set the model to inference mode
    test_loss = 0
    correct = 0
    test_num = 0
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_num += len(data)

            vec = extract_feature_vector_Net(model, data)
            for i in range(len(target)):
                embedding_vectors[target[i]].append(vec[i])

    test_loss /= test_num

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, test_num,
        100. * correct / test_num))

    return embedding_vectors

# The following code is selected from datacamp.com, which is used fro visualizing the high-dimensional embedding
# Reference: https://www.datacamp.com/community/tutorials/introduction-t-sne
def TSNE_visualize(vectors):
    vectors_stack = [torch.stack(vector) for vector in vectors]
    vector_entire = torch.cat(vectors_stack, 0)
    vector_high_D = TSNE(n_components=2).fit_transform(vector_entire)
    index = 0
    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', np.array([0.5, 0.25, 0]), np.array([0.25, 0, 0.5]), np.array([0, 0.5, 0.25])]
    for i in range(len(vectors_stack)):
        c0 = color_list[i]
        length = len(vectors_stack[i])
        plt.scatter(vector_high_D[index:(index + length), 0], vector_high_D[index:(index + length), 1], s=0.1, cmap='viridis', c=c0)
        index += length
    plt.show()
    plt.ylabel('Dimension 1')
    plt.xlabel('Dimension 2')
    plt.title('Feature Vector Visualization')
    plt.savefig('High_Dimensional_Feature_Vector')


def closest_image_visualization(model, device, test_loader):
    length = len(test_loader)
    image_list = []
    vector_list  = []
    model.eval()    # Set the model to inference mode
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        vec = extract_feature_vector_Net(model, data)
        for i in range(len(vec)):
            vector_list.append(vec[i])
            image_list.append(data[i])
    index_list = [626]

    for ind in index_list:
        image = image_list[ind]
        vec = vector_list[ind]
        distance_list = []
        for i in range(len(image_list)):
            norm = torch.norm(vector_list[i]-vec)
            distance_list.append(norm)
        distance_array = np.array(distance_list)
        indexes = np.argsort(distance_array)[:9]

        plt.imshow(image[0,:,:])
        plt.show()
        plt.title('Chosen image' + ' '+ str(ind))
        plt.savefig('Chosen image' + ' '+ str(ind))

        exp = 1
        for j in indexes[1:]:
            plt.imshow(image_list[j][0,:,:])
            plt.title('The ' + str(exp) + '-th image closest to image '+ str(ind))
            plt.savefig('The ' + str(exp) + '-th image closest to image '+ str(ind))
            exp += 1
            plt.show()


def main():
    # Training settings
    # Use the command line to modify the default settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--step', type=int, default=1, metavar='N',
                        help='number of epochs between learning rate reductions (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='evaluate your model on the official test set')
    parser.add_argument('--load-model', type=str,
                        help='model file path')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Evaluate on the official test set
    if args.evaluate:
        assert os.path.exists(args.load_model)

        # Set the test model
        model = Net().to(device)
        model.load_state_dict(torch.load(args.load_model))

        test_dataset = datasets.MNIST('../data', train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

        test(model, device, test_loader)

        # Visualizing the first few wrongly classified images
        test_with_image_output(model, device, test_loader)

        # Visualizing the CNN kernels of trained model

        first_layer_weights = model.conv1.weight.data
        CNN_first_layer_kernels = []
        for i in range(9):
            image = first_layer_weights[i,0,:,:]
            plt.imshow(image)
            plt.title("Kernel "+ str(i+1) + " of CNN's First Layer")
            plt.show()
            plt.savefig("Kernel "+ str(i+1))

        # Visualizing the confusion matrix

        correct_labels = test_dataset.targets
        test_with_confusion_matrix(model, device, test_loader, correct_labels)

        # Visualizing the embedded high dimensional vectors

        vectors = test_with_embedding(model, device, test_loader)
        TSNE_visualize(vectors)

        closest_image_visualization(model, device, test_loader)

        return

    transform_normal = transforms.Compose([       # Data preprocessing
        transforms.ToTensor(),           # Add data augmentation here
        transforms.Normalize((0.1307,), (0.3081,))
    ])


    transform_augmented = transforms.Compose([
    transforms.RandomRotation((10), fill=(0,)),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

    # Pytorch has default MNIST dataloader which loads data at each iteration
    dataset_augment = datasets.MNIST('../data', train=True, download=True,
                transform=transform_augmented)
    dataset_normal = datasets.MNIST('../data', train=True, download=True,
               transform=transform_normal)
    dataset_size = len(dataset_augment)

    np.random.seed(2020)
    # You can assign indices for training/validation or use a random subset for
    # training by using SubsetRandomSampler. Right now the train and validation
    # sets are built from the same indices - this is bad! Change it so that
    # the training and validation sets are disjoint and have the correct relative sizes.
    ratio = 8.0 / 16.0
    trainset_size = np.round(ratio * dataset_size)
    train_indices = np.random.choice(dataset_size, int(trainset_size), replace=False)
    entire_set = range(dataset_size)
    valid_indices_set = set(entire_set).difference(set(train_indices))

    subset_indices_train = train_indices.tolist()
    subset_indices_valid = list(valid_indices_set)

    train_loader = torch.utils.data.DataLoader(
        dataset_augment, batch_size=args.batch_size,
        sampler=SubsetRandomSampler(subset_indices_train)
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_normal, batch_size=args.test_batch_size,
        sampler=SubsetRandomSampler(subset_indices_valid)
    )

    # Load your model [fcNet, ConvNet, Net]
    model = Net().to(device)

    # Try different optimzers here [Adam, SGD, RMSprop]
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # Set your learning rate scheduler
    scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gamma)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, val_loader)

        train_loader_test = torch.utils.data.DataLoader(
            dataset_normal, batch_size=args.batch_size,
            sampler=SubsetRandomSampler(range(dataset_size))
        )
        trainloss, trainratio = calculate_train(model, device, train_loader_test)
        print('\nTrain set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
            trainloss, trainratio))
        scheduler.step()    # learning rate scheduler

    # You may optionally save your model at each epoch here

    if args.save_model:
        torch.save(model.state_dict(), "mnist_model.pt")


if __name__ == '__main__':
    main()
