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
