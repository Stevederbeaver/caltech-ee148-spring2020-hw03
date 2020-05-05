class Net(nn.Module):
    '''
    Build the best MNIST classifier.
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding = 1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride = 2, padding = 1)
        self.batchnm1 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 64, 3, padding = 1)
        self.conv5 = nn.Conv2d(64, 64, 3, padding = 1)
        self.conv6 = nn.Conv2d(64, 64, 3, stride = 2, padding = 1)
        self.batchnm2 = nn.BatchNorm2d(64)

        self.conv7 = nn.Conv2d(64, 128, 3, padding = 1)
        self.batchnm3 = nn.BatchNorm2d(128)

        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.batchnm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchnm1(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchnm1(x)
        x = self.maxpool(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchnm2(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.batchnm2(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.batchnm2(x)
        x = self.maxpool(x)
        x = self.dropout(x)

        x = self.conv7(x)
        x = F.relu(x)
        x = self.batchnm3(x)
        x = self.maxpool(x)
        x = self.dropout(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        output = F.log_softmax(x, dim=1)
        return output
