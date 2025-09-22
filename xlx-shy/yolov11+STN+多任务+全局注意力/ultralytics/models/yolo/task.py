class HydraNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.resnet18(pretrained=True)
        self.n_features = self.net.fc.in_features
        self.net.fc = nn.Identity()

        self.net.fc1 = nn.Sequential(OrderedDict(
            [('linear', nn.Linear(self.n_features, self.n_features)),
             ('relu1', nn.ReLU()),
             ('final', nn.Linear(self.n_features, 1))]))

        self.net.fc2 = nn.Sequential(OrderedDict(
            [('linear', nn.Linear(self.n_features, self.n_features)),
             ('relu1', nn.ReLU()),
             ('final', nn.Linear(self.n_features, 1))]))

        self.net.fc3 = nn.Sequential(OrderedDict(
            [('linear', nn.Linear(self.n_features, self.n_features)),
             ('relu1', nn.ReLU()),
             ('final', nn.Linear(self.n_features, 5))]))

    def forward(self, x):
        age_head = self.net.fc1(self.net(x))
        gender_head = self.net.fc2(self.net(x))
        ethnicity_head = self.net.fc3(self.net(x))
        return age_head, gender_head, ethnicity_head