device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.ToTensor()

trainset = torchvision.datasets.MNIST(
    root=DATA_DIR, train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True)
testset = torchvision.datasets.MNIST(
    root=DATA_DIR, train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
