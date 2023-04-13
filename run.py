#%%
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
import torchvision.transforms as transforms

from tqdm import tqdm
from module.model.resnet import resnet18


def build_datasets(data_root, batch_size, transform):
    trainset = torchvision.datasets.CIFAR10(root=data_root,
                                            train=True,
                                            download=True,
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root=data_root, train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader, classes


def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()


def fix_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.nn.Conv2d(3,1,1,).weight
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def select_device():
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else: 
        device = "cpu"
    
    return device


#%% config
batch_size = 4
data_root = "/Users/wonhyung64/data"
dataset = "cifar-10"
epochs = 10
model_root = "./assets/model_weights"
model_seed = 0
data_seed = 0


# %%

fix_seed(data_seed)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainloader, testloader, classes = build_datasets(data_root, batch_size, transform)

augment = transforms.Compose(
    [transforms.Resize((4, 4)),
     transforms.Resize((32, 32)),]
)

augment2 = transforms.Compose(
    [transforms.Resize((4, 4)),]
)

fix_seed(model_seed)
net = resnet18(num_classes=10, feature_output=True)

cls_loss_fn = nn.CrossEntropyLoss()
feature_loss_fn = nn.MSELoss()

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

device = select_device()

net.to(device)

#%%
base_accuracy = 0
for epoch in range(epochs):
    progress_bar = tqdm(enumerate(trainloader, 0))
    for i, data in progress_bar:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs, (feat1, feat2, feat3, feat4) = net(inputs)
        _, (perturbed_feat1, perturbed_feat2, perturbed_feat3, perturbed_feat4) = net(augment(inputs))

        cls_loss = cls_loss_fn(outputs, labels)
        # feat1_loss = feature_loss_fn(perturbed_feat1, feat1)
        # feat2_loss = feature_loss_fn(perturbed_feat2, feat2)
        # feat3_loss = feature_loss_fn(perturbed_feat3, feat3)
        # feat4_loss = feature_loss_fn(perturbed_feat4, feat4)
        # total_loss = cls_loss + feat1_loss + feat2_loss + feat3_loss + feat4_loss
        total_loss = cls_loss

        total_loss.backward()
        optimizer.step()

        # progress_bar.set_description(
        #     f"Epochs:{epoch+1}/{epochs} | cls:{cls_loss.item():.3f} feat1:{feat1_loss.item():.3f} feat2:{feat2_loss.item():.3f} feat3:{feat3_loss.item():.3f} feat4:{feat4_loss.item():.3f} total:{total_loss.item():.3f}"
        #     )
        progress_bar.set_description(
            f"Epochs:{epoch+1}/{epochs} | cls:{cls_loss.item():.3f}"
            )

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs, _ = net(images)
            _, predictions = torch.max(outputs, 1)

            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    accuracy = {}
    for classname, correct_count in correct_pred.items():
        accuracy[classname] = float(correct_count) / total_pred[classname]

    for classname, acc in accuracy.items(): print(f'Accuracy for class: {classname:5s} is {acc*100:.3f} %')

    epoch_accuracy = np.mean(list(accuracy.values()))

    if epoch_accuracy > base_accuracy:
        print("performance improved!")
        torch.save(net.state_dict(), f"{model_root}/cifar10_resnet18.pth")
        base_accuracy = epoch_accuracy
        
# torch.save(net.state_dict(), f"{model_root}/cifar10_resnet18_s.pth")
print("Finished Training")


#%%
net = resnet18(num_classes=10, feature_output=False)
net.load_state_dict(torch.load(f"{model_root}/cifar10_resnet18.pth"))


#%%
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)

        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


accuracy = {}
for classname, correct_count in correct_pred.items():
    accuracy[classname] = float(correct_count) / total_pred[classname]

for classname, acc in accuracy.items(): print(f'Accuracy for class: {classname:5s} is {acc*100:.3f} %')


#%%
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(augment(images))
        _, predictions = torch.max(outputs, 1)

        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


accuracy = {}
for classname, correct_count in correct_pred.items():
    accuracy[classname] = float(correct_count) / total_pred[classname]

for classname, acc in accuracy.items(): print(f'Accuracy for class: {classname:5s} is {acc*100:.3f} %')

print(np.mean(list(accuracy.values())))

# %%
