import random

import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split

from models.lrae import LRAENetwork


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# produces (0.13066, 0.30811)
def calculate_mnist_mean_std():
    data_train = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=torchvision.transforms.ToTensor())
    mean_train = torch.mean(data_train.data.float()) / 255.
    std_train = torch.std(data_train.data.float()) / 255.
    return (mean_train.numpy(), std_train.numpy())



def make_moving_collate_fn(device):
    def collate_move_to_device(batch):
        inputs, targets = zip(*batch)
        moved_inputs = torch.stack(inputs).to(device)
        moved_targets = torch.tensor(targets).to(device)
        return moved_inputs, moved_targets
    return collate_move_to_device


# from LRA-E paper:
#
# > For both datasets and all models, over 100 epochs, we cal-
# > culate updates over mini-batches of 50 samples. Furthermore,
# > we do not regularize parameters any further, e.g., drop-out
# > (Srivastava et al. 2014) or weight penalties. All feedfoward
# > architectures for all experiments were of either 3, 5, or 8 hid-
# > den layers of 256 processing elements. The post-activation
# > function used was the hyperbolic tangent and the top layer
# > was chosen to be a maximum-entropy classifier (i.e., a soft-
# > max function). The output layer objective for all algorithms
# > was to minimize the categorical negative log likelihood.

def train_mnist(num_epochs=10, batch_size=32, init_lr=0.01, max_lr=0.5, momentum=0.9):
    set_random_seed(6283185)

    num_epochs = 20
    batch_size = 25
    input_size = 28*28
    hidden_size = 256
    num_hiddens = 2
    output_size = 10

    perform_validation = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load dataset
    moving_collate = make_moving_collate_fn(device)
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.13066,), (0.30811,))])

    data_train_full = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transforms)
    train_size = int(0.8 * len(data_train_full))
    valid_size = len(data_train_full) - train_size
    data_train, data_valid = random_split(data_train_full, [train_size, valid_size])

    data_test = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transforms)

    # train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, collate_fn=moving_collate)
    valid_loader = torch.utils.data.DataLoader(data_valid, batch_size=batch_size, collate_fn=moving_collate)

    data_train_subset = torch.utils.data.Subset(data_train, range(64))
    train_loader = torch.utils.data.DataLoader(data_train_subset, batch_size=batch_size, shuffle=True, collate_fn=moving_collate)

    model = LRAENetwork(input_size, output_size, num_hiddens, hidden_size)
    model.to(device)

    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=momentum)
    # oc_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=len(train_loader), epochs=num_epochs)

    print(f'Train examples: {len(data_train)}')
    print(f'Test examples: {len(data_test)}')

    def calc_validation_accuracy():
        with torch.no_grad():
            model.eval()
            valid_correct = 0
            for x_mb, y_mb in valid_loader:
                x_mb = x_mb.view(x_mb.size(0), -1)
                logits = model(x_mb)
                valid_correct += torch.sum(torch.argmax(logits, dim=1) == y_mb)
            valid_acc = valid_correct / len(data_valid)
            print(f'Validation accuracy: {valid_acc}')
            model.train()

    # pre-training accuracy
    if perform_validation:
        calc_validation_accuracy()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        epoch_avg_loss = 0.0
        for it, (x_mb, y_mb) in enumerate(train_loader):
            optimizer.zero_grad()

            x_mb = x_mb.view(x_mb.size(0), -1)
            logits = model(x_mb)
            loss = F.cross_entropy(logits, y_mb)
            epoch_avg_loss = (epoch_avg_loss * it + loss.item()) / (it + 1)

            model.backward(x_mb, y_mb)

            optimizer.step()
            # oc_scheduler.step()

        print(f'Epoch average loss: {epoch_avg_loss}')

        if perform_validation and (epoch + 1) % 5 == 0:
            calc_validation_accuracy()



if __name__ == '__main__':
    train_mnist(num_epochs=10, init_lr = 0.01, max_lr=0.5)
