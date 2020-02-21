from sklearn.metrics import r2_score
import torch
import numpy
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.data as data
import torchvision
# import matplotlib.pyplot as plt


log_interval = 10


def train_one_epoch( model, device, train_loader, optimizer, epoch):
    model.train()

    losses = []
    scores = []
    N_count = 0

    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(device=device, dtype=torch.float32), y.to(device=device, dtype=torch.float32).view(-1, 1)
        N_count += X.size(0)

        # Initialize optimizer
        optimizer.zero_grad()
        # Forward pass
        y_pred = model(X)

        # Calculate batch loss
        loss = F.mse_loss(y_pred, y)
        losses.append(loss.item())

        # Calculate bacth R2 score
        step_score = r2_score(y.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
        scores.append(step_score)

        # backprop
        loss.backward()
        # optimize model params
        optimizer.step()

        # show information
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, R2 Score: {:.2f}'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(),
                100 * numpy.mean(scores)))

    return losses, scores
