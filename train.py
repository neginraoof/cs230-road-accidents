from sklearn.metrics import r2_score
import torch
import numpy
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.data as data
import torchvision
# import matplotlib.pyplot as plt


log_interval = 10


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()

    return n_correct_elems / batch_size


# def accuracy(output, target, topk=(1,)):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)
#
#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target[None])
#
#         res = []
#         for k in topk:
#             correct_k = correct[:k].flatten().sum(dtype=torch.float32)
#             res.append(correct_k * (100.0 / batch_size))
#         return res


def train_one_epoch(model, device, train_loader, optimizer, epoch):
    model.train()

    losses = []
    scores = []
    N_count = 0
    criteration = torch.nn.CrossEntropyLoss()

    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(device=device, dtype=torch.float32), y.to(device=device, dtype=torch.int64)
        N_count += X.size(0)

        # Initialize optimizer
        optimizer.zero_grad()
        # Forward pass
        y_pred = model(X)

        # Calculate batch loss
        loss = criteration(y_pred, y)
        losses.append(loss.item())

        # Calculate accuracy score
        # step_score = r2_score(y.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
        acc1 = calculate_accuracy(y_pred, y)
        scores.append(acc1)

        # backprop
        loss.backward()
        # optimize model params
        optimizer.step()

        # show information
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy Score: {:.2f}'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(),
                100 * numpy.mean(scores)))
    
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, 'last.pth')
    
    return losses, scores
