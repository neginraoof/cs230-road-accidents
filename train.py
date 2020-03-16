# import matplotlib.pyplot as plt
import csv
import os
from utils import TripleCrossEntropy, TripleBinaryCrossEntropy
from model import *


log_interval = 10
dir_name = 'ordinal_model_10fps'
if not os.path.exists(dir_name):
     os.mkdir(dir_name)


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)
    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()
    return n_correct_elems / batch_size


def calculate_ordinal_accuracy(outputs, targets):
    batch_size = targets.size(0)
    pred = torch.round(outputs)
    correct = pred.eq(targets)
    n_correct_elems = correct.float().sum().item()
    return n_correct_elems / 3 * batch_size


def train_one_epoch(model, device, train_loader, optimizer, epoch):
    model.train()

    video_ids = []
    clip_ids = []
    probs = []
    losses = []
    scores = []
    labels = []
    N_count = 0
    criteration = torch.nn.CrossEntropyLoss()
    for batch_idx, (clip_id, X, y, video_id) in enumerate(train_loader):
        X, y = X.to(device=device, dtype=torch.float32), y.to(device=device, dtype=torch.float32)
        N_count += X.size(0)
        # Initialize optimizer
        optimizer.zero_grad()
        # Forward pass
        y_pred = model(X)
        clip_ids.append(clip_id.numpy())
        video_ids.append(video_id.numpy())

        # Calculate batch loss
        if isinstance(model, OrdinalModelPretrained):
            loss = TripleBinaryCrossEntropy(y_pred, y)
            acc1 = calculate_ordinal_accuracy(y_pred, y)
        else:
            loss = criteration(y_pred, y)
            acc1 = calculate_accuracy(y_pred, y)
        losses.append(loss.item())
        scores.append(acc1)
        labels.append(y)
        # Back-propagation
        loss.backward()
        optimizer.step()

        if isinstance(model, OrdinalModelPretrained):
            # y_pred shape: [N, 3]
            p1 = y_pred[:, 0]
            p2 = y_pred[:, 1] - y_pred[:, 0]
            p3 = y_pred[:, 2] - y_pred[:, 1]
            p4 = 1 - y_pred[:, 2]
            prob = torch.stack([p1, p2, p3, p4], dim=1).detach().cpu().numpy()
            probs.append(prob)
        else:
            probs.append(torch.nn.Softmax(dim=1)(y_pred).detach().cpu().numpy())

        # show information
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy Score: {:.2f}'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * np.mean(scores)))

    clip_ids = np.concatenate(clip_ids).reshape(-1, 1)
    video_ids = np.concatenate(video_ids).reshape(-1, 1)
    probs = np.concatenate(probs)
    labels = np.concatenate(labels).reshape(clip_ids.shape[0], -1)
    data = np.concatenate([clip_ids, video_ids, probs, labels], axis=1)

    with open(dir_name+'/train_epoch_{}.csv'.format(epoch), 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(["clip_id", "video_id", "p1", "p2", "p3", "p4", "labels"])
        csv_writer.writerows(data)
        csv_writer.writerow(["Train Losses"])
        csv_writer.writerow(losses)
        csv_writer.writerow(["Scores"])
        csv_writer.writerow(scores)
 
    # Save model state
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, 'last.pth')
    
    return losses, scores
