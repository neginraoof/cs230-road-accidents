import csv
import numpy as np
from train import dir_name
from utils import TripleCrossEntropy, TripleBinaryCrossEntropy
from model import *

def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()

    return n_correct_elems / batch_size


def evaluate(model, device, test_loader):
    model.eval() 
    losses = []
    
    video_ids = []
    clip_ids = []
    probs = []

    scores = []
    y_s = []
    labels = []
    criteration = torch.nn.CrossEntropyLoss()

    N_count = 0
    # Iterate over test data batches
    with torch.no_grad():
        for batch_idx, (clip_id, X, y, video_id) in enumerate(test_loader):
            X, y = X.to(device=device, dtype=torch.float32), y.to(device=device, dtype=torch.float32)
            # Forward pass on test data batch
            y_pred = model(X)

            clip_ids.append(clip_id.numpy())
            video_ids.append(video_id.numpy())

            if isinstance(model, OrdinalModelPretrained):
                loss = TripleBinaryCrossEntropy(y_pred, y)
            else:
                loss = criteration(y_pred, y)
                # Calculate accuracy score
                acc1 = calculate_accuracy(y_pred, y)
                scores.append(acc1)
            losses.append(loss.item())

            labels.append(y)

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

            N_count += X.size(0)
            print("Next Batch ... ", N_count, " from ", len(test_loader.dataset))

    clip_ids = np.concatenate(clip_ids).reshape(-1, 1)
    video_ids = np.concatenate(video_ids).reshape(-1, 1)
    probs = np.concatenate(probs)
    labels = np.concatenate(labels).reshape(clip_ids.shape[0], -1)
    data = np.concatenate([clip_ids, video_ids, probs, labels], axis=1)

    with open(dir_name + '/test.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(["clip_id", "video_id", "p1", "p2", "p3", "p4", "labels"])
        csv_writer.writerows(data)
        csv_writer.writerow(["Test Losses"])
        csv_writer.writerow(losses)
        csv_writer.writerow(["Scores"])
        csv_writer.writerow(scores)

    # Compute test accuracy
    import numpy
    test_score = numpy.mean(scores)

    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(labels), np.mean(losses),
                                                                                        100 * test_score))

    return losses, test_score
