import torch
from sklearn.metrics import r2_score, f1_score
import torch.nn.functional as F
import numpy as np

def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    # pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()

    return n_correct_elems / batch_size

import csv
def evaluate(model, device, optimizer, test_loader):
    model.eval() 
    test_loss = 0
    scores = []
    y_s = []
    clip_ids = []
    probs = []
    video_ids = []
    y_preds = []
    criteration = torch.nn.CrossEntropyLoss()
    criteration = torch.nn.BCEWithLogitsLoss()

    N_count = 0
    # Iterate over test data batches
    with torch.no_grad():
        for clip_id, X, y, video_id in test_loader:
            X, y = X.to(device=device, dtype=torch.float32), y.to(device=device, dtype=torch.float32)
            # Forward pass on test data batch
            y_pred = model(X)

            clip_ids.append(clip_id.numpy())
            video_ids.append(video_id.numpy())
            #print("preeeeeed", torch.nn.Softmax(dim=1)(y_pred))
            probs.append(torch.nn.Softmax(dim=1)(y_pred).detach().cpu().numpy())

            # Calculate MSE Loss
            loss = criteration(y_pred, y)
            test_loss += loss.item()

            acc1 = calculate_accuracy(y_pred, y)
            scores.append(acc1)

            y_s.extend(y)
            y_preds.extend(y_pred)

            N_count += X.size(0)
            print("Next Batch ... ", N_count, " from ", len(test_loader.dataset))

    test_loss /= len(test_loader.dataset)

    clip_ids = np.concatenate(clip_ids).reshape(-1, 1)
    video_ids = np.concatenate(video_ids).reshape(-1, 1)
    probs = np.concatenate(probs)
    data = np.concatenate([clip_ids, video_ids, probs], axis=1)

    with open('test.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(["clip_id", "video_id", "p1", "p2", "p3", "p4"])
        csv_writer.writerows(data)
        csv_writer.writerow(["Losses"])
        csv_writer.writerow([test_loss])
        csv_writer.writerow(["Scores"])
        csv_writer.writerow(scores)


    # Compute test accuracy
    y_s = torch.stack(y_s, dim=0)
    y_preds = torch.stack(y_preds, dim=0)
    test_score = f1_score(y_s.cpu().detach(), y_preds.cpu().detach().numpy() > 0.5, average="samples")
    #test_score = calculate_accuracy(y_s.cpu().data.squeeze().numpy(), y_preds.cpu().data.squeeze().numpy())

    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(y_s), test_loss,
                                                                                        100 * test_score))

    return test_loss, test_score
