import torch
from sklearn.metrics import r2_score
import torch.nn.functional as F


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()

    return n_correct_elems / batch_size


def evaluate(model, device, optimizer, test_loader):
    model.eval()
    test_loss = 0
    y_s = []
    y_preds = []
    criteration = torch.nn.CrossEntropyLoss()

    # Iterate over test data batches
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device=device, dtype=torch.float32), y.to(device=device, dtype=torch.int64)
            # Forward pass on test data batch
            y_pred = model(X)

            # Calculate MSE Loss
            loss = criteration(y_pred, y)
            test_loss += loss.item()

            y_s.extend(y)
            y_preds.extend(y_pred)
            print("Next Batch ...")

    test_loss /= len(test_loader.dataset)

    # Compute test accuracy
    y_s = torch.stack(y_s, dim=0)
    y_preds = torch.stack(y_preds, dim=0)
    test_score = calculate_accuracy(y_s.cpu().data.squeeze().numpy(), y_preds.cpu().data.squeeze().numpy())

    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(y_s), test_loss,
                                                                                        100 * test_score))

    return test_loss, test_score
