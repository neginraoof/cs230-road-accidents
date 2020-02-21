import torch
from sklearn.metrics import r2_score
import torch.nn.functional as F


def evaluate(model, device, optimizer, test_loader):
    model.eval()
    test_loss = 0
    y_s = []
    y_preds = []
    # Iterate over test data batches
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device=device, dtype=torch.float32), y.to(device=device, dtype=torch.float32).view(-1, 1)
            # Forward pass on test data batch
            y_pred = model(X)

            # Calculate MSE Loss
            loss = F.mse_loss(y_pred, y)
            test_loss += loss.item()

            y_s.extend(y)
            y_preds.extend(y_pred)
            print("Next Batch ...")

    test_loss /= len(test_loader.dataset)

    # Compute test accuracy
    y_s = torch.stack(y_s, dim=0)
    y_preds = torch.stack(y_preds, dim=0)
    test_score = r2_score(y_s.cpu().data.squeeze().numpy(), y_preds.cpu().data.squeeze().numpy())

    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(y_s), test_loss,
                                                                                        100 * test_score))

    return test_loss, test_score
