from sklearn.metrics import r2_score
import torch
import numpy as np
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
    print("preeeed shape ", pred.shape)
    print("targets shape ", targets.shape)
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()
    return n_correct_elems / batch_size


def train_one_epoch(model, device, train_loader, optimizer, epoch):
    model.train()

    losses = []
    scores = []
    N_count = 0
    criteration = torch.nn.BCEWithLogitsLoss()
    # criteration = torch.nn.NLLLoss()

    vid_id_mapping = dict()

    for batch_idx, (X, y, video_id) in enumerate(train_loader):
        X, y = X.to(device=device, dtype=torch.float32), y.to(device=device, dtype=torch.float32)
        N_count += X.size(0)

        # Initialize optimizer
        optimizer.zero_grad()
        # Forward pass
        (pred, lin_1, lin_2, lin_3) = model(X)

        sigma_1 = torch.nn.Sigmoid()(lin_1)
        sigma_2 = torch.nn.Sigmoid()(lin_2)
        sigma_3 = torch.nn.Sigmoid()(lin_3)
        r_0 = sigma_1
        r_1 = (1 - sigma_1) * sigma_2
        r_2 = (1 - sigma_1) * (1 - sigma_2) * sigma_3
        r_3 = (1 - sigma_1) * (1 - sigma_2) * (1 - sigma_3)
        out = torch.stack((r_0, r_1, r_2, r_3), dim=1)

        print("stack shape ", out.shape)
        print("pred shape ", pred.shape)

        # Calculate batch loss
        # y = y.unsqueeze(dim=-1)
        # print(y_pred.shape,' and ', y.shape)
        # loss = criteration(y_pred, y)
        # losses.append(loss.item())
        # print("loooooooss ", loss)

        # Calculate accuracy score
        # step_score = r2_score(y.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
        # acc1 = calculate_accuracy(y_pred, y)
        # scores.append(acc1)

        # backprop
        # loss.backward()
        # optimize model params
        optimizer.step()

        # show information
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy Score: {:.2f}'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(),
                100 * np.mean(scores)))

        # for v_id, label, true_label in zip(video_id, y_pred, y):
        #     v_id = v_id.item()
        #     print("ID: ", v_id)
        #     if v_id in vid_id_mapping.keys():
        #         print("Id {} seen before. Prev_v {}, new_v: {}. True v: ".
        #               format(v_id, vid_id_mapping[v_id], np.argmax(label.detach().numpy()), true_label))
        #     else:
        #         print("Id {} set to {}".format(v_id, np.argmax(label.detach().numpy())))
        #         vid_id_mapping[v_id] = np.argmax(label.detach().numpy())
    
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, 'last.pth')
    
    return losses, scores
