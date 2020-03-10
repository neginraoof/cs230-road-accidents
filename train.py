from sklearn.metrics import r2_score, f1_score
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
    # criteration = torch.nn.CrossEntropyLoss()

    clip_ids = []
    video_ids = []
    probs = []

    vid_id_mapping = dict()

    for batch_idx, (clip_id, X, y, video_id) in enumerate(train_loader):
        X, y = X.to(device=device, dtype=torch.float32), y.to(device=device, dtype=torch.float32)
        N_count += X.size(0)

        # Initialize optimizer
        optimizer.zero_grad()
        # Forward pass
        y_pred = model(X)

        # Calculate batch loss
        y_pred = y_pred.squeeze(dim=-1)
        loss = criteration(y_pred, y)
        losses.append(loss.item())
        # print("loss ", loss)

        print("clip id", clip_id)
        clip_ids.append(clip_id.numpy())
        print("video_id", video_id)
        video_ids.append(video_id.numpy())

        probs.append(torch.nn.Softmax(dim=1)(y_pred).detach().cpu().numpy())

        # Calculate accuracy score
        acc1 = f1_score(y.cpu().detach(), y_pred.cpu().detach().numpy() > 0.5, average="samples")
        # step_score = r2_score(y.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
        # acc1 = calculate_accuracy(y_pred, y)
        scores.append(acc1)
        print("Acc ", acc1)

        # backprop
        loss.backward()
        # optimize model params
        optimizer.step()

        # show information
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy Score: {:.2f}'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(),
                100 * np.mean(scores)))
    import csv

    clip_ids = np.concatenate(clip_ids).reshape(-1, 1)

    video_ids = np.concatenate(video_ids).reshape(-1, 1)

    probs = np.concatenate(probs)

    data = np.concatenate([clip_ids, video_ids, probs], axis=1)

    # np.savetxt('array_hf.csv', [arr], delimiter=',', fmt='%d', header='A Sample 2D Numpy Array :: Header',
    #            footer='This is footer')
    with open('train_epoch_{}.csv'.format(epoch), 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(["clip_id", "video_id", "p1", "p2", "p3", "p4"])
        csv_writer.writerows(data)
        csv_writer.writerow(["Losses"])
        csv_writer.writerow(losses)
        csv_writer.writerow(["Scores"])
        csv_writer.writerow(scores)

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
