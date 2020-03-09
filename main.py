import torchvision
from model import *
from datasets import *
from train import *
from evaluate import *
import transforms as T
import pandas as pd
import numpy as np
from utils import *

args = parser.parse_args()
print("Args: ", args)

#Test write 
np.save('dummy_test.npy', np.array([2]))

# Dataloader parameters
batch_size = 20
image_height, image_width = 250, 350  # resize video 2d frame size
n_frames = 15  #number of frames in a video clip
fps = 1
num_classes = 4
categories = [0, 1, 2, 3]

# Detect devices
use_cuda = torch.cuda.is_available()  # check if GPU exists
if use_cuda:
    print("============== USING CUDA ==============")
    params = {'batch_size': batch_size, 'shuffle': True, 'pin_memory': True}
    device = torch.device("cuda")  # use CPU or GPU
else:
    print("============== USING CPU ==============")
    device = torch.device("cpu")
    params = {'batch_size': batch_size, 'shuffle': True, 'pin_memory': True}


train_list, train_label = read_data_labels('train1.csv', categories)
test_list, test_label = read_data_labels('test1.csv', categories)

if args.crop_videos:
    crop_video(train_list)
    crop_video(test_list)

spatial_transform_train = torchvision.transforms.Compose([
    T.ToFloatTensorInZeroOne(),
    T.Resize((image_height, image_width)),
    T.RandomHorizontalFlip(),
    # Normalization done after data is loaded
    # T.RandomCrop((112, 112))
])

spatial_transform_test = torchvision.transforms.Compose([
    T.ToFloatTensorInZeroOne(),
    T.Resize((image_height, image_width)),
    # Normalization done after data is loaded
    # T.CenterCrop((112, 112))
])

print("============== Loading Data ==============")
print("Train {} clips".format(len(train_list)))
print("Test {} clips".format(len(test_list)))
train_set = MyVideoDataset('./video_data_clip', train_list, train_label, n_frames=n_frames, fps=fps, spatial_transform=spatial_transform_train)
valid_set = MyVideoDataset('./video_data_clip', test_list, test_label, n_frames=n_frames, fps=fps, spatial_transform=spatial_transform_test)

train_loader = data.DataLoader(train_set, **params)
valid_loader = data.DataLoader(valid_set, **params)

#  Normalize Data
if args.get_stats:
    m_, s_ = get_stats(train_loader)
    print("Calculated stats: mean ", m_ , "and std ", s_)
else:
    m_ = torch.tensor([0.5707, 0.5650, 0.5351])
    s_ = torch.tensor([0.1882, 0.1890, 0.2004])
train_set.set_stats(m_, s_)
valid_set.set_stats(m_, s_)

# create model
if args.pretrained:
    model = ResNet18(num_classes=4).to(device)
else:
    model = Conv3dModel(image_t_frames=n_frames, image_height=image_height, image_width=image_width, num_classes=num_classes).to(device)
print("Model: ", model)


# Parallelize model to multiple GPUs
if torch.cuda.device_count() > 1:
    print("============== USING", torch.cuda.device_count(), "GPUs ==============")
    model = nn.DataParallel(model)

# training parameters
epochs = 10
learning_rate = 1e-4

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # optimize all cnn parameters

# record training process
epoch_train_losses = []
epoch_train_scores = []
epoch_test_losses = []
epoch_test_scores = []

# start training
for epoch in range(epochs):
    # train, test model
    train_losses, train_scores = train_one_epoch(model, device, train_loader, optimizer, epoch)
    epoch_test_loss, epoch_test_score = evaluate(model, device, optimizer, valid_loader)

    # save results
    epoch_train_losses.append(train_losses)
    epoch_train_scores.append(train_scores)
    epoch_test_losses.append(epoch_test_loss)
    epoch_test_scores.append(epoch_test_score)

    print("Train losses: ", np.array(epoch_train_losses))
    print("Train scores: ", np.array(epoch_train_scores))
    print("Test losses: ", np.array(epoch_test_losses))
    print("Test scores: ", np.array(epoch_test_scores))

#    np.save('./3DCNN_epoch_{}_training_losses.npy'.format(epoch), np.array(epoch_train_losses))
#    np.save('./3DCNN_epoch_{}_training_scores.npy'.format(epoch), np.array(epoch_train_losses))
#    np.save('./3DCNN_epoch_{}_test_loss.npy'.format(epoch), np.array(epoch_test_losses))
#    np.save('./3DCNN_epoch_{}_test_score.npy'.format(epoch), np.array(epoch_test_scores))
