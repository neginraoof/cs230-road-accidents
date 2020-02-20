import torchvision
from model import *
from datasets import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import transforms as T
from train import train_one_epoch
from evaluate import evaluate
import pandas as pd


# set path
data_path = "./jpegs_256/"  # define UCF-101 spatial data path
action_name_path = "./UCF101actions.pkl"  # load preprocessed action names
save_model_path = "./Conv3D_ckpt/"  # save Pytorch models

# 3D CNN parameters
fc_hidden1, fc_hidden2 = 256, 256
dropout = 0.0  # dropout probability
k = 1  # number of target category

# Dataloader parameters
batch_size = 30
image_height, image_width = 256, 342  # resize video 2d frame size
n_frames = 30


# Select which frame to begin & end in videos
begin_frame, end_frame, skip_frame = 1, 17, 1


# Detect devices
use_cuda = torch.cuda.is_available()  # check if GPU exists
if use_cuda:
    print("============== USING CUDA ==============")
    params = {'batch_size': 1, 'shuffle': True, 'num_workers': torch.cuda.device_count(), 'pin_memory': True}
    device = torch.device("cuda")  # use CPU or GPU
else:
    print("============== USING CPU ==============")
    device = torch.device("cpu")
    params = {'batch_size': 1, 'shuffle': True, 'pin_memory': True}

df = pd.read_csv('merged_videos_labels.csv')
video_ids = df['Video ID'].to_numpy()
video_ids = video_ids[:100]
labels = df['Final Label'].to_numpy()
labels = labels[:100]

# TODO: remove these two lines to get all videos from CSV file
video_ids = os.listdir("./video_data")
video_ids = [vid_id.replace('.avi', '') for vid_id in video_ids]
data_subset = df.loc[df['Video ID'].isin(video_ids)]
video_ids = data_subset['Video ID'].to_numpy()
labels = data_subset['Final Label'].to_numpy()

# train, test split
train_list, test_list, train_label, test_label = train_test_split(video_ids, labels, test_size=0.25, random_state=42)

selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

spatial_transform_train = torchvision.transforms.Compose([
    T.ToFloatTensorInZeroOne(),
    T.Resize((image_height, image_width)),
    T.RandomHorizontalFlip(),
    T.Normalize(mean=[0.5],
                std=[0.5])
    # T.RandomCrop((112, 112))
])

spatial_transform_test = torchvision.transforms.Compose([
    T.ToFloatTensorInZeroOne(),
    T.Resize((image_height, image_width)),
    T.Normalize(mean=[0.5],
                std=[0.5])
    # T.CenterCrop((112, 112))
])

train_set = MyVideoDataset('./video_data', train_list, train_label, n_frames=n_frames, spatial_transform=spatial_transform_train)
valid_set = MyVideoDataset('./video_data', test_list, test_label, n_frames=n_frames, spatial_transform=spatial_transform_test)


train_loader = data.DataLoader(train_set, **params)
valid_loader = data.DataLoader(valid_set, **params)

# create model
model = CNN3D(image_t_frames=n_frames, image_height=image_height, image_width=image_width,
              drop_p=dropout, fc_hidden1=fc_hidden1, fc_hidden2=fc_hidden2, num_classes=k).to(device)

# Parallelize model to multiple GPUs
if torch.cuda.device_count() > 1:
    print("============== USING", torch.cuda.device_count(), "GPUs ==============")
    model = nn.DataParallel(model)

# training parameters
epochs = 15
learning_rate = 1e-4
log_interval = 10



optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # optimize all cnn parameters

# record training process
epoch_train_losses = []
epoch_train_scores = []
epoch_test_losses = []
epoch_test_scores = []

# start training
for epoch in range(epochs):
    # train, test model
    train_losses, train_scores = train_one_epoch(log_interval, model, device, train_loader, optimizer, epoch)
    epoch_test_loss, epoch_test_score = evaluate(model, device, optimizer, valid_loader)

    # save results
    epoch_train_losses.append(train_losses)
    epoch_train_scores.append(train_scores)
    epoch_test_losses.append(epoch_test_loss)
    epoch_test_scores.append(epoch_test_score)

    # save all train test results
    A = np.array(epoch_train_losses)
    B = np.array(epoch_train_scores)
    C = np.array(epoch_test_losses)
    D = np.array(epoch_test_scores)
    np.save('./3DCNN_epoch_training_losses.npy', A)
    np.save('./3DCNN_epoch_training_scores.npy', B)
    np.save('./3DCNN_epoch_test_loss.npy', C)
    np.save('./3DCNN_epoch_test_score.npy', D)

# # plot
# fig = plt.figure(figsize=(10, 4))
# plt.subplot(121)
# plt.plot(np.arange(1, epochs + 1), A[:, -1])  # train loss (on epoch end)
# plt.plot(np.arange(1, epochs + 1), C)         #  test loss (on epoch end)
# plt.title("model loss")
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.legend(['train', 'test'], loc="upper left")
# # 2nd figure
# plt.subplot(122)
# plt.plot(np.arange(1, epochs + 1), B[:, -1])  # train accuracy (on epoch end)
# plt.plot(np.arange(1, epochs + 1), D)         #  test accuracy (on epoch end)
# # plt.plot(histories.losses_val)
# plt.title("training scores")
# plt.xlabel('epochs')
# plt.ylabel('accuracy')
# plt.legend(['train', 'test'], loc="upper left")
# title = "./fig_UCF101_3DCNN.png"
# plt.savefig(title, dpi=600)
# # plt.close(fig)
# plt.show()
