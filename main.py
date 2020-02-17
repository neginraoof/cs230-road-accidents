import torchvision.transforms as transforms
import torchvision
from datasets import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pickle
import transforms as T
from train import train_one_epoch
from evaluate import evaluate

# set path
data_path = "./jpegs_256/"  # define UCF-101 spatial data path
action_name_path = "./UCF101actions.pkl"  # load preprocessed action names
save_model_path = "./Conv3D_ckpt/"  # save Pytorch models

# 3D CNN parameters
fc_hidden1, fc_hidden2 = 256, 256
dropout = 0.0  # dropout probability
k = 101  # number of target category

# Dataloader parameters
batch_size = 30
img_x, img_y = 256, 342  # resize video 2d frame size
n_frames = 30


# Select which frame to begin & end in videos
begin_frame, end_frame, skip_frame = 1, 17, 1


# Detect devices
use_cuda = torch.cuda.is_available()  # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU

params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}

# load UCF101 actions names
with open(action_name_path, 'rb') as f:
    action_names = pickle.load(f)  # load UCF101 actions names

# convert labels -> category
le = LabelEncoder()
le.fit(action_names)

# show how many classes there are
list(le.classes_)

# convert category -> 1-hot
action_category = le.transform(action_names).reshape(-1, 1)
enc = OneHotEncoder()
enc.fit(action_category)

# # example
# y = ['HorseRace', 'YoYo', 'WalkingWithDog']
# y_onehot = labels2onehot(enc, le, y)
# y2 = onehot2labels(le, y_onehot)

actions = []
fnames = os.listdir(data_path)

all_names = []
for f in fnames:
    loc1 = f.find('v_')
    loc2 = f.find('_g')
    actions.append(f)

    all_names.append(f)

# list all data files
all_X_list = all_names[1:4]  # all video file names
all_y_list = labels2cat(le, actions)[1:4]  # all video labels

# train, test split
train_list, test_list, train_label, test_label = train_test_split(all_X_list, all_y_list, test_size=0.25, random_state= 42)

# image transformation
transform = transforms.Compose([transforms.Resize([img_x, img_y]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5], std=[0.5])])

selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()


spatial_transform_train = torchvision.transforms.Compose([
    T.ToFloatTensorInZeroOne(),
    T.Resize((img_x, img_y)),
    T.RandomHorizontalFlip(),
    T.Normalize(mean=[0.43216, 0.394666, 0.37645],
                std=[0.22803, 0.22145, 0.216989])
    # T.RandomCrop((112, 112))
])

spatial_transform_test = torchvision.transforms.Compose([
    T.ToFloatTensorInZeroOne(),
    T.Resize((img_x, img_y)),
    T.Normalize(mean=[0.43216, 0.394666, 0.37645],
                std=[0.22803, 0.22145, 0.216989])
    # T.CenterCrop((112, 112))
])

train_set = MyVideoDataset('./jpegs_256', train_list, train_label, n_frames=n_frames, spatial_transform=spatial_transform_train)
valid_set = MyVideoDataset('./jpegs_256', test_list, test_label, n_frames=n_frames, spatial_transform=spatial_transform_test)


train_loader = data.DataLoader(train_set, **params)
valid_loader = data.DataLoader(valid_set, **params)

# create model
model = CNN3D(t_dim=n_frames, img_x=img_x, img_y=img_y,
              drop_p=dropout, fc_hidden1=fc_hidden1, fc_hidden2=fc_hidden2, num_classes=k).to(device)

# Parallelize model to multiple GPUs
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
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