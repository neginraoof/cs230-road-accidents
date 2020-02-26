import torchvision
from model import *
from datasets import *
from train import *
from evaluate import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import transforms as T
import pandas as pd
import numpy as np



#Test write 
np.save('dummy_test.npy', np.array([2]))

# Dataloader parameters
batch_size = 15
image_height, image_width = 256, 342  # resize video 2d frame size
n_frames = 20  #number of frames in a video clip
num_classes = 4
categories = [1, 2, 3, 4]


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

# Transform labels to categories
labels = np.rint(labels)
danger_category = np.asarray(categories).reshape(-1,)
label_encoder = LabelEncoder()
# print(danger_category)
label_encoder.fit(danger_category)
label_cats = label_encoder.transform(labels.reshape(-1,))

# train, test split
train_list, test_list, train_label, test_label = train_test_split(video_ids, label_cats, test_size=0.25, random_state=42)

spatial_transform_train = torchvision.transforms.Compose([
    T.ToFloatTensorInZeroOne(),
    T.Resize((image_height, image_width)),
    T.RandomHorizontalFlip(),
    T.Normalize(mean=[0.5],
                std=[0.8])
    # T.RandomCrop((112, 112))
])

spatial_transform_test = torchvision.transforms.Compose([
    T.ToFloatTensorInZeroOne(),
    T.Resize((image_height, image_width)),
    T.Normalize(mean=[0.5],
                std=[0.8])
    # T.CenterCrop((112, 112))
])


print("============== Loading Data ==============")
train_set = MyVideoDataset('./video_data', train_list, train_label, n_frames=n_frames, spatial_transform=spatial_transform_train)
valid_set = MyVideoDataset('./video_data', test_list, test_label, n_frames=n_frames, spatial_transform=spatial_transform_test)


train_loader = data.DataLoader(train_set, **params)
valid_loader = data.DataLoader(valid_set, **params)

# create model
model = Conv3dModel(image_t_frames=n_frames, image_height=image_height, image_width=image_width, num_classes=num_classes).to(device)

# Parallelize model to multiple GPUs
if torch.cuda.device_count() > 1:
    print("============== USING", torch.cuda.device_count(), "GPUs ==============")
    model = nn.DataParallel(model)

# training parameters
epochs = 15
learning_rate = 1e-5

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

    np.save('./3DCNN_epoch_training_losses.npy', np.array(epoch_train_losses))
    np.save('./3DCNN_epoch_training_scores.npy', np.array(epoch_train_losses))
    np.save('./3DCNN_epoch_test_loss.npy', np.array(epoch_test_losses))
    np.save('./3DCNN_epoch_test_score.npy', np.array(epoch_test_scores))

