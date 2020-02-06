import numpy as np
import os
import torch
import torchvision
import argparse
import cv2


str2bool = lambda x: (str(x).lower() == 'true')

parser = argparse.ArgumentParser(
    description='PyTorch Jester Training using JPEG')
parser.add_argument('--model_name', '-n', help='name of the model')
parser.add_argument('--config', '-c', help='json config file path')
parser.add_argument('--eval_only', '-e', default=False, type=str2bool,
                    help="evaluate trained model on validation data.")
parser.add_argument('--test_only', '-t', default=False, type=str2bool,
                    help="test the trained model on the test set.")
parser.add_argument('--resume', '-r', default=False, type=str2bool,
                    help="resume training from given checkpoint.")
parser.add_argument('--use_gpu', default=True, type=str2bool,
                    help="flag to use gpu or not.")
parser.add_argument('--gpus', '-g', help="gpu ids for use.")
parser.add_argument('--batch_size', '-bs', help="batch size")
parser.add_argument('--workers', '-w', help="num workers")
parser.add_argument('--last_lr', '-lr')
parser.add_argument('--momentum', '-m')
parser.add_argument('--weight_decay', '-wd')

args = parser.parse_args()


def load_video(video_file, channels=3, time_depth=5, x_size=240, y_size=256):
    # Open the video file
    cap = cv2.VideoCapture(video_file)
    # nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = torch.FloatTensor(channels, time_depth, x_size, y_size)
    failedClip = False
    for f in range(time_depth):

        ret, frame = cap.read()
        if ret:
            frame = torch.from_numpy(frame)
            # HWC2CHW
            frame = frame.permute(2, 0, 1)
            frames[:, f, :, :] = frame

        else:
            print("Skipped!")
            failedClip = True
            break

    # for c in range(3):
    #     frames[c] -= self.mean[c]
    frames /= 255
    return frames, failedClip

def main():
    global args #, best_prec1

    # set run output folder
    model_name = args.model_name
    print("Train output dir: ./trains-{}".format(model_name))
    save_dir = os.path.join("./train-", model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        # os.makedirs(os.path.join(save_dir, 'plots'))

    print("Validation output dir: ./validation-{}".format(model_name))
    val_dir = os.path.join("./validation-", model_name)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
        # os.makedirs(os.path.join(val_dir, 'plots'))

    # # adds a handler for Ctrl+C
    # def signal_handler(signal, frame):
    #     """
    #     Remove the output dir, if you exit with Ctrl+C and
    #     if there are less then 3 files.
    #     It prevents the noise of experimental runs.
    #     """
    #     num_files = len(glob.glob(save_dir + "/*"))
    #     if num_files < 1:
    #         shutil.rmtree(save_dir)
    #     print('You pressed Ctrl+C!')
    #     sys.exit(0)
    # # assign Ctrl+C signal handler
    # signal.signal(signal.SIGINT, signal_handler)

    # create model

    model = CNNModule()

    # Use GPUs
    device = torch.device("cpu")
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        gpus = [int(i) for i in args.gpus.split(',')]
        print("Using GPUs: {}".format(args.gpus))
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # optionally resume from a checkpoint
    # if args.resume:
    #     if os.path.isfile(config['checkpoint']):
    #         print("loading checkpoint '{}'".format(args.resume))
    #         checkpoint = torch.load(config['checkpoint'])
    #         args.start_epoch = 0
    #         best_prec1 = checkpoint['best_prec1']
    #         for key, value in checkpoint['state_dict'].items() :
    #             print (key)
    #         model.load_state_dict(checkpoint['state_dict'])
    #         print("loaded checkpoint '{}' (epoch {})"
    #               .format(config['checkpoint'], checkpoint['epoch']))
    #     else:
    #         print("no checkpoint found at '{}'".format(
    #             config['checkpoint']))

    normalize = torchvision.transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                                 std=[0.22803, 0.22145, 0.216989])

    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.ToFloatTensorInZeroOne(),
        torchvision.transforms.Resize((128, 171)),
        torchvision.transforms.RandomHorizontalFlip(),
        normalize,
        torchvision.transforms.RandomCrop((112, 112))
    ])
    
    ## Load Data and create clips:
    ## dataset_test.video_clips.compute_clips(args.clip_len, 1, frame_rate=15)  
    ## data_loader = torch.utils.data.DataLoader(
    #     dataset, batch_size=args.batch_size,
    #     sampler=train_sampler, num_workers=args.workers,
    #     pin_memory=True, collate_fn=collate_fn)

    # data_loader_test = torch.utils.data.DataLoader(
    #     dataset_test, batch_size=args.batch_size,
    #     sampler=test_sampler, num_workers=args.workers,
    #     pin_memory=True, collate_fn=collate_fn)

    transform_valid = torchvision.transforms.Compose([
        torchvision.transforms.ToFloatTensorInZeroOne(),
        torchvision.transforms.Resize((128, 171)),
        normalize,
        torchvision.transforms.CenterCrop((112, 112))
    ])


    train_data = VideoFolder(root=config['train_data_folder'],
                             csv_file_input=config['train_data_csv'],
                             csv_file_labels=config['labels_csv'],
                             clip_size=config['clip_size'],
                             nclips=1,
                             step_size=config['step_size'],
                             is_val=False,
                             transform=transform_train,
                             )

    # print(" > Using {} processes for data loader.".format(
    #     config["num_workers"]))
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=config['batch_size'], shuffle=True,
        num_workers=config['num_workers'], pin_memory=True,
        drop_last=True)


    print("Data loader ============== Num Processes {}".format(args.workers))
    frames, status = load_video("./video_data/drop.avi")
    # data_loader = torch.utils.data.DataLoader(
    #     dataset, batch_size=args.batch_size,
    #     sampler=train_sampler, num_workers=args.workers,
    #     pin_memory=True, collate_fn=collate_fn)

    # data_loader_test = torch.utils.data.DataLoader(
    #     dataset_test, batch_size=args.batch_size,
    #     sampler=test_sampler, num_workers=args.workers,
    #     pin_memory=True, collate_fn=collate_fn)

    # assert len(train_data.classes) == args.num_classes

    # define loss function (criterion) and pptimizer
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # define optimizer
    lr = args.lr * args.world_size
    last_lr = args.last_lr
    momentum = args.momentum
    weight_decay = args.weight_decay
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,amsgrad =True) 


    # if args.resume:
    #     checkpoint = torch.load(args.checkpoint, map_location='cpu')
    #     model_without_ddp.load_state_dict(checkpoint['model'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    #     args.start_epoch = checkpoint['epoch'] + 1

    # if args.eval_only:
    #     evaluate(model, criterion, data_loader_test, device=device)
    #     return

    # set end condition by num epochs
    num_epochs = int(args.num_epochs)
    if num_epochs == -1:
        num_epochs = 999999




def train(train_data, target, model, criterion, optimizer, epoch, device):
    print("=========== Starting Training =============")
    print("Training with {} epochs.".format(epoch))
    start_epoch = args.start_epoch if args.resume else 0
    # train_writer = tensorboardX.SummaryWriter("logs")
    model.train()

    for epoch in range(start_epoch, epoch):
        # if args.distributed:
        #     train_sampler.set_epoch(epoch)
        train_one_epoch(train_data, target, model, criterion, optimizer, device=device)
        evaluate(model, criterion, data_loader_test, device=device)

        # # write in tensorboard
        # train_writer.add_scalar('loss', train_loss, epoch + 1)
        # train_writer.add_scalar('top1', train_top1, epoch + 1)
        # train_writer.add_scalar('top5', train_top5, epoch + 1)

        # train_writer.add_scalar('val_loss', val_loss, epoch + 1)
        # train_writer.add_scalar('val_top1', val_top1, epoch + 1)
        # train_writer.add_scalar('val_top5', val_top5, epoch + 1)

        # # remember best prec@1 and save checkpoint
        # is_best = val_top1 > best_prec1
        # best_prec1 = max(val_top1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': "Conv4Col",
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, config)


def train_one_epoch(train_data, target, model, criterion, optimizer, device):


    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for input, target in zip(train_loader, target):

        input, target = input.to(device), target.to(device)

        model.zero_grad()

        # compute output and loss
        output,_ = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.detach(), target.detach().cpu(), topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if i % config["print_freq"] == 0:
        #     print('Epoch: [{0}][{1}/{2}]\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
        #           'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
        #               epoch, i, len(train_loader), loss=losses, top1=top1, top5=top5))
    return losses.avg, top1.avg, top5.avg


if __name__ == '__main__':
    load_video()