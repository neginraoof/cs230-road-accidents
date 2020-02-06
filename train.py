import numpy as np
import os
import torch
import torchvision
import argparse
import cv2
from dataset import VideoDataset


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

    spatial_transform_train = torchvision.transforms.Compose([
        torchvision.transforms.ToFloatTensorInZeroOne(),
        torchvision.transforms.Resize((128, 171)),
        torchvision.transforms.RandomHorizontalFlip(),
        normalize,
        torchvision.transforms.RandomCrop((112, 112))
    ])

    # temporal_transform = LoopPadding(opt.sample_duration)
    train_dataset = VideoDataset(video_dir, 
                # spatial_transform=spatial_transform_train,
                #  temporal_transform=temporal_transform,
                 sample_duration=args.sample_duration)

    train_sampler = RandomClipSampler(train_dataset.dataset, args.clips_per_video)
    # test_sampler = UniformClipSampler(test_dataset.video_clips, args.clips_per_video)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        sampler=train_sampler, num_workers=args.workers,
        pin_memory=True, collate_fn=collate_fn)
    # train_data = VideoFolder(root=config['train_data_folder'],
    #                          csv_file_input=config['train_data_csv'],
    #                          csv_file_labels=config['labels_csv'],
    #                          clip_size=config['clip_size'],
    #                          nclips=1,
    #                          step_size=config['step_size'],
    #                          is_val=False,
    #                          transform=transform_train,
    #                          )


    # define loss function (criterion) and pptimizer
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # # define optimizer
    # lr = args.lr * args.world_size
    # last_lr = args.last_lr
    # momentum = args.momentum
    # weight_decay = args.weight_decay
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr,amsgrad =True) 


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
    # num_epochs = int(args.num_epochs)
    # if num_epochs == -1:
    #     num_epochs = 999999


    for epoch in range(args.start_epoch, args.epochs):
        print("=============== Epoch ================")
        print(inputs)
        train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader,
                        device, epoch, args.print_freq, args.apex)
        evaluate(model, criterion, data_loader_test, device=device)

        ## or
        ## train(model, criterion, optimizer, lr_scheduler, data_loader,
                        # device, epoch, args.print_freq, args.apex)

        # if args.output_dir:
        #     checkpoint = {
        #         'model': model_without_ddp.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'lr_scheduler': lr_scheduler.state_dict(),
        #         'epoch': epoch,
        #         'args': args}
        #     utils.save_on_master(
        #         checkpoint,
        #         os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
        #     utils.save_on_master(
        #         checkpoint,
        #         os.path.join(args.output_dir, 'checkpoint.pth'))

    # Report Time


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
    main()