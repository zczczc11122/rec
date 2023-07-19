import os
#https://code.byted.org/zhaocong.zc/video_cls_template
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import sys
import time
import copy
import shutil
import json
import logging
from collections import OrderedDict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import matplotlib.pyplot as plt
from moco.moco_dataset_v1 import ListFileDataSet
from moco.mocov2_model import MoCoV2
from transforms import *

from moco.moco_config_v1_pretrain import parser
from v1.metrics_v1 import AverageMeter, accuracy
from optimizer import adjust_learning_rate
from v1.label_parse_v1 import parse_label
import torch.distributed as dist
from util.fix_seed import set_fixed_seed

args = parser.parse_args()
best_acc = 0.0
best_loss = float("inf")
best_epoch = 0

"""
初始化日志
"""
if args.dist_type != "ddp" or args.local_rank == 0:
    os.makedirs(os.path.join(args.save_path, args.experiment_pref), exist_ok=True)
    logger = logging.getLogger('template_video_classification')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(filename)s - %(lineno)d - %(levelname)s - %(message)s')
    if args.evaluate:
        handler_file = logging.FileHandler(os.path.join(args.save_path, args.experiment_pref, 'val.log'), mode='w')
    else:
        handler_file = logging.FileHandler(os.path.join(args.save_path, args.experiment_pref, 'train.log'), mode='w')
    handler_file.setFormatter(formatter)
    handler_stream = logging.StreamHandler(sys.stdout)
    handler_stream.setFormatter(formatter)
    logger.addHandler(handler_file)
    logger.addHandler(handler_stream)
else:
    logger = logging.getLogger('template_video_classification')
logger.parent = None
######################################################################
# Draw Curve
if args.dist_type != "ddp" or args.local_rank == 0:
    x_epoch = []

    y_loss = {}  # loss history
    y_loss['train'] = []
    y_loss['val'] = []
    y_loss['test'] = []

    y_acc = {}
    y_acc['train'] = []
    y_acc['val'] = []
    y_acc['test'] = []

    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="loss")
    ax1 = fig.add_subplot(122, title="acc")
    draw_flag = True


def _restore_ckp():
    checkpoint = None
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))
    return checkpoint


checkpoint = _restore_ckp()


def draw_curve(current_epoch):
    global draw_flag
    x_epoch.append(current_epoch)

    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax0.plot(x_epoch, y_loss['test'], 'go-', label='test')

    ax1.plot(x_epoch, y_acc['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_acc['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_acc['test'], 'go-', label='test')

    if draw_flag == True:
        ax0.legend()
        ax1.legend()
        draw_flag = False
    fig.savefig(os.path.join(".", 'train.jpg'))
    fig.savefig(os.path.join(args.save_path, args.experiment_pref, 'train.jpg'))


def _init_model():
    global best_acc
    model = MoCoV2(args)
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            # checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            # args.start_epoch = checkpoint['epoch']
            # best_acc = checkpoint['best_acc']
            new_state_dict = copyStateDict(checkpoint['state_dict'])
            model.load_state_dict(new_state_dict, strict=True)
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    if args.use_gpu:
        logger.info("Let's use {} GPUs !".format(args.num_gpus))
        if args.dist_type == "dp":
            model = torch.nn.DataParallel(model).cuda()
        elif args.dist_type == "ddp":
            if args.is_syn_bn:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            torch.cuda.set_device(args.local_rank)
            model.cuda(args.local_rank)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                              find_unused_parameters=True,
                                                              broadcast_buffers=False)

    return model


def _save_model_args():
    if args.evaluate:
        return
    experiment_path = os.path.join(args.save_path, args.experiment_pref)
    os.makedirs(experiment_path, exist_ok=True)
    args_dict = vars(args)
    with open(os.path.join(experiment_path, 'args.json'), 'w') as fp:
        json.dump(args_dict, fp)

def main():
    set_fixed_seed(100, args, checkpoint)
    # cudnn.benchmark = True
    args.nprocs = torch.cuda.device_count()

    main_worker(args.local_rank)

def main_worker(local_rank):
    global best_acc
    global best_loss
    global best_epoch
    if args.dist_type == "ddp":
        dist.init_process_group(backend='nccl')
        dist.barrier()

    label_dict = parse_label(args)
    model = _init_model()
    if local_rank == 0 or args.dist_type != "ddp":
        _save_model_args()
    logger.info(f"local_rank: {local_rank}")

    input_size = model.module.input_size
    crop_size = model.module.crop_size
    scale_size = model.module.scale_size
    input_mean = model.module.input_mean
    input_std = model.module.input_std

    common_trans = torchvision.transforms.Compose([
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        GroupNormalize(input_mean, input_std)
    ])

    train_augmentation = [
        GroupRandomResizedCrop(crop_size, (0.2, 1.0)),
        GroupColorJitter(0.4, 0.4, 0.4, 0.1, 0.8),
        GroupRandomGrayscale(0.2),
        GroupGaussianBlur([0.1, 2.0], 0.5),
        GroupRandomHorizontalFlipV2(),
        common_trans
    ]
    val_augmentation = [
        GroupScale(int(scale_size)),
        GroupCenterCrop(crop_size),
        common_trans

    ]

    train_dataset = ListFileDataSet(
        prefix_path=args.prefix_path,
        info_file=args.info_file,
        list_file=args.train_list_file,
        label_dict=label_dict,
        num_segments=args.num_segments,
        dim=args.dim,  # person | expression | style | topic
        train=True,
        image_tmpl='{:03d}.jpg',
        transform=torchvision.transforms.Compose(train_augmentation),
        bert_path=args.bert_path,
        local_rank=local_rank
    )

    val_dataset = ListFileDataSet(
        prefix_path=args.prefix_path,
        info_file=args.info_file,
        list_file=args.val_list_file,
        label_dict=label_dict,
        num_segments=args.num_segments_val,
        dim=args.dim,  # person | expression | style | topic
        train=False,
        image_tmpl='{:03d}.jpg',
        transform=torchvision.transforms.Compose(val_augmentation),
        bert_path=args.bert_path,
        local_rank=local_rank
    )
    test_dataset = ListFileDataSet(
        prefix_path=args.prefix_path,
        info_file=args.info_file,
        list_file=args.test_list_file,
        label_dict=label_dict,
        num_segments=args.num_segments_val,
        dim=args.dim,  # person | expression | style | topic
        train=False,
        image_tmpl='{:03d}.jpg',
        transform=torchvision.transforms.Compose(val_augmentation),
        bert_path=args.bert_path,
        local_rank=local_rank
    )

    logger.info(f"train_dataset{len(train_dataset)}")
    logger.info(f"val_dataset{len(val_dataset)}")
    logger.info(f"test_dataset{len(test_dataset)}")
    if args.dist_type != "ddp":
        train_sampler = RandomSampler(train_dataset)
        val_sampler = SequentialSampler(val_dataset)
        test_sampler = SequentialSampler(test_dataset)
        train_data_loader_batch_size = args.batch_size * args.num_gpus
        val_data_loader_batch_size = args.batch_size_val * args.num_gpus
        num_workers = args.workers

    else:

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, drop_last=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=False)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False, drop_last=False)
        train_data_loader_batch_size = args.batch_size
        val_data_loader_batch_size = args.batch_size_val
        num_workers = int(args.workers)

    # Data loading code
    # train_loader drop_last=True 原因：不丢掉的话，可能会存在最后一个batch_size为1的情况，bn会出问题
    # 如果想要断点重训表现一致，DataLoader有个参数persistent_workers，必须设为False(默认就是False)，具体可见分享的断点重训
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=train_data_loader_batch_size,
                              num_workers=num_workers,
                              sampler=train_sampler,
                              pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=val_data_loader_batch_size,
                            num_workers=num_workers,
                            sampler=val_sampler,
                            pin_memory=True,
                            drop_last=True)
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=val_data_loader_batch_size,
                            num_workers=num_workers,
                            sampler=test_sampler,
                            pin_memory=True,
                            drop_last=True)

    logger.info(f"train_dataloader{len(train_loader)}")
    logger.info(f"val_dataloader{len(val_loader)}")
    logger.info(f"test_dataloader{len(test_loader)}")
    # define loss function (criterion) and optimizer

    if args.dist_type == "ddp":
        criterion = nn.CrossEntropyLoss().cuda(local_rank)
    else:
        criterion = nn.CrossEntropyLoss().cuda()

    num_steps_per_epoch = len(train_dataset) // (args.batch_size * args.num_gpus)
    logger.info(f"num_steps_per_epoch{num_steps_per_epoch}")

    optimizer = torch.optim.SGD(params=model.module.parameters(),
                                lr=args.lr,
                                momentum=0.9,
                                weight_decay=1e-4)
    # arcface中优化器的写法
    # optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
    #                             lr=lr, weight_decay=weight_decay)

    if args.resume:
        # checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        if checkpoint is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])

    if args.evaluate:
        logger.info('start to validate...')
        validate(val_loader, model, criterion, -1, "val", local_rank)
        validate(test_loader, model, criterion, -1, "test", local_rank)
        logger.info('finished validate!')
        return

    # training loop
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        if args.dist_type == "ddp":
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)
        train_loss, train_acc = train(train_loader,
                                      model,
                                      criterion,
                                      optimizer,
                                      epoch,
                                      local_rank)

        # evaluate on validation set
        val_loss, val_acc = validate(val_loader, model, criterion, epoch, "val", local_rank)
        test_loss, test_acc = validate(test_loader, model, criterion, epoch, "test", local_rank)

        if args.dist_type != "ddp" or local_rank == 0:
            y_loss['train'].append(train_loss)
            y_acc['train'].append(train_acc)
            y_loss['val'].append(val_loss)
            y_acc['val'].append(val_acc)
            y_loss['test'].append(test_loss)
            y_acc['test'].append(test_acc)
            draw_curve(epoch)

            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)

            if is_best:
                best_epoch = epoch
            _save_checkpoint({'epoch': epoch + 1, 'arch': args.arch,
                              'state_dict': model.state_dict(), 'best_acc': best_acc,
                              'optimizer':optimizer.state_dict(),
                              'cuda_rng_state': torch.cuda.get_rng_state(),
                              'torch_rng_state': torch.get_rng_state(),
                              'np_rng_state': np.random.get_state(),
                              'py_rng_state': random.getstate()},
                             is_best,
                             '{}_checkpoint.pth.tar'.format('last'))
            logger.info(f'best epoch: {best_epoch}\t val_acc: {best_acc}')



def train(train_loader, model, criterion, optimizer, epoch, local_rank):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

    model.train()

    end = time.time()
    for step, data in enumerate(train_loader):
        (_, images) = data
        data_time.update(time.time() - end)
        if args.dist_type == "ddp":
            images[0] = images[0].cuda(local_rank)
            images[1] = images[1].cuda(local_rank)

        else:
            images[0] = images[0].cuda()
            images[1] = images[1].cuda()

        output, target = model(vision_q=images[0], vision_k=images[1])
        loss = criterion(output, target)
        acc = accuracy(output, target, topk=(1,))[0][0]

        if args.dist_type == "ddp":
            batch_size = images[0].size(0)
            acces.update(acc.data.item(), batch_size)
            losses.update(loss.data.item(), batch_size)
            torch.distributed.barrier()
            acces.all_reduce(local_rank)
            losses.all_reduce(local_rank)
        else:
            batch_size = images[0].size(0)
            acces.update(acc.data.item(), batch_size)
            losses.update(loss.data.item(), batch_size)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if step % args.print_freq == 0:
            logger.info('Epoch: [{epoch}][{step}/{len}]\t'
                        'batch: {batch_size} local_rank/nprocs: {local_rank}/{nprocs}\t'
                        'b_lr: {b_lr:.8f}\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                        'Acc {acces.val:.3f} ({acces.avg:.3f})'.format(epoch=epoch, step=step, len=len(train_loader),
                                                                       batch_size=batch_size, local_rank=local_rank, nprocs=args.nprocs,
                                                                       b_lr=optimizer.param_groups[-1]['lr'],
                                                                       batch_time=batch_time,
                                                                       data_time=data_time,
                                                                       losses=losses,
                                                                       acces=acces))
    return losses.avg, acces.avg


def validate(val_loader, model, criterion, epoch, data_type, local_rank):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

    model.eval()

    end = time.time()
    with torch.no_grad():
        for step, (vid, images) in enumerate(val_loader):
            if args.dist_type == "ddp":
                images[0] = images[0].cuda(local_rank)
                images[1] = images[1].cuda(local_rank)
            else:
                images[0] = images[0].cuda()
                images[1] = images[1].cuda()
            output, target = model(vision_q=images[0], vision_k=images[1])
            loss = criterion(output, target)
            acc = accuracy(output, target, topk=(1,))[0][0]
            batch_size = images[0].size(0)

            losses.update(loss.data.item(), batch_size)
            acces.update(acc.data.item(), batch_size)
            batch_time.update(time.time() - end)
            end = time.time()
        if args.dist_type == "ddp":
            torch.distributed.barrier()
            losses.all_reduce(local_rank)
            acces.all_reduce(local_rank)
            batch_time.all_reduce(local_rank)

        logger.info('{data_type}: [{epoch}/{total_epoch}]\t'
                    'batch: {batch_size} local_rank/nprocs: {local_rank}/{nprocs}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                    'Acc {acces.val:.3f} ({acces.avg:.3f})'.format(data_type=data_type, epoch=epoch, total_epoch=args.epochs,
                                                                   batch_size=batch_size, local_rank=local_rank,
                                                                   nprocs=args.nprocs,
                                                                   batch_time=batch_time,
                                                                   losses=losses,
                                                                   acces=acces))
    if args.dist_type != "ddp" or local_rank == 0:
        logger.info(f'{data_type} Results: Acc {acces.avg:.3f} Loss {losses.avg:.5f}')
    return losses.avg, acces.avg


def _save_checkpoint(state, is_best, checkpoint_filename):
    target_path = os.path.join(args.save_path, args.experiment_pref)
    os.makedirs(target_path, exist_ok=True)
    torch.save(state, os.path.join(target_path, checkpoint_filename))
    if is_best:
        best_name = 'best.pth.tar'
        shutil.copyfile(os.path.join(target_path, checkpoint_filename),
                        os.path.join(target_path, best_name))


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith('module'):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = '.'.join(k.split('.')[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


if __name__ == '__main__':
    #python -m torch.distributed.launch --nproc_per_node=8 main_v1.py
    main()
