import os
#https://code.byted.org/zhaocong.zc/video_cls_template
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import sys
import time
import shutil
import json
import logging
from collections import OrderedDict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, DistributedSampler, WeightedRandomSampler
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.distributed as dist
import matplotlib.pyplot as plt

from clip.clip_model import CLIP
from clip.clip_config import parser
from clip.clip_metrics import AverageMeter, multi_acc
from transforms import *
from optimizer import WarmupAndExponentialDecayScheduler, WarmupAndOrderedScheduler, WarmupAndReduceLROnPlateauScheduler
from loss import ClipLoss
from v1.label_parse_v1 import parse_label
from util.fix_seed import set_fixed_seed

args = parser.parse_args()
best_acc = 0.0
best_loss = float("inf")
best_epoch = 0

dims = ['expression', 'material', 'person', 'style', 'topic']
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
label_dict = parse_label(args)

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
    global best_loss
    if args.dataset_type == 'v1':
        id2_label = label_dict[args.dim]['id2cls']
    else:
        id2_label = None
    model = CLIP(args, id2_label)
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            # checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            best_loss = checkpoint['best_loss']
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
    set_fixed_seed(1000, args, checkpoint)
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
    if args.dataset_type == 'v1':
        from clip.clip_dataset_v1 import ListFileDataSet
    else:
        from clip.clip_dataset_v2 import ListFileDataSet

    train_dataset = ListFileDataSet(
        prefix_path=args.prefix_path,
        info_file=args.info_file,
        list_file=args.train_list_file,
        label_dict=label_dict,
        num_segments=args.num_segments,
        dim=args.dim,
        train=True,
        image_tmpl='{:03d}.jpg',
        transform=torchvision.transforms.Compose([
            GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
            GroupRandomHorizontalFlip(is_flow=False),
            common_trans
        ]),
        bert_path=args.bert_path,
        local_rank=local_rank
    )
    val_dataset = ListFileDataSet(
        prefix_path=args.prefix_path,
        info_file=args.info_file,
        list_file=args.val_list_file,
        label_dict=label_dict,
        num_segments=args.num_segments,
        dim=args.dim,
        train=False,
        image_tmpl='{:03d}.jpg',
        transform=torchvision.transforms.Compose([
            GroupScale(int(scale_size)),
            GroupCenterCrop(crop_size),
            common_trans
        ]),
        bert_path=args.bert_path,
        local_rank=local_rank
    )
    test_dataset = ListFileDataSet(
        prefix_path=args.prefix_path,
        info_file=args.info_file,
        list_file=args.test_list_file,
        label_dict=label_dict,
        num_segments=args.num_segments,
        dim=args.dim,
        train=False,
        image_tmpl='{:03d}.jpg',
        transform=torchvision.transforms.Compose([
            GroupScale(int(scale_size)),
            GroupCenterCrop(crop_size),
            common_trans
        ]),
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
        #DistributedSampler 中 drop_last如果为 False，则采样器将填充样本数以使其可被进程数整除，以使数据均匀分布，所以最终验证集上的数据可能比实际的数据会多
        #DistributedSampler 中的drop_last指的是基于总数据量能不能整除进程数，选择是否扔掉多出来的部分
        #DataLoader中的drop_last指的是基于sampler中的数据量能不能整除batch数（这部分未证实，暂时这么认为），选择是否扔掉多出来的部分
        #所以DistributedSampler 中的drop_last 和DataLoader中的drop_last互不影响，自己设置自己的（这部分未证实，暂时这么认为）
        #DataLoader 中的num_workers，在分布式中会变成num_workers*进程数（已证实）
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=False)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False, drop_last=False)
        train_data_loader_batch_size = args.batch_size
        val_data_loader_batch_size = args.batch_size_val
        # num_workers = int(args.workers / args.nprocs)
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
                            drop_last=False)

    logger.info(f"train_dataloader{len(train_loader)}")
    logger.info(f"val_dataloader{len(val_loader)}")
    logger.info(f"test_dataloader{len(test_loader)}")
    # define loss function (criterion) and optimizer

    if args.clip_type == 'only_label':
        only_label = True
    else:
        only_label = False

    clip_loss = ClipLoss(local_loss=False,
                         gather_with_grad=True,
                         cache_labels=True,
                         rank=local_rank,
                         world_size=args.num_gpus,
                         only_label=only_label)

    num_steps_per_epoch = len(train_dataset) // (args.batch_size * args.num_gpus)
    logger.info(f"num_steps_per_epoch{num_steps_per_epoch}")
    vision_params = model.module.vision_model.backbone
    text_params = model.module.text_model
    backbone_params = list(torch.nn.ModuleList([vision_params, text_params]).parameters())
    backbone_params_id = list(map(id, backbone_params))
    normal_params = list(filter(lambda p: id(p) not in backbone_params_id, list(model.module.parameters())))

    if args.optim_type == 'sgd':
        backbone_optimizer = torch.optim.SGD(params=backbone_params,
                                             lr=args.lr,
                                             momentum=0.9)
        normal_optimizer = torch.optim.SGD(params=normal_params,
                                           lr=args.lr,
                                           momentum=0.9)
    elif args.optim_type == 'adam':
        backbone_optimizer = torch.optim.Adam(params=backbone_params,
                                              lr=args.lr)
        normal_optimizer = torch.optim.Adam(params=normal_params,
                                            lr=args.lr)
    elif args.optim_type == 'adamw':
        backbone_optimizer = torch.optim.AdamW(params=backbone_params,
                                               lr=args.lr,
                                               weight_decay=args.weight_decay)
        normal_optimizer = torch.optim.AdamW(params=normal_params,
                                             lr=args.lr,
                                             weight_decay=args.weight_decay)
    else:
        raise ValueError("args.optim_type is illegal")
    # arcface中优化器的写法
    # optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
    #                             lr=lr, weight_decay=weight_decay)

    if args.warmup_type == "WarmupAndOrdered":
        backbone_scheduler = WarmupAndOrderedScheduler(optimizer=backbone_optimizer,
                                                       num_steps_per_epoch=num_steps_per_epoch,
                                                       divide_every_n_epochs=args.divide_every_n_epochs,
                                                       decay_rate=args.lr_decay_rate,
                                                       num_warmup_epochs=args.num_warmup_epochs,
                                                       epoch_start=args.start_epoch)
        normal_scheduler = WarmupAndOrderedScheduler(optimizer=normal_optimizer,
                                                     num_steps_per_epoch=num_steps_per_epoch,
                                                     divide_every_n_epochs=args.divide_every_n_epochs,
                                                     decay_rate=args.lr_decay_rate,
                                                     num_warmup_epochs=0.0,
                                                     epoch_start=args.start_epoch)
    elif args.warmup_type == "WarmupAndReduceLROnPlateau":
        backbone_after_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            backbone_optimizer, mode='min', factor=0.1, patience=1, verbose=False,
            threshold=0.1, threshold_mode='abs', cooldown=0, min_lr=1e-06, eps=1e-06)
        normal_after_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            normal_optimizer, mode='min', factor=0.1, patience=1, verbose=False,
            threshold=0.1, threshold_mode='abs', cooldown=0, min_lr=1e-06, eps=1e-06)

        backbone_scheduler = WarmupAndReduceLROnPlateauScheduler(optimizer=backbone_optimizer,
                                                                 num_steps_per_epoch=num_steps_per_epoch,
                                                                 divide_every_n_epochs=args.divide_every_n_epochs,
                                                                 decay_rate=args.lr_decay_rate,
                                                                 num_warmup_epochs=args.num_warmup_epochs,
                                                                 epoch_start=args.start_epoch,
                                                                 after_scheduler=backbone_after_scheduler)
        normal_scheduler = WarmupAndReduceLROnPlateauScheduler(optimizer=normal_optimizer,
                                                               num_steps_per_epoch=num_steps_per_epoch,
                                                               divide_every_n_epochs=args.divide_every_n_epochs,
                                                               decay_rate=args.lr_decay_rate,
                                                               num_warmup_epochs=0.0,
                                                               epoch_start=args.start_epoch,
                                                               after_scheduler=normal_after_scheduler)
    elif args.warmup_type == "WarmupAndExponentialDecay":
        backbone_scheduler = WarmupAndExponentialDecayScheduler(optimizer=backbone_optimizer,
                                                                num_steps_per_epoch=num_steps_per_epoch,
                                                                divide_every_n_epochs=args.divide_every_n_epochs,
                                                                decay_rate=args.lr_decay_rate,
                                                                num_warmup_epochs=args.num_warmup_epochs,
                                                                epoch_start=args.start_epoch)

        normal_scheduler = WarmupAndExponentialDecayScheduler(optimizer=normal_optimizer,
                                                              num_steps_per_epoch=num_steps_per_epoch,
                                                              divide_every_n_epochs=args.divide_every_n_epochs,
                                                              decay_rate=args.lr_decay_rate,
                                                              num_warmup_epochs=0.0,
                                                              epoch_start=args.start_epoch)
    else:
        backbone_scheduler = None
        normal_scheduler = None

    if args.resume:
        # checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        if checkpoint is not None:
            backbone_optimizer.load_state_dict(checkpoint['backbone_optimizer'])
            normal_optimizer.load_state_dict(checkpoint['normal_optimizer'])
            backbone_scheduler.load_state_dict(checkpoint['backbone_scheduler'])
            normal_scheduler.load_state_dict(checkpoint['normal_scheduler'])


    if args.evaluate:
        logger.info('start to validate...')
        validate(val_loader, model, clip_loss, -1, "val", local_rank)
        validate(test_loader, model, clip_loss, -1, "test", local_rank)
        logger.info('finished validate!')
        return

    # training loop
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        if args.dist_type == "ddp":
            train_sampler.set_epoch(epoch)

        train_loss, train_acc = train(train_loader,
                                      model,
                                      clip_loss,
                                      (backbone_optimizer, normal_optimizer),
                                      (backbone_scheduler, normal_scheduler),
                                      epoch,
                                      local_rank)

        # evaluate on validation set
        val_loss, val_acc = validate(val_loader, model, clip_loss, epoch, "val", local_rank)
        test_loss, test_acc = validate(test_loader, model, clip_loss, epoch, "test", local_rank)

        if args.warmup_type == "WarmupAndReduceLROnPlateau":
            backbone_scheduler.step(metrics=val_loss)
            normal_scheduler.step(metrics=val_loss)

        if args.dist_type != "ddp" or local_rank == 0:
            y_loss['train'].append(train_loss)
            y_acc['train'].append(train_acc)
            y_loss['val'].append(val_loss)
            y_acc['val'].append(val_acc)
            y_loss['test'].append(test_loss)
            y_acc['test'].append(test_acc)
            draw_curve(epoch)

            # is_best = val_acc > best_acc
            # best_acc = max(val_acc, best_acc)

            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)

            if is_best:
                best_epoch = epoch

            _save_checkpoint({'epoch': epoch + 1, 'arch': args.arch,
                              'state_dict': model.state_dict(), 'best_acc': best_acc, 'best_loss': best_loss,
                              'backbone_optimizer': backbone_optimizer.state_dict(),
                              'normal_optimizer': normal_optimizer.state_dict(),
                              'backbone_scheduler': backbone_scheduler.state_dict(),
                              'normal_scheduler': normal_scheduler.state_dict(),
                              'cuda_rng_state': torch.cuda.get_rng_state(),
                              'torch_rng_state': torch.get_rng_state(),
                              'np_rng_state': np.random.get_state(),
                              'py_rng_state': random.getstate()},
                             is_best,
                             '{}_checkpoint.pth.tar'.format('last'))
            logger.info(f'best epoch: {best_epoch}\t val_loss: {best_loss}')


def train(train_loader, model, clip_loss, optimizers, schedulers, epoch, local_rank):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

    model.train()
    # optimizer = optimizers
    # scheduler = schedulers
    end = time.time()
    for step, data in enumerate(train_loader):

        (_, features, labels) = data
        data_time.update(time.time() - end)
        if args.dist_type == "ddp":
            feature = [f.cuda(local_rank) for f in features]
            if args.dataset_type == 'v1':
                labels = labels.cuda(local_rank)
        else:
            feature = [f.cuda() for f in features]
            if args.dataset_type == 'v1':
                labels = labels.cuda()

        video_embedding, text_feature, logit_scale = model(*feature)
        if args.clip_type == 'only_label':
            loss, logits_per_image, labels = clip_loss(video_embedding, text_feature, logit_scale, labels)
        else:
            loss, logits_per_image, labels = clip_loss(video_embedding, text_feature, logit_scale)

        acc = multi_acc(logits_per_image, labels)

        if args.dist_type == "ddp":
            batch_size = logits_per_image.size(0)
            acces.update(acc.data.item(), batch_size)
            losses.update(loss.data.item(), batch_size)
            torch.distributed.barrier()
            acces.all_reduce(local_rank)
            losses.all_reduce(local_rank)
        else:
            batch_size = logits_per_image.size(0)
            acces.update(acc.data.item(), batch_size)
            losses.update(loss.data.item(), batch_size)

        for optimizer in optimizers:
            optimizer.zero_grad()
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        for scheduler in schedulers:
            scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if step % args.print_freq == 0:
            logger.info('Epoch: [{epoch}][{step}/{len}]\t'
                        'batch: {batch_size} local_rank/nprocs: {local_rank}/{nprocs}\t'
                        'b_lr: {b_lr:.8f} n_lr: {n_lr:.8f}\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                        'Acc {acces.val:.3f} ({acces.avg:.3f})'.format(epoch=epoch, step=step, len=len(train_loader),
                                                                       batch_size=video_embedding.size(0), local_rank=local_rank,
                                                                       nprocs=args.nprocs,
                                                                       b_lr=optimizers[0].param_groups[-1]['lr'],
                                                                       n_lr=optimizers[1].param_groups[-1]['lr'],
                                                                       batch_time=batch_time,
                                                                       data_time=data_time,
                                                                       losses=losses,
                                                                       acces=acces))
        #         break
    return losses.avg, acces.avg


def validate(val_loader, model, clip_loss, epoch, data_type, local_rank):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

    model.eval()

    end = time.time()
    with torch.no_grad():
        for step, (vid, features, labels) in enumerate(val_loader):
            if args.dist_type == "ddp":
                feature = [f.cuda(local_rank) for f in features]
                if args.dataset_type == 'v1':
                    labels = labels.cuda(local_rank)
            else:
                feature = [f.cuda() for f in features]
                if args.dataset_type == 'v1':
                    labels = labels.cuda()
            video_embedding, text_feature, logit_scale = model(*feature)
            if args.clip_type == 'only_label':
                loss, logits_per_image, labels = clip_loss(video_embedding, text_feature, logit_scale, labels)
            else:
                loss, logits_per_image, labels = clip_loss(video_embedding, text_feature, logit_scale)

            acc = multi_acc(logits_per_image, labels)

            if args.dist_type == "ddp":
                batch_size = logits_per_image.size(0)
                acces.update(acc.data.item(), batch_size)
                losses.update(loss.data.item(), batch_size)
                torch.distributed.barrier()
                acces.all_reduce(local_rank)
                losses.all_reduce(local_rank)
            else:
                batch_size = logits_per_image.size(0)
                acces.update(acc.data.item(), batch_size)
                losses.update(loss.data.item(), batch_size)

            batch_time.update(time.time() - end)
            end = time.time()

        logger.info('{data_type}: [{epoch}/{total_epoch}]\t'
                    'batch: {batch_size} local_rank/nprocs: {local_rank}/{nprocs}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                    'Acc {acces.val:.3f} ({acces.avg:.3f})'.format(data_type=data_type, epoch=epoch,
                                                                   total_epoch=args.epochs,
                                                                   batch_size=video_embedding.size(0), local_rank=local_rank,
                                                                   nprocs=args.nprocs,
                                                                   batch_time=batch_time,
                                                                   losses=losses,
                                                                   acces=acces))

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
