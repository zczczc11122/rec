import os
#https://code.byted.org/zhaocong.zc/video_cls_template
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import sys
import time
import copy
import shutil
import json
import logging
import laplace
from collections import OrderedDict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, DistributedSampler, WeightedRandomSampler
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import matplotlib.pyplot as plt
from v1.dataset_v1 import ListFileDataSet
from v1.dataset_bnn_v1 import ListFileDataSetBBN
# from models_Timsf import MultiModel
from v1.models_cnn_v1 import VideoCNNModel
from audio import vggish
from transforms import *
from transformers import BertTokenizer
from moco.moco_config_v1_cls import parser
from v1.metrics_v1 import AverageMeter, multi_acc, plot_epochs_stats, plot_heatmap, plot_report, write_val_result
from optimizer import WarmupAndExponentialDecayScheduler, WarmupAndOrderedScheduler, WarmupAndReduceLROnPlateauScheduler
from loss import SmoothCrossEntropyLoss, KLLoss, FocalLoss
from v1.label_parse_v1 import parse_label
import torch.distributed as dist
from util.dist_utils import reduce_mean, all_gather
from util.fix_seed import set_fixed_seed
from util.distill_utils import get_current_consistency_weight
from util.dist_sampler import WeightedRandomSamplerDDP
#遗留问题：分布式目前不支持带权重采样的sample方式，需要重写方法
# 关注分布式的workers数量
# convert_sync_batchnorm加载参数问题
#注val 的 samlpe 的drop_last和 dataloader的drop_last 会不会冲突
#看日志中的batch_size、epoch数,   dataloader长度和num_steps_per_epoch一不一样
#看学习率变化是否符合预期
#find_unused_parameters=True 什么用
#logger存在重复打印问题
#optimizers[0].param_groups[-1] 为啥是-1（[-1]）和WarmupAndOrderedScheduler的min_delta_to_update_lr一起看
#classification_report  的 acc 和 micro ,以及为啥val的混淆矩阵少一类，micro中的pre、recall、f1一模一样？

args = parser.parse_args()
best_acc = 0.0
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
    model = VideoCNNModel(args)
    if args.moco_resume:
        if os.path.isfile(args.moco_resume):
            logger.info("=> loading checkpoint '{}'".format(args.moco_resume))
            moco_checkpoint = torch.load(args.moco_resume, map_location=torch.device('cpu'))
            state_dict = moco_checkpoint['state_dict']
            new_state_dict = OrderedDict()
            if list(state_dict.keys())[0].startswith('module'):
                start_idx = 1
            else:
                start_idx = 0
            for k, v in state_dict.items():
                layer_name_list = k.split('.')[start_idx:]
                if layer_name_list[0] == 'vision_model_q':
                    layer_name_list = layer_name_list[1:]
                    new_state_dict['.'.join(layer_name_list)] = v
            model.vision_model.load_state_dict(new_state_dict, strict=True)

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            # checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
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

    if args.use_bbn:
        train_dataset = ListFileDataSetBBN(
            prefix_path=args.prefix_path,
            info_file=args.info_file,
            list_file=args.train_list_file,
            label_dict=label_dict,
            num_segments=args.num_segments,
            num_cls=model.module.n_classes,
            dim=args.dim,  # person | expression | style | topic
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
    else:
        train_dataset = ListFileDataSet(
            prefix_path=args.prefix_path,
            info_file=args.info_file,
            list_file=args.train_list_file,
            label_dict=label_dict,
            num_segments=args.num_segments,
            dim=args.dim,  # person | expression | style | topic
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
        num_segments=args.num_segments_val,
        dim=args.dim,  # person | expression | style | topic
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
        num_segments=args.num_segments_val,
        dim=args.dim,  # person | expression | style | topic
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
        if args.resample:
            train_sampler = WeightedRandomSampler(weights=train_dataset.weights,
                                                  num_samples=len(train_dataset),
                                                  replacement=True)
        else:
            train_sampler = RandomSampler(train_dataset)
        val_sampler = SequentialSampler(val_dataset)
        test_sampler = SequentialSampler(test_dataset)
        train_data_loader_batch_size = args.batch_size * args.num_gpus
        val_data_loader_batch_size = args.batch_size_val * args.num_gpus
        num_workers = args.workers

    else:
        if args.resample:
            train_sampler = WeightedRandomSamplerDDP(train_dataset,
                                                     weights=train_dataset.weights,
                                                     replacement=True,
                                                     drop_last=True)
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

    if args.cls_loss_type == 'focal_loss':
        criterion_ce = FocalLoss(class_num=model.module.n_classes)
    elif args.cls_loss_type == 'smoothing_loss':
        criterion_ce = SmoothCrossEntropyLoss(smoothing=args.smoothing)
    else:
        raise ValueError('args.cls_loss_type is illegal')
    criterion_kl = KLLoss()

    if args.dist_type == "ddp":
        criterion_ce = criterion_ce.cuda(local_rank)
        criterion_kl = criterion_kl.cuda(local_rank)
    else:
        criterion_ce = criterion_ce.cuda()
        criterion_kl = criterion_kl.cuda()


    num_steps_per_epoch = len(train_dataset) // (args.batch_size * args.num_gpus)
    logger.info(f"num_steps_per_epoch{num_steps_per_epoch}")
    vision_params = model.module.vision_model.backbone
    title_params = model.module.title_model
    ocr_params = model.module.ocr_model
    audio_params = model.module.audio_model
    if args.bert_fintuing:
        backbone_params = list(torch.nn.ModuleList([vision_params, title_params, ocr_params, audio_params]).parameters())
    else:
        backbone_params = list(torch.nn.ModuleList([vision_params, title_params, audio_params]).parameters())

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
                                            backbone_optimizer, mode='max', factor=0.1, patience=1, verbose=False,
                                            threshold=0.5, threshold_mode='abs', cooldown=0, min_lr=1e-06, eps=1e-06)
        normal_after_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                            normal_optimizer, mode='max', factor=0.1, patience=1, verbose=False,
                                            threshold=0.5, threshold_mode='abs', cooldown=0, min_lr=1e-06, eps=1e-06)

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
        validate(val_loader, model, (criterion_ce, criterion_kl), -1, "val", label_dict[args.dim]["id2cls"], local_rank)
        validate(test_loader, model, (criterion_ce, criterion_kl), -1, "test", label_dict[args.dim]["id2cls"], local_rank)
        logger.info('finished validate!')
        return

    # training loop
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        if args.dist_type == "ddp":
            train_sampler.set_epoch(epoch)

        train_loss, train_acc = train(train_loader,
                                      model,
                                      (criterion_ce, criterion_kl),
                                      (backbone_optimizer, normal_optimizer),
                                      (backbone_scheduler, normal_scheduler),
                                      epoch,
                                      local_rank)

        # evaluate on validation set
        val_loss, val_acc = validate(val_loader, model, (criterion_ce, criterion_kl), epoch, "val", label_dict[args.dim]["id2cls"], local_rank)
        test_loss, test_acc = validate(test_loader, model, (criterion_ce, criterion_kl), epoch, "test", label_dict[args.dim]["id2cls"], local_rank)

        if args.warmup_type == "WarmupAndReduceLROnPlateau":
            backbone_scheduler.step(metrics=val_acc)
            normal_scheduler.step(metrics=val_acc)

        if args.dist_type != "ddp" or local_rank == 0:
            y_loss['train'].append(train_loss)
            y_acc['train'].append(train_acc)
            y_loss['val'].append(val_loss)
            y_acc['val'].append(val_acc)
            y_loss['test'].append(test_loss)
            y_acc['test'].append(test_acc)
            draw_curve(epoch)

            is_best = val_acc > best_acc
            best_acc = max(val_acc, best_acc)
            if is_best:
                best_epoch = epoch
            _save_checkpoint({'epoch': epoch + 1, 'arch': args.arch,
                              'state_dict': model.state_dict(), 'best_acc': best_acc,
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
            _export_ts_models()
            logger.info(f'best epoch: {best_epoch}\t val_acc: {best_acc}')
    if args.dist_type != "ddp" or local_rank == 0:
        print("testing speed .....")
        test_speed()
        print('done!!!')


def train(train_loader, model, criterions, optimizers, schedulers, epoch, local_rank):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

    model.train()
    criterion_ce, criterion_kl = criterions
    # optimizer = optimizers
    # scheduler = schedulers
    consistency_weight = get_current_consistency_weight(epoch, consistency_rampup=args.consistency_rampup)
    end = time.time()
    for step, data in enumerate(train_loader):
        if args.use_bbn:
            (_, features, label, _, sample_feature, sample_label) = data
        else:
            (_, features, label) = data
        data_time.update(time.time() - end)
        if args.dist_type == "ddp":
            feature = [f.cuda(local_rank) for f in features]
            label = label.cuda(local_rank)
            if args.use_bbn:
                sample_feature = [f.cuda(local_rank) for f in sample_feature]
                sample_label = sample_label.cuda(local_rank)
        else:
            feature = [f.cuda() for f in features]
            label = label.cuda()
            if args.use_bbn:
                sample_feature = [f.cuda() for f in sample_feature]
                sample_label = sample_label.cuda()

        if args.use_bbn:
            # bnn需要model forward 2次
            # 网上有人解释：在分布式训练中，如果对同一模型进行多次调用则会触发报错（使用nn.DataParallel并行训练或者单卡训练均可正常运行），
            # 即nn.parallel.DistributedDataParallel方法封装的模型，forword()函数和backward()函数必须交替执行，
            # 如果执行多个（次）forward()然后执行一次backward()则会报错。解决方案：设置DistributedDataParallel 的 broadcast_buffers=False
            if args.bbn_loss_method == 'weight_logist':
                # 这里实现和官方的BBN实现略有差异,官方的BBN相当于使用不同的分类器进行相加，且模型最后一个block两个分支也不一样
                # 这里实现的是相同分类器，且两个分支模型参数也完全一样
                outputs = model(*feature)
                predict_logists = outputs['predict']['video']
                sample_outputs = model(*sample_feature)
                sample_predict_logists = sample_outputs['predict']['video']

                l = 1 - ((epoch) / args.bbn_div_epoch) ** 2
                bbn_predict_logists = l * predict_logists + (1 - l) * sample_predict_logists
                loss = l * criterion_ce(bbn_predict_logists, label) + (1 - l) * criterion_ce(bbn_predict_logists, sample_label)
            elif args.bbn_loss_method == 'weight_loss':
                # 这里实现和官方的BBN实现差距较大
                # 这里直接将loss加权
                outputs = model(*feature)
                predict_logists = outputs['predict']['video']
                sample_outputs = model(*sample_feature)
                sample_predict_logists = sample_outputs['predict']['video']

                l = 1 - ((epoch) / args.bbn_div_epoch) ** 2
                loss = l * criterion_ce(predict_logists, label) + (1 - l) * criterion_ce(sample_predict_logists,
                                                                                             sample_label)
            else:
                raise ValueError("args.bbn_loss_method is illegal")
        else:
            outputs = model(*feature)
            predict_logists = outputs['predict']['video']
            loss = criterion_ce(predict_logists, label)

        if args.is_distillation:
            output_vision = outputs['predict']['vision']
            output_title = outputs['predict']['title']
            output_ocr = outputs['predict']['ocr']
            output_audio = outputs['predict']['audio']
            if args.bert_fintuing:
                loss_ce = criterion_ce(output_vision, label) + \
                             criterion_ce(output_title, label) + \
                             criterion_ce(output_ocr, label) + \
                             criterion_ce(output_audio, label)
                loss_kl = criterion_kl(output_vision, predict_logists) + \
                             criterion_kl(output_title, predict_logists) + \
                             criterion_kl(output_ocr, predict_logists) + \
                             criterion_kl(output_audio, predict_logists)
            else:
                loss_ce = criterion_ce(output_vision, label) + \
                          criterion_ce(output_audio, label)
                loss_kl = criterion_kl(output_vision, predict_logists) + \
                          criterion_kl(output_audio, predict_logists)

            loss += loss_ce
            loss += consistency_weight * loss_kl
        if args.use_bbn:
            if args.bbn_loss_method == 'weight_logist':
                acc = l * multi_acc(bbn_predict_logists, label) + (1 - l) * multi_acc(bbn_predict_logists, sample_label)
            elif args.bbn_loss_method == 'weight_loss':
                acc = l * multi_acc(predict_logists, label) + (1 - l) * multi_acc(sample_predict_logists, sample_label)
        else:
            acc = multi_acc(predict_logists, label)
        if args.dist_type == "ddp":
            batch_size = feature[0].size(0)
            acces.update(acc.data.item(), batch_size)
            losses.update(loss.data.item(), batch_size)
            torch.distributed.barrier()
            acces.all_reduce(local_rank)
            losses.all_reduce(local_rank)
        else:
            batch_size = feature[0].size(0)
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
                                                                       batch_size=batch_size, local_rank=local_rank, nprocs=args.nprocs,
                                                                       b_lr=optimizers[0].param_groups[-1]['lr'],
                                                                       n_lr=optimizers[1].param_groups[-1]['lr'],
                                                                       batch_time=batch_time,
                                                                       data_time=data_time,
                                                                       losses=losses,
                                                                       acces=acces))
        #         break
    return losses.avg, acces.avg


def validate(val_loader, model, criterions, epoch, data_type, idx2class, local_rank):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

    vid_list = []
    target_list = []
    pred_tag_list = []
    pred_prob_list = []
    model.eval()
    criterion_ce, criterion_kl = criterions

    # consistency_weight = get_current_consistency_weight(epoch, consistency_rampup=args.consistency_rampup)
    end = time.time()
    with torch.no_grad():
        for step, (vid, features, label) in enumerate(val_loader):
            if args.dist_type == "ddp":
                feature = [f.cuda(local_rank) for f in features]
                label = label.cuda(local_rank)
            else:
                feature = [f.cuda() for f in features]
                label = label.cuda()
            outputs = model(*feature)

            predict_logists = outputs['predict']['video']
            pred_softmax = torch.softmax(predict_logists, dim=1)
            pred_probs, pred_tags = torch.max(pred_softmax, dim=1)

            vid_list.extend(list(vid))
            target_list.extend(label.cpu().numpy().tolist())
            pred_tag_list.extend(pred_tags.cpu().numpy().tolist())
            pred_prob_list.extend(pred_probs.cpu().numpy().tolist())

            loss = criterion_ce(predict_logists, label)
            acc = multi_acc(predict_logists, label)

            # if args.is_distillation:
            #     output_vision = outputs['predict']['vision']
            #     output_title = outputs['predict']['title']
            #     output_ocr = outputs['predict']['ocr']
            #     output_audio = outputs['predict']['audio']
            #     loss_ce = criterion_ce(output_vision, label) + \
            #               criterion_ce(output_title, label) + \
            #               criterion_ce(output_ocr, label) + \
            #               criterion_ce(output_audio, label)
            #     loss_kl = criterion_kl(output_vision, predict_logists) + \
            #               criterion_kl(output_title, predict_logists) + \
            #               criterion_kl(output_ocr, predict_logists) + \
            #               criterion_kl(output_audio, predict_logists)
            #     loss += loss_ce
            #     loss += consistency_weight * loss_kl

            batch_size = feature[0].size(0)
            losses.update(loss.data.item(), batch_size)
            acces.update(acc.data.item(), batch_size)
            batch_time.update(time.time() - end)
            end = time.time()
        if args.dist_type == "ddp":
            torch.distributed.barrier()
            losses.all_reduce(local_rank)
            acces.all_reduce(local_rank)
            batch_time.all_reduce(local_rank)
            vid_list_tensor = torch.tensor([int(i) for i in vid_list], dtype=torch.long).cuda(local_rank)
            target_list_tensor = torch.tensor([int(i) for i in target_list], dtype=torch.long).cuda(local_rank)
            pred_tag_list_tensor = torch.tensor([int(i) for i in pred_tag_list], dtype=torch.long).cuda(local_rank)
            pred_prob_list_tensor = torch.tensor([float(i) for i in pred_prob_list], dtype=torch.float32).cuda(local_rank)

            vid_list = all_gather(vid_list_tensor, args.nprocs, local_rank).cpu().numpy().tolist()
            target_list = all_gather(target_list_tensor, args.nprocs, local_rank).cpu().numpy().tolist()
            pred_tag_list = all_gather(pred_tag_list_tensor, args.nprocs, local_rank).cpu().numpy().tolist()
            pred_prob_list = all_gather(pred_prob_list_tensor, args.nprocs, local_rank).cpu().numpy().tolist()

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
        plot_heatmap(stats={'y_test': target_list, 'y_pred': pred_tag_list},
                     save_path=os.path.join(args.save_path, args.experiment_pref, f'{data_type}_heatmap_{epoch}_{acces.avg}.png'),
                     idx2class=idx2class)
        plot_report(stats={'y_test': target_list, 'y_pred': pred_tag_list},
                    save_path=os.path.join(args.save_path, args.experiment_pref, f'{data_type}_report_{epoch}_{acces.avg}.json'),
                    idx2class=idx2class)
        write_val_result(
            stats={'vid': vid_list, 'y_test': target_list, 'y_pred': pred_tag_list, 'y_value': pred_prob_list},
            save_path=os.path.join(args.save_path, args.experiment_pref, f'{data_type}_case_result_{epoch}_{acces.avg}.csv'),
            idx2class=idx2class)
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

def test_speed():
    run_num = 10
    num_segments = args.num_segments_val
    bert_path = args.bert_path
    target_pt_path = os.path.join(args.save_path, args.experiment_pref, "model_pt")

    mytokenizer = BertTokenizer.from_pretrained(bert_path)
    title_text_idx = mytokenizer.encode_plus('哈哈哈', max_length=50, padding='max_length', truncation=True)
    title_input_ids = torch.LongTensor(title_text_idx['input_ids']).unsqueeze(0).cuda()
    title_token_type_ids = torch.LongTensor(title_text_idx['token_type_ids']).unsqueeze(0).cuda()
    title_attention_mask = torch.LongTensor(title_text_idx['attention_mask']).unsqueeze(0).cuda()

    ocr_text_idx = mytokenizer.encode_plus('哈哈哈', max_length=50, padding='max_length', truncation=True)
    ocr_input_ids = torch.LongTensor(ocr_text_idx['input_ids']).unsqueeze(0).cuda()
    ocr_token_type_ids = torch.LongTensor(ocr_text_idx['token_type_ids']).unsqueeze(0).cuda()
    ocr_attention_mask = torch.LongTensor(ocr_text_idx['attention_mask']).unsqueeze(0).cuda()

    image = torch.rand(1, 3 * num_segments, 224, 224).cuda()
    image2 = torch.rand(1, 3 * num_segments, 224, 224).cuda()
    audio = torch.rand((1, 32000)).cuda()

    features = [image, audio, title_input_ids, title_token_type_ids, title_attention_mask, ocr_input_ids,
                ocr_token_type_ids, ocr_attention_mask]

    model_2 = torch.jit.load(os.path.join(target_pt_path, 'model.pt'))
    output2 = model_2(*features)
    start = time.time()
    for i in range(run_num):
        output2 = model_2(*features)
    end = time.time()
    logger.info(f'total: {end - start} run_num:{run_num}')
    logger.info(f'ave: {(end - start) / run_num}')
    logger.info(f'fps: {run_num / (end - start)}')


def _export_ts_models():
    # device = 'cpu'
    prototxt = '''model_meta {
    framework_type: PYTORCH
    framework_version: "1.8"
    deploy_path: "model.pt"
    input {
        name: "image"
        min_shape: [1, 24, 224, 224]
        max_shape: [10, 24, 224, 224]
        dtype: FP32
    }
    output {
        name: "output"
        min_shape: [1, 2]
        max_shape: [10, 2]
        dtype: FP32
    }
}
runtime_conf {
    enable_batching: false
    max_batching_time_ms: 24
    allowed_batch_sizes: [1, 2, 10]
}'''
    export_args = copy.deepcopy(args)
    # export_args.num_segments = 8  # by default, don't modify!!

    export_model = VideoCNNModel(export_args, deploy=True)
    checkpoint = torch.load(os.path.join(args.save_path, args.experiment_pref, 'last_checkpoint.pth.tar'), map_location=torch.device('cpu'))
    new_state_dict = copyStateDict(checkpoint['state_dict'])
    export_model.load_state_dict(new_state_dict, strict=True)
    export_model.eval()
    export_model.cuda()

    target_pt_path = os.path.join(args.save_path, args.experiment_pref, "model_pt")
    os.makedirs(target_pt_path, exist_ok=True)

    mytokenizer = BertTokenizer.from_pretrained(args.bert_path)

    title_text_idx = mytokenizer.encode_plus('哈哈哈', max_length=50, padding='max_length', truncation=True)
    title_input_ids = torch.LongTensor(title_text_idx['input_ids']).unsqueeze(0).cuda()
    title_token_type_ids = torch.LongTensor(title_text_idx['token_type_ids']).unsqueeze(0).cuda()
    title_attention_mask = torch.LongTensor(title_text_idx['attention_mask']).unsqueeze(0).cuda()

    ocr_text_idx = mytokenizer.encode_plus('哈哈哈', max_length=50, padding='max_length', truncation=True)
    ocr_input_ids = torch.LongTensor(ocr_text_idx['input_ids']).unsqueeze(0).cuda()
    ocr_token_type_ids = torch.LongTensor(ocr_text_idx['token_type_ids']).unsqueeze(0).cuda()
    ocr_attention_mask = torch.LongTensor(ocr_text_idx['attention_mask']).unsqueeze(0).cuda()

    image = torch.rand(1, 3 * export_args.num_segments_val, 224, 224).cuda()
    image2 = torch.rand(1, 3 * export_args.num_segments_val, 224, 224).cuda()
    audio = torch.rand((1, 32000)).cuda()

    features = [image, audio, title_input_ids, title_token_type_ids, title_attention_mask, ocr_input_ids,
                ocr_token_type_ids, ocr_attention_mask]

    output = export_model(*features)
    torch.jit.trace(export_model, (image2, audio, title_input_ids, title_token_type_ids, title_attention_mask, ocr_input_ids,
                ocr_token_type_ids, ocr_attention_mask)).save(os.path.join(target_pt_path, 'model.pt'))
    model_2 = torch.jit.load(os.path.join(target_pt_path, 'model.pt'))
    output2 = model_2(*features)
    print(output == output2)

    laplace_deploy_file = {
        "framework": "pytorch",
        "model": "model.pt",
        "inputs": [{"name": "image", "batch_axis": 1}],
        "outputs": [{"name": "output", "batch_axis": 1}]
    }
    # with open(os.path.join(target_pt_path, 'model.json'), 'w') as json_fh:
    #     json.dump(laplace_deploy_file, json_fh, ensure_ascii=False)
    logger.info('export ts model success.')

    # warmup_dir = os.path.join(target_pt_path, 'assets.extra')
    # os.makedirs(warmup_dir, exist_ok=True)
    # warmup_path = os.path.join(warmup_dir, 'tf_serving_warmup_requests')
    # dummy_x = np.ones((1, 3 * export_args.num_segments, 224, 224), np.float32)
    # feed_dict = {
    #     "image": laplace.make_tensor_proto(dummy_x).SerializeToString()
    # }
    # laplace.Model.make_inference_warmup_data(warmup_path, feed_dict)
    #
    # with open(os.path.join(target_pt_path, 'model_conf.prototxt'), 'w') as pro_f:
    #     pro_f.write(prototxt)
    # logger.info('export ts model success.')

    # laplace_deploy_file = {
    #     "framework": "pytorch",
    #     "model": "model.pt",
    #     "inputs": [
    #         {"name": "video_inputs", "batch_axis": 1},
    #         {"name": "title_inputs", "batch_axis": 1},
    #         {"name": "ocr_inputs", "batch_axis": 1},
    #     ],
    #     "outputs": [
    #         {"name": "predict", "batch_axis": 1},
    #         {"name": "embedding", "batch_axis": 1}
    #     ]
    # }
    # ts_vision_inputs = torch.rand(1, 3 * 10, 224, 224).to(device)
    # ts_title_inputs = torch.randint(low=0, high=10, size=(1, args.text_max_size)).to(device)
    # ts_ocr_inputs = torch.randint(low=0, high=10, size=(1, args.text_max_size * 2)).to(device)
    # ts_keywords_inputs = torch.randint(low=0, high=10, size=(1, args.keywords_max_size)).to(device)
    # traced = torch.jit.trace(export_model, (ts_vision_inputs, ts_title_inputs, ts_ocr_inputs, ts_keywords_inputs))
    # traced.save(os.path.join(args.save_path, args.experiment_pref, 'model.pt'))
    # with open(os.path.join(args.save_path, args.experiment_pref, 'model.json'), 'w') as json_fh:
    #     json.dump(laplace_deploy_file, json_fh, ensure_ascii=False)
    # logger.info('export ts model success.')


if __name__ == '__main__':
    #python -m torch.distributed.launch --nproc_per_node=8 main_v1.py
    main()
