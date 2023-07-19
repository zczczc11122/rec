import argparse

parser = argparse.ArgumentParser(description="PyTorch implementation of Jianying Template Video Classification Networks")

# ========================= Base Configs =========================
parser.add_argument('--dataset-type', type=str, default="v1", choices=['v1', 'v2'])
parser.add_argument('--info-file', type=str, help='datatset info file path',
                    default='/opt/tiger/mlx_notebook/cc/classification/video/data/data_v1/vu_jy_tag_01.parquet')
parser.add_argument('--train-list-file', type=str, help='train list file path',
                    default='/opt/tiger/mlx_notebook/cc/classification/video/data/data_v1/train.pkl')
parser.add_argument('--test-list-file', type=str, help='test list file path',
                    default='/opt/tiger/mlx_notebook/cc/classification/video/data/data_v1/test.pkl')
parser.add_argument('--val-list-file', type=str, help='val list file path',
                    default='/opt/tiger/mlx_notebook/cc/classification/video/data/data_v1/val.pkl')
parser.add_argument('--prefix-path', type=str, help='train val test prefix path',
                    default='/opt/tiger/mlx_notebook/cc/classification/video/data/data_v1')

# parser.add_argument('--dataset-type', type=str, default="v2", choices=['v1', 'v2'])
# parser.add_argument('--info-file', type=str, help='datatset info file path',
#                     default='/opt/tiger/mlx_notebook/cc/classification/video/data/data_v2/info.parquet')
# parser.add_argument('--train-list-file', type=str, help='train list file path',
#                     default='/opt/tiger/mlx_notebook/cc/classification/video/data/data_v2/train.pkl')
# parser.add_argument('--test-list-file', type=str, help='test list file path',
#                     default='/opt/tiger/mlx_notebook/cc/classification/video/data/data_v2/test.pkl')
# parser.add_argument('--val-list-file', type=str, help='val list file path',
#                     default='/opt/tiger/mlx_notebook/cc/classification/video/data/data_v2/val.pkl')
# parser.add_argument('--prefix-path', type=str, help='train val test prefix path',
#                     default='/opt/tiger/mlx_notebook/cc/classification/video/data/data_v2')

parser.add_argument('--idx2class_file_expression', type=str, help='expression label path',
                    default='./vocabulary/video_classification_class_id_expression.txt')
parser.add_argument('--idx2class_file_person', type=str, help='person label path',
                    default='./vocabulary/video_classification_class_id_person.txt')
parser.add_argument('--idx2class_file_style', type=str, help='style label path',
                    default='./vocabulary/video_classification_class_id_style.txt')
parser.add_argument('--idx2class_file_topic', type=str, help='topic label path',
                    default='./vocabulary/video_classification_class_id_topical.txt')

# ========================= distributed Configs =========================
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist-type', default="ddp", type=str, choices=['ddp', 'dp'])
parser.add_argument('--is-syn-bn', type=bool, default=False)

# ========================= CLIP Configs =========================
parser.add_argument('--clip-type', type=str, default="title_label",
                    choices=['only_label', 'title_label'])

# ========================= Model Configs =========================
parser.add_argument('--arch', type=str, default="resnet18",
                    choices=['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                             'Inception3', 'inception_v3',
                             'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3'
                             ])
parser.add_argument('--vlad-add-final-fc', type=bool, default=True)
parser.add_argument('--vlad-hidden-size', type=int, default=1024)
parser.add_argument('--vlad-add-se', type=bool, default=True)
parser.add_argument('--bert-fintuing', type=bool, default=True)
parser.add_argument('--consensus_type', type=str, default='tsm_trn_base_tsn',
                    choices=['avg', 'netvlad', 'nextvlad', 'timesformer', 'tsmnetvlad', 'tsm_trn_base_tsn', 'trn', 'trn_base_tsn', 'videoswin'])
# parser.add_argument('--resample', default=False, type=bool, help='class balanced resample')

# ========================= Vision Configs =========================
parser.add_argument('--num-segments', type=int, default=10)
parser.add_argument('--num-segments_val', type=int, default=10)
parser.add_argument('--vlad-cluster-size', type=int, default=64)
parser.add_argument('--timsf-pretrian', type=str, default='../TimeSformer_divST_8x32_224_K600.pyth')
parser.add_argument('--video-swin-pretrian', type=str, default='../swin_base_patch244_window877_kinetics600_22k.pth')
parser.add_argument('--dim', type=str, default="topic",
                    choices=['person', 'expression', 'style', 'topic'])

# ========================= Text Configs =========================
parser.add_argument('--bert-path', type=str, default='/opt/tiger/mlx_notebook/cc/classification/video/chinese-bert-wwm-ext', help='you can use yourself bert_file_path like /home/work/bert-**-model')

# ========================= Learning Configs =========================
parser.add_argument('--epochs', default=14, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=64, type=int, help='mini-batch size')
#parser.add_argument('--batch-size', default=16, type=int, help='mini-batch size')
parser.add_argument('--batch-size-val', default=64, type=int, help='mini-batch size')
parser.add_argument('--dropout', '--do', default=0.3, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', default=0.0001, type=float, help='weight decay')
parser.add_argument('--divide-every-n-epochs', default=1, type=int, help='learning rate decay every n epochs')
parser.add_argument('--lr-decay-rate', default=0.85, type=float, help='learning rate decay rate')
parser.add_argument('--num-warmup-epochs', default=1, type=float, help='learning rate decay rate')
parser.add_argument('--warmup-type', default='WarmupAndReduceLROnPlateau', choices=['WarmupAndOrdered', 'WarmupAndReduceLROnPlateau', 'WarmupAndExponentialDecay'],
                    type=str, help='warmup type')
parser.add_argument('--optim-type', default='adamw', choices=['adamw', 'adam', 'sgd'],
                    type=str, help='warmup type')

# ========================= Monitor Configs =========================
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')

# ========================= Runtime Configs =========================
parser.add_argument('--workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--num-gpus', type=int, default=8)
parser.add_argument('--use-gpu', default=True, action='store_false')

# ========================= Checkpoint Configs =========================
parser.add_argument('--resume', default='',
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', default=False, type=bool, help='evaluate model on validation set')
parser.add_argument('--save_path', type=str, default="./checkpoints_top_v1")
parser.add_argument('--experiment_pref', type=str, default="clip_20221031_1_train")
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')


# ========================= Motion Configs =========================
parser.add_argument('--trn-hidden-size', type=int, default=1024)
parser.add_argument('--tsm-is-shift', type=bool, default=True)
parser.add_argument('--tsm-non-local', type=bool, default=False)


