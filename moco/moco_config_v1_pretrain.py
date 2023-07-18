import argparse

parser = argparse.ArgumentParser(description="PyTorch implementation of Jianying Template Video Classification Networks")

# ========================= Base Configs =========================
parser.add_argument('--info-file', type=str, help='datatset info file path',
                    default='/mnt/bn/zc-model-nas-lq/data/data_v1/vu_jy_tag_01.parquet')
parser.add_argument('--train-list-file', type=str, help='train list file path',
                    default='/mnt/bn/zc-model-nas-lq/data/data_v1/train.pkl')
parser.add_argument('--test-list-file', type=str, help='test list file path',
                    default='/mnt/bn/zc-model-nas-lq/data/data_v1/test.pkl')
parser.add_argument('--val-list-file', type=str, help='val list file path',
                    default='/mnt/bn/zc-model-nas-lq/data/data_v1/val.pkl')
parser.add_argument('--prefix-path', type=str, help='train val test prefix path',
                    default='/mnt/bn/zc-model-nas-lq/data/data_v1')
# parser.add_argument('--num-class', type=int, default=2)

# ========================= distributed Configs =========================
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist-type', default="ddp", type=str, choices=['ddp', 'dp'])
parser.add_argument('--is-syn-bn', type=bool, default=False)

# ========================= Mocov3 Configs =========================
parser.add_argument('--moco-m', default=0.99, type=float,
                    help='moco momentum of updating momentum encoder (default: 0.99)')



# ========================= Label Configs =========================
parser.add_argument('--smoothing', default=0., type=float, help='label smoothing ratio (default: 0.1)')

parser.add_argument('--idx2class_file_expression', type=str, help='expression label path',
                    default='./vocabulary/video_classification_class_id_expression.txt')
parser.add_argument('--idx2class_file_person', type=str, help='person label path',
                    default='./vocabulary/video_classification_class_id_person.txt')
parser.add_argument('--idx2class_file_style', type=str, help='style label path',
                    default='./vocabulary/video_classification_class_id_style.txt')
parser.add_argument('--idx2class_file_topic', type=str, help='topic label path',
                    default='./vocabulary/video_classification_class_id_topical.txt')

# ========================= Model Configs =========================
parser.add_argument('--arch', type=str, default="resnet18",
                    choices=['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                             'Inception3', 'inception_v3',
                             'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3'
                             ])
parser.add_argument('--only-vision', type=bool, default=False)
parser.add_argument('--use-moe', type=bool, default=False)
parser.add_argument('--dim', type=str, default="topic",
                    choices=['person', 'expression', 'style', 'topic'])
parser.add_argument('--use-MLP', type=bool, default=False)
parser.add_argument('--vlad-add-final-fc', type=bool, default=True)
parser.add_argument('--vlad-hidden-size', type=int, default=1024)
parser.add_argument('--vlad-add-se', type=bool, default=True)
parser.add_argument('--bert-fintuing', type=bool, default=False)
parser.add_argument('--consensus_type', type=str, default='tsm_trn_base_tsn',
                    choices=['avg', 'netvlad', 'nextvlad', 'timesformer', 'tsmnetvlad', 'tsm_trn_base_tsn', 'trn', 'trn_base_tsn', 'videoswin'])
parser.add_argument('--resample', default=False, type=bool, help='class balanced resample')

# ========================= Vision Configs =========================
parser.add_argument('--num-segments', type=int, default=10)
parser.add_argument('--num-segments-val', type=int, default=10)
parser.add_argument('--vlad-cluster-size', type=int, default=64)
parser.add_argument('--timsf-pretrian', type=str, default='/mnt/bn/zc-model-nas-lq/model_pretrain_weight/TimeSformer_divST_8x32_224_K600.pyth')
parser.add_argument('--video-swin-pretrian', type=str, default='/mnt/bn/zc-model-nas-lq/model_pretrain_weight/swin_base_patch244_window877_kinetics600_22k.pth')

# parser.add_argument('--timsf-hidden-size', type=int, default=768)
# parser.add_argument('--cover-arch', type=str, default="resnet50")

# ========================= Text Configs =========================
# parser.add_argument('--word-idx-file', type=str,
#                     default='/opt/tiger/mlx_notebook/video_classification/data/word_to_idx_add_ocr.pkl')
# parser.add_argument('--word-embedding-path', type=str,
#                     default='/opt/tiger/mlx_notebook/video_classification/data/word_embedding_SougouNews_add_ocr.npz')
# parser.add_argument('--text-max-size', type=int, default=32)
# parser.add_argument('--text-output-size', type=int, default=512)
# parser.add_argument('--text-dropout', type=float, default=0.5)
# parser.add_argument('--textcnn-num-filters', type=int, default=128)
# parser.add_argument('--textcnn-filter-sizes', type=str, default='2,3,4')
# parser.add_argument('--textrcnn-hidden-size', type=int, default=256)
# parser.add_argument('--textrcnn-num-layers', type=int, default=1)
# parser.add_argument('--use-bert', type=bool, default=True)

parser.add_argument('--bert-path', type=str, default='/mnt/bn/zc-model-nas-lq/model_pretrain_weight/chinese-bert-wwm-ext', help='you can use yourself bert_file_path like /home/work/bert-**-model')

# ========================= Audio Configs =========================
# parser.add_argument('--olny-use-audio', type=bool, default=False)
# parser.add_argument('--num-sec', type=int, default=10)
parser.add_argument('--audio-pretrian', type=str, default='/mnt/bn/zc-model-nas-lq/model_pretrain_weight/11200_iterations.pth')
# parser.add_argument('--audio-hdf5path', type=str, default='/opt/tiger/mlx_notebook/audio/')
# parser.add_argument('--audio-hdf5-test-index-files', type=str, default='/opt/tiger/mlx_notebook/video_classification/data/test_style_file_0822_index')
# parser.add_argument('--audio-hdf5-train-index-files', type=str, default='/opt/tiger/mlx_notebook/video_classification/data/train_style_file_0822_index')

# ========================= Fusion Configs =========================
# parser.add_argument('--fusion-output-size', type=int, default=1024)
parser.add_argument('--fusion-type', type=str, default='concat',
                    choices=['concat', 'lmf', 'gmu'])
parser.add_argument('--fusion-embedding-size', type=int, default=1024)
parser.add_argument('--se-gating-type', type=str, default='BNSEModule',
                    choices=['SqueezeContextGating', 'BNSEModule', 'ContextGating'])

# ========================= Distillation Configs =========================
parser.add_argument('--is-distillation', default=False, type=bool, help='is use ONE distillation')
parser.add_argument('--consistency-rampup', default=0, type=int, help='consistency-rampup')


# ========================= Learning Configs =========================
parser.add_argument('--epochs', default=14, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=32, type=int, help='mini-batch size')
#parser.add_argument('--batch-size', default=16, type=int, help='mini-batch size')
parser.add_argument('--batch-size-val', default=32, type=int, help='mini-batch size')
parser.add_argument('--dropout', '--do', default=0.3, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')
parser.add_argument('--lr', '--learning-rate', default=0.003, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', default=0.0001, type=float, help='weight decay')
parser.add_argument('--divide-every-n-epochs', default=1, type=int, help='learning rate decay every n epochs')
parser.add_argument('--lr-decay-rate', default=0.85, type=float, help='learning rate decay rate')
parser.add_argument('--num-warmup-epochs', default=1, type=float, help='learning rate decay rate')
parser.add_argument('--warmup-type', default='WarmupAndReduceLROnPlateau', choices=['WarmupAndOrdered', 'WarmupAndReduceLROnPlateau', 'WarmupAndExponentialDecay'],
                    type=str, help='warmup type')
parser.add_argument('--optim-type', default='adamw', choices=['adamw', 'adam', 'sgd'],
                    type=str, help='warmup type')

# ========================= Loss Configs =========================
parser.add_argument('--cls-loss-type', default='smoothing_loss', choices=['focal_loss', 'smoothing_loss'],
                    type=str, help='classifier loss type')
parser.add_argument('--use-bbn', default=False, type=bool, help='is use bbn')
parser.add_argument('--bbn-loss-method', default='weight_logist', choices=['weight_loss', 'weight_logist'],
                    type=str, help='bbn loss method')
parser.add_argument('--bbn-div-epoch', type=int, default=28)

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
parser.add_argument('--save_path', type=str, default="/mnt/bn/zc-model-nas-lq/model_result/checkpoints_top_v1")
parser.add_argument('--experiment_pref', type=str, default="20230220_train_mocov3_pretrain_v1")
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')


# ========================= Motion Configs =========================
# parser.add_argument('--subsample-num', type=int, default=2)
# parser.add_argument('--trn-feature-dim', type=int, default=1024)
# parser.add_argument('--tsm-shift-div', type=int, default=8)
# parser.add_argument('--tsm-shift-place', type=str, default='blockres')
# parser.add_argument('--tsm-temporal-pool', type=bool, default=False)
parser.add_argument('--trn-hidden-size', type=int, default=1024)
parser.add_argument('--tsm-is-shift', type=bool, default=True)
parser.add_argument('--tsm-non-local', type=bool, default=False)


