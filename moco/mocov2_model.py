import logging
from torch import nn
import torch
import torch.distributed
from vision.models import TSNNeXtVLADSE, TSNNetVLADSE, Timesformer, TSMNetVLADSE, TSMRelationModuleBaseTsn, \
    TSNRelationModuleMultiScale, TSNRelationModuleBaseTsn, VideoSwin, TSNAVG

logger = logging.getLogger('template_video_classification')

class MoCoV2(nn.Module):
    def __init__(self,
                 args,
                 deploy=False):
        super(MoCoV2, self).__init__()

        self.K = 65536
        self.m = 0.999
        self.T = 0.07
        self.dim = 128
        self.mlp = True
        self.args = args

        if self.args.consensus_type == "netvlad":
            self.vision_model_q = TSNNetVLADSE(self.args)
            self.vision_model_k = TSNNetVLADSE(self.args)
        elif self.args.consensus_type == "nextvlad":
            self.vision_model_q = TSNNeXtVLADSE(self.args)
            self.vision_model_k = TSNNeXtVLADSE(self.args)
        elif self.args.consensus_type == "trn":
            self.vision_model_q = TSNRelationModuleMultiScale(self.args)
            self.vision_model_k = TSNRelationModuleMultiScale(self.args)
        elif self.args.consensus_type == "trn_base_tsn":
            self.vision_model_q = TSNRelationModuleBaseTsn(self.args)
            self.vision_model_k = TSNRelationModuleBaseTsn(self.args)
        elif self.args.consensus_type == "tsmnetvlad":
            assert 'resnet' in args.arch.lower(), 'tsm arch目前只支持 resnet'
            self.vision_model_q = TSMNetVLADSE(self.args)
            self.vision_model_k = TSMNetVLADSE(self.args)
        elif self.args.consensus_type == "tsm_trn_base_tsn":
            assert 'resnet' in args.arch.lower(), 'tsm arch目前只支持 resnet'
            self.vision_model_q = TSMRelationModuleBaseTsn(self.args)
            self.vision_model_k = TSMRelationModuleBaseTsn(self.args)
        elif self.args.consensus_type == "timesformer":
            assert self.args.num_segments == 8, 'timesformer 对应 num_segments 必须为8'
            self.vision_model_q = Timesformer(self.args, deploy)
            self.vision_model_k = Timesformer(self.args, deploy)
        elif self.args.consensus_type == "videoswin":
            #VideoSwin 论文中fnum_segments用的32，感觉太多了，我们还是用10
            #经分析，最好还是8的倍数，可见笔记分析
            assert self.args.num_segments % 8 == 0, 'videoswin 对应 num_segments 为8的倍数'
            self.vision_model_q = VideoSwin(self.args, deploy)
            self.vision_model_k = VideoSwin(self.args, deploy)
        elif self.args.consensus_type == "avg":
            self.vision_model_q = TSNAVG(self.args)
            self.vision_model_k = TSNAVG(self.args)
        else:
            raise ValueError("args.consensus_type is illegal")

        if self.mlp:  # hack: brute-force replacement
            dim_mlp = self.vision_model_q.get_output_dim()
            projection_head_q = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp),
                nn.ReLU(),
                nn.Linear(dim_mlp, self.dim)
            )
            projection_head_k = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp),
                nn.ReLU(),
                nn.Linear(dim_mlp, self.dim)
            )

            self.vision_model_q_mlp = nn.Sequential(
                self.vision_model_q,
                projection_head_q
            )

            self.vision_model_k_mlp = nn.Sequential(
                self.vision_model_k,
                projection_head_k
            )
        else:
            self.vision_model_q_mlp = nn.Sequential(
                self.vision_model_q
            )
            self.vision_model_k_mlp = nn.Sequential(
                self.vision_model_k
            )

        for param_q, param_k in zip(
            self.vision_model_q_mlp.parameters(), self.vision_model_k_mlp.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(self.dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
                self.vision_model_q_mlp.parameters(), self.vision_model_k_mlp.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, vision_q, vision_k):
        """Encoder, Pool, Predit
            expected shape of 'features': (n_batch, 5, input_dim)
        """

        q = self.vision_model_q_mlp(vision_q)
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            vision_k, idx_unshuffle = self._batch_shuffle_ddp(vision_k)

            k = self.vision_model_k_mlp(vision_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels

    @property
    def input_size(self):
        return self.vision_model_q.input_size

    @property
    def input_mean(self):
        return self.vision_model_q.input_mean

    @property
    def input_std(self):
        return self.vision_model_q.input_std

    @property
    def crop_size(self):
        return self.vision_model_q.input_size

    @property
    def scale_size(self):
        return int(self.vision_model_q.input_size / 0.875)

    def summary(self):
        logger.info("""
    Initializing Template Video Classify Framework
        frame base model:       {}
        num_segments:           {}
        consensus_module:       {}
        dropout_ratio:          {}
        frame_feature_dim:      {}
            """.format(self.args.arch,
                       self.args.num_segments,
                       self.args.consensus_type,
                       self.args.dropout,
                       self.vision_model_q.feature_dim))


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


if __name__ == '__main__':
    # model = VideoCNNModel()
    print()


