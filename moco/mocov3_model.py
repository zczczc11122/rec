import logging
from torch import nn
import torch
import torch.distributed
from vision.models import TSNNeXtVLADSE, TSNNetVLADSE, Timesformer, TSMNetVLADSE, TSMRelationModuleBaseTsn, \
    TSNRelationModuleMultiScale, TSNRelationModuleBaseTsn, VideoSwin, TSNAVG

logger = logging.getLogger('template_video_classification')

class MoCoV3(nn.Module):
    def __init__(self,
                 args,
                 deploy=False):
        super(MoCoV3, self).__init__()

        self.T = 1
        self.mlp_dim = 4096
        self.dim = 256
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

        vision_dim = self.vision_model_q.get_output_dim()
        projection_head_q = self._build_mlp(2, vision_dim, self.mlp_dim, self.dim)
        projection_head_k = self._build_mlp(2, vision_dim, self.mlp_dim, self.dim)

        self.vision_model_q_mlp = nn.Sequential(
            self.vision_model_q,
            projection_head_q
        )

        self.vision_model_k_mlp = nn.Sequential(
            self.vision_model_k,
            projection_head_k
        )

        # predictor
        self.predictor = self._build_mlp(2, self.dim, self.mlp_dim, self.dim, False)

        for param_q, param_k in zip(self.vision_model_q_mlp.parameters(), self.vision_model_k_mlp.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient


    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_q, param_k in zip(self.vision_model_q_mlp.parameters(), self.vision_model_k_mlp.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)

    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def forward(self, vision_q, vision_k, m):
        """Encoder, Pool, Predit
            expected shape of 'features': (n_batch, 5, input_dim)
        """
        # compute features
        q1 = self.predictor(self.vision_model_q_mlp(vision_q))
        q2 = self.predictor(self.vision_model_q_mlp(vision_k))

        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder

            # compute momentum features as targets
            k1 = self.vision_model_k_mlp(vision_q)
            k2 = self.vision_model_k_mlp(vision_k)

        return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)

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
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

if __name__ == '__main__':
    # model = VideoCNNModel()
    print()


