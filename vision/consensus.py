import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NetVLADConsensus(torch.nn.Module):
    def __init__(self,
                 feature_size,
                 num_segments,
                 num_clusters,
                 output_feature_size,
                 add_final_fc,
                 add_batch_norm=True,
                 ) -> None:
        super().__init__()
        self.feature_size = feature_size
        self.num_segments = num_segments
        self.add_batch_norm = add_batch_norm
        self.num_clusters = num_clusters
        self.add_final_fc = add_final_fc

        self.activation_weights = torch.nn.Linear(in_features=feature_size, out_features=num_clusters)
        self.bn = torch.nn.BatchNorm1d(num_features=num_clusters)
        self.cluster = torch.nn.Parameter(torch.randn(1, feature_size, num_clusters))
        if self.add_final_fc:
            self.linear = torch.nn.Sequential(torch.nn.Dropout(0.2),
                                              torch.nn.Linear(in_features=self.feature_size * self.num_clusters,
                                                              out_features=output_feature_size))
            self.vlad_hidden_size = output_feature_size
        else:

            self.vlad_hidden_size = self.feature_size * self.num_clusters

    def forward(self, x):
        num_segments = x.size()[1]
        # batch_size, num_segments, feature_size
        x = x.view(-1, self.feature_size)
        # batch_size * num_segments, feature_size

        activation = self.activation_weights(x)
        # batch_size * num_segments, num_clusters
        if self.add_batch_norm:
            activation = self.bn(activation)
        activation = torch.nn.Softmax(dim=1)(activation).view(-1, num_segments, self.num_clusters)
        # batch_size * num_segments, num_clusters --- view ---> batch_size, num_segments, num_clusters

        activation_segments_sum = torch.sum(activation, -2, keepdim=True)
        # batch_size, 1, num_clusters

        cluster_weighted = torch.mul(activation_segments_sum, self.cluster)
        # batch_size, feature_size, num_clusters

        activation = activation.permute(0, 2, 1)
        # batch_size, num_clusters, num_segments

        x = x.view(-1, num_segments, self.feature_size)
        # batch_size, num_segments, feature_size

        x_weighted = torch.matmul(activation, x)
        # batch_size, num_clusters, feature_size
        x_weighted = x_weighted.permute(0, 2, 1)
        # batch_size, feature_size, num_clusters

        vlad = x_weighted - cluster_weighted

        vlad = F.normalize(vlad, p=2, dim=1)
        vlad = vlad.reshape(-1, self.num_clusters * self.feature_size)
        vlad = F.normalize(vlad, p=2, dim=1)
        if self.add_final_fc:
            vlad = self.linear(vlad)
        return vlad

    def get_output_dim(self):
        return self.vlad_hidden_size

def general_weight_initialization(module: nn.Module):
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        if module.weight is not None:
            nn.init.uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight)
        # print("Initing linear")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

class TimeFirstBatchNorm1d(nn.Module):
    def __init__(self, dim, groups=None):
        super().__init__()
        self.groups = groups
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, tensor):
        _, length, dim = tensor.size()
        if self.groups:
            dim = dim // self.groups
        tensor = tensor.view(-1, dim)
        tensor = self.bn(tensor)
        if self.groups:
            return tensor.view(-1, length, self.groups, dim)
        else:
            return tensor.view(-1, length, dim)

class NetXtVLADConsensus(torch.nn.Module):
    def __init__(self,
                 feature_size,
                 num_segments,
                 num_clusters,
                 output_feature_size,
                 add_final_fc,
                 add_batch_norm=True,
                 alpha=100.0,
                 groups: int = 8,
                 expansion: int = 2,
                 p_drop=0.25,
                 normalize_input=True
                 ) -> None:
        super().__init__()
        assert feature_size % groups == 0, "`dim` must be divisible by `groups`"
        assert expansion > 1
        self.p_drop = p_drop
        self.cluster_dropout = nn.Dropout2d(p_drop)
        self.num_clusters = num_clusters
        self.dim = feature_size
        self.expansion = expansion
        self.grouped_dim = feature_size * expansion // groups
        self.groups = groups
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.add_batchnorm = add_batch_norm
        self.add_final_fc = add_final_fc
        self.expansion_mapper = nn.Linear(feature_size, feature_size * expansion)
        if add_final_fc:
            self.linear = nn.Linear(in_features=feature_size * num_clusters * expansion // groups, out_features=output_feature_size)
            self.vlad_hidden_size = output_feature_size
        else:
            self.vlad_hidden_size = feature_size * num_clusters * expansion // groups
        if add_batch_norm:
            self.soft_assignment_mapper = nn.Sequential(
                nn.Linear(feature_size * expansion, num_clusters * groups, bias=False),
                TimeFirstBatchNorm1d(num_clusters, groups=groups)
            )
        else:
            self.soft_assignment_mapper = nn.Linear(
                feature_size * expansion, num_clusters * groups, bias=True)
        self.attention_mapper = nn.Linear(
            feature_size * expansion, groups
        )
        self.centroids = nn.Parameter(
            torch.rand(num_clusters, self.grouped_dim))
        self.final_bn = nn.BatchNorm1d(num_clusters * self.grouped_dim)
        self._init_params()

    def _init_params(self):
        if self.add_final_fc:
            for component in (self.soft_assignment_mapper, self.attention_mapper,
                              self.expansion_mapper, self.linear):
                for module in component.modules():
                    general_weight_initialization(module)
        else:
            for component in (self.soft_assignment_mapper, self.attention_mapper,
                              self.expansion_mapper):
                for module in component.modules():
                    general_weight_initialization(module)
        if self.add_batchnorm:
            self.soft_assignment_mapper[0].weight = nn.Parameter(
                (2.0 * self.alpha * self.centroids).repeat((self.groups, self.groups))
            )
            nn.init.constant_(self.soft_assignment_mapper[1].bn.weight, 1)
            nn.init.constant_(self.soft_assignment_mapper[1].bn.bias, 0)
        else:
            self.soft_assignment_mapper.weight = nn.Parameter(
                (2.0 * self.alpha * self.centroids).repeat((self.groups, self.groups))
            )
            self.soft_assignment_mapper.bias = nn.Parameter(
                (- self.alpha * self.centroids.norm(dim=1)
                 ).repeat((self.groups,))
            )
    def forward(self, x):
        """NeXtVlad Adaptive Pooling
                Arguments:
                    x {torch.Tensor} -- shape: (n_batch, len, dim)
                Returns:
                    torch.Tensor -- shape (n_batch, n_cluster * dim / groups)
                """

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=2)  # across descriptor dim
        # expansion
        # shape: (n_batch, len, dim * expansion)
        x = self.expansion_mapper(x)
        # soft-assignment
        # shape: (n_batch, len, n_cluster, groups)
        soft_assign = self.soft_assignment_mapper(x).view(
            x.size(0), x.size(1), self.num_clusters, self.groups
        )
        soft_assign = F.softmax(soft_assign, dim=2)
        attention = torch.sigmoid(self.attention_mapper(x))
        # (n_batch, len, n_cluster, groups, dim / groups)
        activation = (
            attention[:, :, None, :, None] *
            soft_assign[:, :, :, :, None]
        )
        # calculate residuals to each clusters
        # (n_batch, n_cluster, dim / groups)
        second_term = (
            activation.sum(dim=3).sum(dim=1) *
            self.centroids[None, :, :]
        )
        first_term = (
            # (n_batch, len, n_cluster, groups, dim / groups)
            activation *
            x.view(x.size(0), x.size(1), 1, self.groups, self.grouped_dim)
        ).sum(dim=3).sum(dim=1)
        # vlad shape (n_batch, n_cluster, dim / groups)
        vlad = first_term - second_term
        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        # flatten shape (n_batch, n_cluster * dim / groups)
        vlad = vlad.view(x.size(0), -1)  # flatten
        # vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        vlad = self.final_bn(vlad)
        if self.p_drop:
            vlad = self.cluster_dropout(
                vlad.view(x.size(0), self.num_clusters, self.grouped_dim, 1)
            ).view(x.size(0), -1)
        #print(vlad.shape)
        if self.add_final_fc:
            vlad = self.linear(vlad)
        return vlad

    def get_output_dim(self):
        return self.vlad_hidden_size

class RelationModuleMultiScale(torch.nn.Module):
    # Temporal Relation module in multiply scale, suming over [2-frame relation, 3-frame relation, ..., n-frame relation]

    def __init__(self, img_feature_dim, num_frames, output_feature_size, deploy=False):
        super(RelationModuleMultiScale, self).__init__()
        self.subsample_num = 3 # how many relations selected to sum up
        self.img_feature_dim = img_feature_dim
        self.scales = [i for i in range(num_frames, 1, -1)] # generate the multiple frame relations
        self.deploy = deploy

        self.relations_scales = []
        self.subsample_scales = []
        for scale in self.scales:
            relations_scale = self.return_relationset(num_frames, scale)
            self.relations_scales.append(relations_scale)
            self.subsample_scales.append(min(self.subsample_num, len(relations_scale))) # how many samples of relation to select in each forward pass

        self.output_feature_size = output_feature_size
        self.num_frames = num_frames
        num_bottleneck = 256
        self.fc_fusion_scales = nn.ModuleList() # high-tech modulelist
        for i in range(len(self.scales)):
            scale = self.scales[i]
            fc_fusion = nn.Sequential(
                        nn.ReLU(),
                        nn.Linear(scale * self.img_feature_dim, num_bottleneck),
                        nn.ReLU(),
                        nn.Linear(num_bottleneck, self.output_feature_size),
                        )

            self.fc_fusion_scales += [fc_fusion]

        # print('Multi-Scale Temporal Relation Network Module in use', ['%d-frame relation' % i for i in self.scales])

    def forward(self, input):
        # the first one is the largest scale
        act_all = input[:, self.relations_scales[0][0] , :]
        act_all = act_all.view(act_all.size(0), self.scales[0] * self.img_feature_dim)
        act_all = self.fc_fusion_scales[0](act_all)

        for scaleID in range(1, len(self.scales)):
            # iterate over the scales
            idx_relations_randomsample = np.arange(self.subsample_scales[scaleID])
            # if self.deploy:
            #     idx_relations_randomsample = np.arange(self.subsample_scales[scaleID])
            # else:
            #     idx_relations_randomsample = np.random.choice(len(self.relations_scales[scaleID]), self.subsample_scales[scaleID], replace=False)
            for idx in idx_relations_randomsample:
                act_relation = input[:, self.relations_scales[scaleID][idx], :]
                act_relation = act_relation.view(act_relation.size(0), self.scales[scaleID] * self.img_feature_dim)
                act_relation = self.fc_fusion_scales[scaleID](act_relation)
                act_all += act_relation
        return act_all

    def return_relationset(self, num_frames, num_frames_relation):
        import itertools
        return list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))

    def get_output_dim(self):
        return self.output_feature_size


class RelationModuleBaseTsn(torch.nn.Module):
    """
     Temporal Relation module in multiply scale, suming over [2-frame relation, 3-frame relation, ..., n-frame] relation
    """

    def __init__(self,
                 feature_dim,
                 output_feature_dim,
                 num_segments,
                 scales=[10, 6, 2]):
        super(RelationModuleBaseTsn, self).__init__()
        self.feature_dim = feature_dim
        self.output_feature_dim = output_feature_dim
        self.num_segments = num_segments

        self.scales = scales

        self.relations_scale1 = RelationModuleBaseTsn.tsn_sample(self.num_segments, self.scales[0])
        self.relations_scale2 = RelationModuleBaseTsn.tsn_sample(self.num_segments, self.scales[1])
        self.relations_scale3 = RelationModuleBaseTsn.tsn_sample(self.num_segments, self.scales[2])

        num_bottleneck = 1024

        self.fc_fusion_scale1 = nn.Sequential(nn.ReLU(),
                                              nn.Linear(len(self.relations_scale1) * self.feature_dim, num_bottleneck),
                                              nn.ReLU(),
                                              nn.Linear(num_bottleneck, self.output_feature_dim))

        self.fc_fusion_scale2 = nn.Sequential(nn.ReLU(),
                                              nn.Linear(len(self.relations_scale2) * self.feature_dim, num_bottleneck),
                                              nn.ReLU(),
                                              nn.Linear(num_bottleneck, self.output_feature_dim))

        self.fc_fusion_scale3 = nn.Sequential(nn.ReLU(),
                                              nn.Linear(len(self.relations_scale3) * self.feature_dim, num_bottleneck),
                                              nn.ReLU(),
                                              nn.Linear(num_bottleneck, self.output_feature_dim))

    def forward(self, x):
        batch_size, num_segments, feature_dim = x.size()
        assert num_segments == self.num_segments
        # segments_ratio = num_segments // self.num_segments
        #
        # relations_scale1 = [i * segments_ratio for i in self.relations_scale1]
        # relations_scale2 = [i * segments_ratio for i in self.relations_scale2]
        # relations_scale3 = [i * segments_ratio for i in self.relations_scale3]
        relations_scale1 = self.relations_scale1
        relations_scale2 = self.relations_scale2
        relations_scale3 = self.relations_scale3

        # the first one is the largest scale
        relation_1 = x[:, relations_scale1, :]
        relation_1 = relation_1.view(batch_size, self.scales[0] * self.feature_dim)
        relation_1 = self.fc_fusion_scale1(relation_1)

        relation_2 = x[:, relations_scale2, :]
        relation_2 = relation_2.view(batch_size, self.scales[1] * self.feature_dim)
        relation_2 = self.fc_fusion_scale2(relation_2)

        relation_3 = x[:, relations_scale3, :]
        relation_3 = relation_3.view(batch_size, self.scales[2] * self.feature_dim)
        relation_3 = self.fc_fusion_scale3(relation_3)

        relation_all = relation_1 + relation_2 + relation_3
        return relation_all

    @staticmethod
    def return_relation_set(num_frames, num_frames_relation):
        import itertools
        return list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))

    @staticmethod
    def tsn_sample(num_frames_relation, num_segments):
        frame_len = num_frames_relation
        tick = frame_len / float(num_segments)
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(num_segments)])
        offsets = offsets.tolist()
        return offsets

    def get_output_dim(self):
        return self.output_feature_dim


class AverageConsensus(torch.nn.Module):

    def __init__(self, hidden_size) -> None:
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, x):
        output = x.mean(dim=1, keepdim=False)
        return output

    def get_output_dim(self):
        return self.hidden_size

if __name__ == '__main__':
    model = NetXtVLADConsensus(1024, 5, 64, 1024)
    input = torch.rand(10, 5, 4680)
    out = model(input)
    print(out.shape)