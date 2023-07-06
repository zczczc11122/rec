import json
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
import numpy as np
import torch.distributed as dist
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置

idx2class = {0: "table_card", 1: "no_table_card"}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.history = []

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self, local_rank):
        total = torch.tensor([self.sum, self.count], dtype=torch.float32).cuda(local_rank)
        dist.all_reduce(total, dist.ReduceOp.SUM)
        self.tmp_sum, self.tmp_count = total.tolist()
        self.avg = self.tmp_sum / self.tmp_count


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def plot_epochs_stats(stats, save_path):
    df_acc = pd.DataFrame.from_dict(stats['acc']).reset_index(). \
        melt(id_vars=['index']).rename(columns={"index": "epochs"})
    df_losses = pd.DataFrame.from_dict(stats['losses']).reset_index(). \
        melt(id_vars=['index']).rename(columns={"index": "epochs"})
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 7))
    sns.lineplot(data=df_acc, x="epochs", y="value",
                 hue="variable", ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
    sns.lineplot(data=df_losses, x="epochs", y="value",
                 hue="variable", ax=axes[1]).set_title('Train-Val Loss/Epoch')
    plt.savefig(save_path)
    plt.close()


def plot_heatmap(stats, save_path, idx2class):
    labels = []
    target_names = []
    for i in sorted(list(idx2class.keys())):
        labels.append(i)
        target_names.append(idx2class[i])
    plt.figure(figsize=(16, 9), dpi=100)
    df = pd.DataFrame(confusion_matrix(stats['y_test'], stats['y_pred'], labels=labels))
    confusion_matrix_df = df.rename(columns=idx2class, index=idx2class)
    heatmap = sns.heatmap(confusion_matrix_df, annot=True, fmt='d')
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=15)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=15)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path)
    plt.close()


def plot_report(stats, save_path, idx2class):
    labels = []
    target_names = []
    for i in sorted(list(idx2class.keys())):
        labels.append(i)
        target_names.append(idx2class[i])

    report = classification_report(stats['y_test'], stats['y_pred'],
                                   labels=labels,
                                   target_names=target_names,
                                   output_dict=True)
    with open(save_path, 'w') as fh:
        json.dump(report, fh, ensure_ascii=False)
    df = pd.DataFrame(report).transpose().sort_values(by=['support'])
    df.to_csv(f"{save_path[:-5]}.csv", index=True, encoding='utf8')


def multi_plot_report(stats, save_path, cls_type):
    result = {}
    # if cls_type == "form":
    #     idx2class = idx2class_form
    # else:
    #     idx2class = idx2class_theme

    y_pred = torch.tensor(stats['y_pred'], dtype=torch.float32)
    label = torch.tensor(stats['y_test'], dtype=torch.float32)
    m = nn.Sigmoid()
    y_pred = m(y_pred)
    accuracy_th = 0.5
    pred_result = y_pred > accuracy_th
    pred_result = pred_result.float()
    pred_result = pred_result.numpy()
    label = label.numpy()

    report = classification_report(label, pred_result,
                                   labels=[k for k, v in idx2class.items()],
                                   target_names=[v for _, v in idx2class.items()],
                                   output_dict=True)
    with open(save_path, 'w') as fh:
        json.dump(report, fh, ensure_ascii=False)


def write_val_result(stats, save_path, idx2class):
    with open(save_path, 'w') as fh:
        fh.write('vid,true_label,pred_label,correct,value\n')
        for vid, true_label_index, pred_label_index, pred_list_value in zip(stats['vid'], stats['y_test'],
                                                                            stats['y_pred'], stats['y_value']):
            true_label = idx2class[true_label_index]
            pred_label = idx2class[pred_label_index]
            pre_value = pred_list_value
            correct = 1 if true_label == pred_label else 0
            fh.write('{},{},{},{},{}\n'.format(vid, true_label, pred_label, correct, pre_value))


def gen_index(inputs_torch, thre):
    np_arr = inputs_torch.numpy()
    index = np.argwhere(np_arr >= thre).reshape(-1).tolist()
    return index


def multi_write_val_result(stats, save_path, cls_type):
    contents = []
    for vid, true_label, pred_label in zip(stats['vid'], stats['y_test'], stats['y_pred']):
        pred_label = torch.tensor(pred_label, dtype=torch.float32)
        true_label = torch.tensor(true_label, dtype=torch.float32)
        m = nn.Sigmoid()
        y_pred = m(pred_label)
        accuracy_th = 0.5
        pred_result = y_pred > accuracy_th
        pred_result = pred_result.float()

        pre_list = gen_index(pred_result, 1)
        label_list = gen_index(true_label, 1)

        correct = 0
        for pre in pre_list:
            if pre in label_list:
                correct = 1
            else:
                correct = 0
                break
        true_label_list = [idx2class[t] for t in label_list]
        pred_label_list = [idx2class[p] for p in pre_list]
        pred_value_list = [float(y_pred[p].cpu().numpy()) for p in pre_list]
        contents.append([vid, str(true_label_list), str(pred_label_list), correct,
                         str(pred_value_list)])
    df = pd.DataFrame(contents)
    df.columns = ['vid', 'true_label', 'pred_label', 'correct', 'value']
    df.to_csv(save_path, encoding='utf_8_sig', index=False, sep=',')

