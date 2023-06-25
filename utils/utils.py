import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import shutil
import time, datetime
import logging
import math
import numpy as np
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils

def avg_p(output, target):
    sorted, indices = torch.sort(output, dim=1, descending=True)
    tp = 0
    s = 0
    for i in range(target.size(1)):
        idx = indices[0,i]
        if target[0,idx] == 1:
            tp = tp + 1
            pre = tp / (i+1)
            s = s + pre
    if tp == 0:
        AP = 0
    else:
        AP = s/tp
    return AP

def cal_ap(y_pred,y_true):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    ap = torch.zeros(y_pred.size(0))   ## 总样本个数
    # compute average precision for each class
    for k in range(y_pred.size(0)):
        # sort scores
        scores = y_pred[k,:].reshape([1,-1])
        targets = y_true[k,:].reshape([1,-1])
        # compute average precision
        # ap[k] = average_precision(scores, targets, difficult_examples)
        ap[k] = avg_p(scores, targets)
    return ap



def cal_one_error(output, target):
    num_class, num_instance = output.size(1), output.size(0)
    one_error = 0
    for i in range(num_instance):
        indicator = 0
        Label = []
        not_Label = []
        temp_tar = target[i, :].reshape(1, num_class)
        # Label_size = sum(sum(temp_tar ==torch.ones([1,num_class])))
        for j in range(num_class):  ## 遍历类别
            if (temp_tar[0, j] == 1):
                Label.append(j)
            else:
                not_Label.append(j)
        temp_out = output[i, :].cpu().numpy()
        maximum = max(temp_out)
        index = np.argmax(temp_out)
        for j in range(num_class):
            if (temp_out[j] == maximum):
                if index in Label:
                    indicator = 1
                    break
        if indicator == 0:
            one_error = one_error + 1

    one_error = one_error / num_instance
    return torch.from_numpy(np.array(one_error)).to(output.device)

def cal_coverage(output, target):
    num_class, num_instance = output.size(1), output.size(0)
    cover = 0
    for i in range(num_instance):
        Label = []
        not_Label = []
        temp_tar = target[i,:].reshape(1,num_class)
        Label_size = sum(sum(temp_tar ==torch.ones([1,num_class])))
        for j in range(num_class):  ## 遍历类别
            if(temp_tar[0,j]==1):
                Label.append(j)
            else:
                not_Label.append(j)
        temp_out = output[i,:]
        _,inde = torch.sort(temp_out)  ## 升序
        inde = inde.cpu().numpy().tolist()
        temp_min = num_class
        for m in range(Label_size):
            loc = inde.index(Label[m])
            if (loc<temp_min):
                temp_min = loc
        cover = cover + (num_class-temp_min)

    cover_result = (cover/num_instance)-1
    return torch.from_numpy(np.array(cover_result)).to(output.device)


def cal_RankingLoss(output, target):
    num_class, num_instance = output.size(1), output.size(0)
    rankloss = 0
    for i in range(num_instance):
        Label = []  ## 存储正标签的索引
        not_Label = []  ## 存储负标签的索引
        temp_tar = target[i,:].reshape(1,num_class)
        Label_size = sum(sum(temp_tar ==torch.ones([1,num_class]).to(output.device)))
        for j in range(num_class):
            if (temp_tar[0, j] >0):
                Label.append(j)
            else:
                not_Label.append(j)
        temp = 0
        for m in range(Label_size):
            for n in range(num_class-Label_size):   ## 比较每一个正标签和所有负标签的预测概率大小，小于等于次数加1
                if output[i,Label[m]]<=output[i,not_Label[n]]: ## n表示负标签总个数
                    temp += 1
        if Label_size==0:
            continue
        else:
            rankloss = rankloss + temp / (Label_size * (num_class-Label_size))

    RankingLoss = rankloss / num_instance
    return RankingLoss


def cal_HammingLoss(output, target):
    labels_num = output.shape[1]
    test_data_num = output.shape[0]
    hammingLoss = 0
    for i in range(test_data_num):
        notEqualNum = 0
        for j in range(labels_num):
            if (output[i][j] >=0.5 and target[i][j]<0.5) or (output[i][j] <0.5 and target[i][j]>=0.5) :
                notEqualNum = notEqualNum + 1
        hammingLoss = hammingLoss + notEqualNum/labels_num
    HammingLoss = hammingLoss/test_data_num
    return torch.from_numpy(np.array(HammingLoss)).to(output.device)

# def cal_HammingLoss(output, target):
#     pre_output = torch.zeros(output.size(0),output.size(1))
#     for i in range(output.size(0)):
#         for j in range(output.size(1)):
#             if output[i,j]>=0:
#                 pre_output[i,j]=1
#             else:
#                 pre_output[i,j]=0
#     num_class, num_instance = output.size(1), output.size(0)
#     miss_sum = 0
#     for i in range(num_instance):
#         # temp_out = torch.sign(output[i,:]).cuda(0)
#         miss_pairs = sum(pre_output[i,:]!=target[i,:])
#         miss_sum += miss_pairs
#     HammingLoss = miss_sum/(num_class*num_instance)
#
#     return HammingLoss



'''record configurations'''
class record_config():
    def __init__(self, args):
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        today = datetime.date.today()

        self.args = args
        self.job_dir = Path(args.job_dir)

        def _make_dir(path):
            if not os.path.exists(path):
                os.makedirs(path)

        _make_dir(self.job_dir)

        config_dir = self.job_dir / 'config.txt'
        #if not os.path.exists(config_dir):
        if args.use_resume:
            with open(config_dir, 'a') as f:
                f.write(now + '\n\n')
                for arg in vars(args):
                    f.write('{}: {}\n'.format(arg, getattr(args, arg)))
                f.write('\n')
        else:
            with open(config_dir, 'w') as f:
                f.write(now + '\n\n')
                for arg in vars(args):
                    f.write('{}: {}\n'.format(arg, getattr(args, arg)))
                f.write('\n')

def get_logger(file_path):

    logger = logging.getLogger('gal')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def save_checkpoint(state, is_best, save, name):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, name)
        shutil.copyfile(filename, best_filename)

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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

def progress_bar(current, total, msg=None):
    _, term_width = os.popen('stty size', 'r').read().split()
    term_width = int(term_width)

    TOTAL_BAR_LENGTH = 65.
    last_time = time.time()
    begin_time = last_time

    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

class AveragePrecisionMeter(object):
    """
    The APMeter measures the average precision per class.
    The APMeter is designed to operate on `NxK` Tensors `output` and
    `target`, and optionally a `Nx1` Tensor weight where (1) the `output`
    contains model output scores for `N` examples and `K` classes that ought to
    be higher when the model is more convinced that the example should be
    positively labeled, and smaller when the model believes the example should
    be negatively labeled (for instance, the output of a sigmoid function); (2)
    the `target` contains only values 0 (for negative examples) and 1
    (for positive examples); and (3) the `weight` ( > 0) represents weight for
    each sample.
    """

    def __init__(self, difficult_examples=True):
        super(AveragePrecisionMeter, self).__init__()
        self.reset()
        self.difficult_examples = difficult_examples

    def reset(self):
        """Resets the meter with empty member variables"""
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())
        self.filenames = []

    def add(self, output, target, filename):
        """
        Args:
            output (Tensor): NxK tensor that for each of the N examples
                indicates the probability of the example belonging to each of
                the K classes, according to the model. The probabilities should
                sum to one over all classes
            target (Tensor): binary NxK tensort that encodes which of the K
                classes are associated with the N-th input
                    (eg: a row [0, 1, 0, 1] indicates that the example is
                         associated with classes 2 and 4)
            weight (optional, Tensor): Nx1 tensor representing the weight for
                each example (each weight > 0)
        """
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        # make sure storage is of sufficient size
        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))

        # store scores and targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)

        self.filenames += filename # record filenames

    def value(self):
        """Returns the model's average precision for each class
        Return:
            ap (FloatTensor): 1xK tensor, with avg precision for each class k
        """

        if self.scores.numel() == 0:
            return 0
        ap = torch.zeros(self.scores.size(1))
        rg = torch.arange(1, self.scores.size(0)).float()
        # compute average precision for each class
        for k in range(self.scores.size(1)):
            # sort scores
            scores = self.scores[:, k]
            targets = self.targets[:, k]
            # compute average precision
            ap[k] = AveragePrecisionMeter.average_precision(scores, targets, self.difficult_examples)
        return ap

    @staticmethod
    def average_precision(output, target, difficult_examples=True):

        # sort examples
        sorted, indices = torch.sort(output, dim=0, descending=True)

        # Computes prec@i
        pos_count = 0.
        total_count = 0.
        precision_at_i = 0.
        for i in indices:
            label = target[i]
            if difficult_examples and label == 0:
                continue
            if label == 1:
                pos_count += 1
            total_count += 1
            if label == 1:
                precision_at_i += pos_count / total_count
        precision_at_i /= pos_count
        return precision_at_i

    def overall(self):
        if self.scores.numel() == 0:
            return 0
        scores = self.scores.cpu().numpy()
        targets = self.targets.clone().cpu().numpy()
        targets[targets == -1] = 0
        return self.evaluation(scores, targets)

    def overall_topk(self, k):
        targets = self.targets.clone().cpu().numpy()
        targets[targets == -1] = 0
        n, c = self.scores.size()
        scores = np.zeros((n, c)) - 1
        index = self.scores.topk(k, 1, True, True)[1].cpu().numpy()
        tmp = self.scores.cpu().numpy()
        for i in range(n):
            for ind in index[i]:
                scores[i, ind] = 1 if tmp[i, ind] >= 0 else -1
        return self.evaluation(scores, targets)

    def evaluation(self, scores_, targets_):
        n, n_class = scores_.shape
        Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
        for k in range(n_class):
            scores = scores_[:, k]
            targets = targets_[:, k]
            targets[targets == -1] = 0
            Ng[k] = np.sum(targets == 1)
            Np[k] = np.sum(scores >= 0)
            Nc[k] = np.sum(targets * (scores >= 0))
        Np[Np == 0] = 1
        OP = np.sum(Nc) / np.sum(Np)
        OR = np.sum(Nc) / np.sum(Ng)
        OF1 = (2 * OP * OR) / (OP + OR)

        CP = np.sum(Nc / Np) / n_class
        CR = np.sum(Nc / Ng) / n_class
        CF1 = (2 * CP * CR) / (CP + CR)
        return OP, OR, OF1, CP, CR, CF1

def adjust_optimizer(args, model):

    if args.optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=float(args.lr), momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=args.weight_decay)
    return optimizer


