import os
import time, datetime
import argparse
import sys
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from src.models import MULTModel
from utils import utils
from loader import dataloader
from torch import nn
from utils import CreateModel


parser = argparse.ArgumentParser(description='MULTILABEL-classification')
parser.add_argument('-f', default='', type=str)

# Fixed
parser.add_argument('--model', type=str, default='MulT', help='name of the model to use (Transformer, etc.)')

# Tasks
parser.add_argument('--data_dir', type=str, default='/media/Harddisk/lsy/meipai_dataset/shuffle_data', help='path to datasets')


# Dropouts
parser.add_argument('--attn_dropout', type=float, default=0.1, help='attention dropout')
parser.add_argument('--attn_dropout_a', type=float, default=0.0, help='attention dropout (for audio)')
parser.add_argument('--attn_dropout_v', type=float, default=0.0, help='attention dropout (for visual)')
parser.add_argument('--relu_dropout', type=float, default=0.1, help='relu dropout')
parser.add_argument('--embed_dropout', type=float, default=0.25, help='embedding dropout')
parser.add_argument('--res_dropout', type=float, default=0.1, help='residual block dropout')
parser.add_argument('--out_dropout', type=float, default=0.0, help='output layer dropout')

# Architecture
parser.add_argument('--n_levels', type=int, default=1, help='number of layers in the network (default: 5)')
parser.add_argument('--num_heads', type=int, default=1, help='number of heads for the transformer network (default: 5)')
parser.add_argument('--attn_mask', action='store_false', help='use attention mask for Transformer (default: true)')

# Tuning
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--clip', type=float, default=0.8, help='gradient clip value (default: 0.8)')
parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate (default: 1e-3)')
parser.add_argument('--optimizer_type', type=str, default='SGD', help='optimizer type, default=SGD')
parser.add_argument('--epochs', type=int, default=80, help='number of training epochs')
parser.add_argument('--when', type=int, default=5, help='when to decay learning rate (default: 20)')
parser.add_argument('--batch_chunk', type=int, default=1, help='number of chunks per batch (default: 1)')

# Logistics
parser.add_argument('--log_interval', type=int, default=30, help='frequency of result logging (default: 30)')
parser.add_argument('--no_cuda', action='store_true', help='do not use cuda')
parser.add_argument('--name', type=str, default='mult', help='name of the trial (default: "mult")')

#
parser.add_argument('--job_dir', type=str, default='/media/Harddisk/lsy/models', help='path to save models')
parser.add_argument('--scheduler_type', type=str, default='ReduceLROnPlateau', help='learning rate strategy')
parser.add_argument('--scheduler_step', type=str, default='15,30', help='learning rate adjustment steps')
parser.add_argument('--num_workers', type=int, default=4, help='numbers of threads workers')
parser.add_argument('--momentum', type=float, default=0.999, help='momentum')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')

# parser.add_argument('--resume_dir', type=str, default='./libs/linear', help='checkpoint model path')

parser.add_argument('--test_only', action='store_true', help='validate all index on testset')
parser.add_argument('--use_resume', action='store_true', help='whether continue training from the same directory')

parser.add_argument('--print_freq', type=int, default=1, help='print frequency')
parser.add_argument('--gpu', type=str, default='0', help='Select gpu to use')
parser.add_argument('--yaml_dir', type=str, default='./utils/Mult.yaml', help='path to networks configuration')

# parser.add_argument('--use_resume',  default='True',action='store_true', help='whether continue training from the same directory')
# parser.add_argument('--resume_dir', type=str, default='/media/Harddisk/lsy/models', help='checkpoint model path')
# parser.add_argument('--test_only', default='True', action='store_true', help='validate all index on testset')
args = parser.parse_args()


# basic configurations

#devices = torch.device("cuda:%s"%(args.gpu) if torch.cuda.is_available() else "cpu")
devices = torch.device( "cpu")

model = CreateModel(args.yaml_dir).to(devices)
criterion = nn.BCEWithLogitsLoss().to(devices)
optimizer = utils.adjust_optimizer(args, model)
now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
logger = utils.get_logger(os.path.join(args.job_dir, 'logger'+now+'.log'))

cudnn.benchmark = True
cudnn.enabled = True
logger.info('args = %s', args)
logger.info(model)
train_data = dataloader(args.data_dir, number=None, is_train=True)
train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
valid_data = dataloader(args.data_dir, number=None, is_train=False)
valid_loader = DataLoader(valid_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)

if args.scheduler_type == 'ReduceLROnPlateau':
    schedule_step = list(map(int, args.scheduler_step.split(',')))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=schedule_step[0])
elif args.scheduler_type == 'MultiStepLR':
    schedule_step = list(map(int, args.scheduler_step.split(',')))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=schedule_step, gamma=0.1)


def train(epoch, train_loader, model, criterion, optimizer):
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    aps = utils.AverageMeter('mAP', ':.4e')
    model.train()

    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
    logger.info('learning_rate: '+str(cur_lr))
    start = time.time()
    num_iter = len(train_loader)
    for i, (vis, aud, tra, label) in enumerate(train_loader):
        vis = vis.to(devices)
        aud = aud.to(devices)
        tra = tra.to(devices)
        label = label.to(devices)
        # compute output
        logits, loss_total = model(vis, aud, tra)
        loss_classify = criterion(logits, label)
        loss = loss_classify + loss_total
        mAP = utils.cal_ap(logits, label).mean()
        # record index
        losses.update(loss.item(), vis.size(0))
        aps.update(mAP.item(), vis.size(0))
        # compute gradient and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        data_time.update(time.time() - start)
        if (i+1) % args.print_freq == 0:
            logger.info('Train Epoch[{0}]({1}/{2}): '
                        'Loss: {losses.avg:.4f}, '
                        'mAP: {aps.avg:.4f}, '
                        'Time: {data_time.avg:.4f}. '.format(epoch, i+1, num_iter, losses=losses, aps=aps,
                                                             data_time=data_time))
    return losses.avg, aps.avg

# valid stage
def valid(epoch, valid_loader, model, criterion):
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    aps = utils.AverageMeter('mAP', ':.4e')
    if args.test_only:
        one_errors = utils.AverageMeter('One Error', ':.4e')
        coverages = utils.AverageMeter('Coverage', ':.4e')
        rankinglosses = utils.AverageMeter('RankingLoss', ':.4e')
        hamminglosses = utils.AverageMeter('HammingLoss', ':.4e')

    model.eval()
    start = time.time()

    with torch.no_grad():
        for i, (vis, aud, tra, label) in enumerate(valid_loader):
            vis = vis.to(devices)
            tra = tra.to(devices)
            aud = aud.to(devices)
            label = label.to(devices)
            # compute output
            logits, loss_total= model(vis, aud, tra)
            loss_classify = criterion(logits, label)
            loss = loss_classify + loss_total
            mAP = utils.cal_ap(logits, label).mean()
            # record index
            losses.update(loss.item(), vis.size(0))
            aps.update(mAP.item(), vis.size(0))
            if args.test_only:
                one_error = utils.cal_one_error(logits, label)
                coverage = utils.cal_coverage(logits, label)
                rankingloss = utils.cal_RankingLoss(logits, label)
                hammingloss = utils.cal_HammingLoss(logits, label)
                one_errors.update(one_error.item(), vis.size(0))
                coverages.update(coverage.item(), vis.size(0))
                rankinglosses.update(rankingloss.item(), vis.size(0))
                hamminglosses.update(hammingloss.item(), vis.size(0))

        data_time.update(time.time() - start)
        logger.info('Valid Epoch[{0}]: '
                    'Loss: {losses.avg:.4f}, '
                    'mAP: {aps.avg:.4f}, '
                    'Time: {data_time.avg:.4f}. '.format(epoch, losses=losses, aps=aps,
                                                         data_time=data_time))
    if args.test_only:
        return aps.avg, one_errors.avg, coverages.avg, rankinglosses.avg, hamminglosses.avg
    else:
        return losses.avg, aps.avg

def main():
    start_epoch = 0
    best_map, best_one_error, best_coverage, best_rankingloss, best_hammingloss = 0., 0., 0., 0., 0.
    if args.use_resume:
        # logger.info('resume from pretrained model ...')
        checkpoint = torch.load(os.path.join(args.resume_dir, 'best_model.pth'))
        state_dict = checkpoint['state_dict']
        start_epoch = int(checkpoint['epoch'])+1
        best_map = float(checkpoint['best_map'])
        model.load_state_dict(state_dict)

    if args.test_only:
        aps, one_errors, coverages, rankinglosses, hamminglosses = valid(start_epoch, valid_loader, model, criterion)
        logger.info('=> Validation: '
                    'mAP: {aps:.4f}, '
                    'One_Error: {one_errors:.4f}, '
                    'Coverages: {coverages:.4f}, '
                    'RankingLoss: {rankinglosses:.4f}, '
                    'HammingLoss: {hamminglosses:.4f}. '.format(aps=aps, one_errors=one_errors, coverages=coverages, \
                                                                rankinglosses=rankinglosses, hamminglosses=hamminglosses))
    else:
        model_list = []
        epoch = start_epoch
        while epoch < args.epochs:
            train_loss, train_map = train(epoch, train_loader, model, criterion, optimizer)
            valid_loss, valid_map = valid(epoch, valid_loader, model, criterion)

            is_best = False
            if valid_map > best_map:
                best_map = valid_map
                is_best = True

            state_dict = {
                'epoch' : epoch,
                'state_dict' : model.state_dict(keep_vars=True),
                'best_map' : best_map,
                'optimizer' : optimizer.state_dict()
            }
            name = 'best_model_%.2f.pth'%(best_map*100)
            utils.save_checkpoint(state_dict, is_best=is_best, save=args.job_dir, name=name)
            scheduler.step(best_map)
            epoch += 1
            logger.info('=> Best index: '
                        'mAP: {best_map:.4f}. '
                        .format(best_map=best_map))
            model_list.append(os.path.join(args.job_dir, name))

        for i in range(len(model_list)-1):
            os.remove(model_list[i])


if __name__ == '__main__':

    main()
    print('over')