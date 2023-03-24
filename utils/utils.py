'''
    helper funtions including:
      - adjust_lr_rate(): adjust learning rate
'''

def adjust_lr_rate(init_lr, optimizer, epoch, lradj):
    '''Set the learning rate to the initial LR decayed by every [lradj=20] epochs'''
    lr = init_lr * (0.1 ** (epoch // lradj))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr