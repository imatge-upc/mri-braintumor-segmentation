import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


#  Crec que estava malament
def dice_coefficient(outputs, targets, threshold=0.5, eps=1e-8):

    batch_size = targets.size(0)
    y_pred = outputs[:, 0, :, :, :]
    y_truth = targets[:, 0, :, :, :]
    y_pred = y_pred > threshold
    y_pred = y_pred.float()
    intersection = torch.sum(torch.mul(y_pred, y_truth)) + eps / 2
    union = torch.sum(y_pred) + torch.sum(y_truth) + eps
    dice = 2 * intersection / union

    return dice / batch_size
