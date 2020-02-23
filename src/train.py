from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model.vnet import vnet

def train(nll, elu_flag, batch_size, cuda):

    model = vnet.VNet(elu=elu_flag, nll=nll)
    model = nn.parallel.DataParallel(model, device_ids=1)

    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    if cuda:
        model = model.cuda()

    # Load dataset somehow
    dataset = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Training loop
    model.train()
    for input, labels in tqdm(loader):
        image = input.cuda().requires_grad_()
