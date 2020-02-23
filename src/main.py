from torch.utils.data import DataLoader
from torchvision import transforms
from src.dataset.dataset_generator import BratsDataset
from src.dataset import io
from src.dataset import visualization_utils as visualization
import logging
logging.basicConfig(level=logging.INFO)

# Params
path = '/Users/lauramora/Documents/MASTER/TFM/Data/MICCAI_BraTS_2019_Data_Training/'
batch_size= 4

# Read Dataset
logging.info('Creating Dataset..')
data, labels = io.get_dataset(path)

transform = transforms.Compose([transforms.ToTensor()])
dataset = BratsDataset(data, labels, transform)

loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

logging.info('Plot First Batch')
for data_batch, labels_batch, paths in loader:
    visualization.plot_batch(data_batch, labels_batch, paths)
    break