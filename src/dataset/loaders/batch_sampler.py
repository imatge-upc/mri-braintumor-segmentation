import random
from torch.utils.data import Sampler, DataLoader
from src.dataset.loaders.brats_dataset_whole_volume import BratsDataset



class BratsSampler(Sampler):

    def __init__(self, dataset, n_patients, n_samples):

        self.batch_size = n_patients*n_samples
        self.n_samples = n_samples
        self.n_patients = n_patients
        self.dataset_indices = list(range(0, len(dataset)))


    def __iter__(self):
        batch_indices = []
        random.shuffle(self.dataset_indices)

        for patient_id in self.dataset_indices:

            batch_indices.extend(patient_id for _ in range(self.n_samples))

            if len(batch_indices) == self.batch_size:
                yield batch_indices
                batch_indices = []

        if len(batch_indices) > 0:
            yield batch_indices

    def __len__(self):
        return (len(self.dataset_indices) + self.batch_size - 1) // self.batch_size




class BratsPatchSampler(Sampler):

    def __init__(self, dataset, n_patients, n_samples):

        self.batch_size = n_patients*n_samples
        self.n_samples = n_samples
        self.n_patients = n_patients
        self.dataset_indices = list(range(0, len(dataset)))
        self.dataset = dataset.data
        self.patches_by_patient = self._generate_structure()

    def _generate_structure(self):
        patches_by_patient = {}
        for index, patient_patch in enumerate(self.dataset):
            patient = patient_patch.patient

            if patient not in patches_by_patient.keys():
                patches_by_patient[patient] = []
            patches_by_patient[patient].append(index)

        return patches_by_patient


    def __iter__(self):

        while len(self.patches_by_patient) > 0:
            batch_indices = []
            patients_for_batch = random.sample(self.patches_by_patient.items(), min(self.n_patients, len(self.patches_by_patient)))

            for patient, patches in patients_for_batch:
                selected_patches = random.sample(patches, min(self.n_samples, len(patches)))
                batch_indices.extend(selected_patches)

                for patch_id in selected_patches:
                    self.patches_by_patient[patient].remove(patch_id)

                if not self.patches_by_patient[patient]:
                    del self.patches_by_patient[patient]

            yield batch_indices

    def __len__(self):
        return (len(self.dataset_indices) + self.batch_size - 1) // self.batch_size




if __name__ == "__main__":
    from src.dataset.utils import dataset
    from torchvision import transforms as T
    import importlib

    data, _ = dataset.read_brats("/Users/lauramora/Documents/MASTER/TFM/Data/2020/train/random_tumor_distribution/brats20_data.csv")
    data_train = data[:40]
    data_val = data[:40]
    modalities_to_use = {BratsDataset.flair_idx: True, BratsDataset.t1_idx: True, BratsDataset.t2_idx: True,
                         BratsDataset.t1ce_idx: True}
    sampling_method = importlib.import_module("src.dataset.patching.random_tumor_distribution")

    dataset = BratsDataset(data_train, modalities_to_use, sampling_method, (64,64,64), T.Compose([T.ToTensor()]))
    sampler = BratsPatchSampler(dataset, n_patients=2, n_samples=3)
    train_loader = DataLoader(dataset=dataset, batch_sampler=sampler, num_workers=1)


    for i, (patient, volumes, segmentations) in enumerate(train_loader):
        print(f"item {i} --> {volumes.shape}")
        print(f"item {i} --> {segmentations.shape}")
