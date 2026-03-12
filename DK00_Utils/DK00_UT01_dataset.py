import pandas as pd
from torch.utils.data import Dataset

from DK00_Utils.DK00_UT00_config import CONSTANTS as C
from DK00_Utils.DK00_UT01_datasetProcess import RealSample

class RealDataset(Dataset):
    """
    Dataset class for IMU and bvh data.
    """

    def __init__(self, test_set=None, transform=None, sampling_freq=50, train=True ,version=str):
        """
        Initializer.
        :param test_set: (easy, hard, unseen), determines which testset is used
        :param transform: Optional transform to be applied to every sample.
        """
        self.path = C.DATA_DIR

        if train == False:
            assert test_set in ['easy', 'hard', 'unseen']
        self.sampling_freq = sampling_freq
        self.train = train
        self.test_set = test_set
        self.VERSION = version

        self.test_set_unseen = pd.read_csv(C.UNSEEN_MOTION_CSV)

        if self.train:
            subjects = C.subjects_train
        elif self.test_set == 'easy':
            subjects = C.subjects_easy
        elif self.test_set == 'hard':
            subjects = C.subjects_hard
        elif self.test_set == 'unseen':
            subjects = C.subjects_train

        # Trials to train and test on
        npz_files = []
        for subject in subjects:
            if subject in ['03', '61', '62', '66']:
                trials = ['Norm_Pre', 'White', 'Norm_Post']
            elif subject in ['35']:
                trials = ['Norm_Pre', 'White', 'Pink']
            elif subject in ['87']:
                trials = ['Norm_Pre', 'Pink', 'Norm_Post']
            else:
                trials = ['Norm_Pre', 'White', 'Pink', 'Norm_Post']

            for trial in trials:
                npz_files.append((subject, trial))

        self.npz_files = npz_files
        self.transform = transform
        self.counter = 0
        # Dictionary to track how many times each item is accessed
        self.item_counter = {i: 0 for i in range(len(self.npz_files))}
    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, item):
        # Read out the index of the test sequence
        self.counter += 1
        self.item_counter[item] += 1
        test_split_index = self.test_set_unseen.loc[int(self.npz_files[item][0]) - 1, self.npz_files[item][1]]
        sample = RealSample.from_npz(self.path, self.npz_files[item], self.sampling_freq, self.VERSION, test_split_index)

        try:
            if self.transform is not None:
                sample = self.transform(sample)

            if sample is None:
                print("Sample skipped due to insufficient frames.")
                print(self.npz_files[item])
                # print(self.counter) # TO test if correct nubmer of dataset is loaeded
                # print(self.item_counter[item])
                return None

            return sample

        except ValueError as e:
            print(f"Error: {e}")
            # Skip the sample or take another action
            return None



class ValDataset(Dataset):
    def __init__(self, subject_nr, trial, transform=None, sampling_freq=50,version=str):
        """
        Initializer.
        """
        self.path = C.DATA_DIR
        self.npz_files = [(subject_nr, trial)]
        self.sampling_freq = sampling_freq
        self.version = version

        self.transform = transform

        self.test_set_unseen = pd.read_csv(C.UNSEEN_MOTION_CSV)

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, item):
        test_split_index = self.test_set_unseen.loc[int(self.npz_files[item][0]) - 1, self.npz_files[item][1]]
        sample = RealSample.from_npz(self.path, self.npz_files[item], self.sampling_freq, self.version, test_split_index)

        try:
            if self.transform is not None:
                sample = self.transform(sample)

            if sample is None:
                print("Sample skipped due to insufficient frames.")
                print(self.npz_files[item])
                # print(self.counter) # TO test if correct nubmer of dataset is loaeded
                # print(self.item_counter[item])
                return None

            return sample

        except ValueError as e:
            print(f"Error: {e}")
            # Skip the sample or take another action
            return None