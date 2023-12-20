from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio
from preprocess import AudioUtil 

class AudioDataLoader(Dataset):
    def __init__(self, df, data_path):
        self.df = df
        self.data_path = str(data_path)
        self.duration = 4000
        self.sr = 44100
        self.channel = 2
        self.shift_pct = 0.4

    def __len__(self):
        return len(self.df)  # number of items in the dataset

    def __getitem__(self, idx):
        audio_file = self.data_path + self.df.loc[idx, 'relative_path']
        class_id = self.df.loc[idx, 'classID']

        aud = AudioUtil.open(audio_file)
        
        # apply data preprocessing techniques
        reaud = AudioUtil.resample(aud, self.sr)
        rechannel = AudioUtil.rechannel(reaud, self.channel)
        duration_aud = AudioUtil.padTrunc(rechannel, self.duration)
        shift_aud = AudioUtil.timeShift(duration_aud, self.shift_pct)
        sgram = AudioUtil.spectroGram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
        aug_sgram = AudioUtil.spectroAugment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

        return aug_sgram, class_id