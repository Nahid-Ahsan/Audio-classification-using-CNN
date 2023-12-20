import math, random
import torch
import torchaudio
from torchaudio import transforms
from IPython.display import Audio

class AudioUtil():

    @staticmethod
    def open(audio_file):
        """
        Load an audio file and return the signal
        as a tensor and sample rate 
        """
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)


    @staticmethod
    def rechannel(aud, new_channel):
        """
        convert the given audio to the desired number of channels
        """ 
        sig, sr = aud

        if (sig.shape[0] == new_channel):
            return aud

        if new_channel == 1:
            # convert from stereo to mono by selecting only the first channel
            resig = sig[:1, :]
        else:
            # convert from mono to stereo by selecting only duplicating the first channel
            resig = torch.cat([sig,sig])

        return ((resig, sr))

    @staticmethod
    def resample(aud, newsr):
        """
        resample applies to a single channel, we resample one channel at a time
        """ 
        sig, sr = aud

        if sr == newsr:
            return aud

        num_channels = sig.shape[0]
        # resample the first channel
        first_channel = torchaudio.transforms.Resample(sr, newsr)(sig[:1, :])

        resig = first_channel  # Initialize resig with the first channel

        if num_channels > 1:
            # resample the second channel and merge both channels
            second_channel = torchaudio.transforms.Resample(sr, newsr)(sig[1:, :])
            resig = torch.cat([first_channel, second_channel])

        return (resig, newsr)


    @staticmethod
    def padTrunc(aud, max_ms):
        """
        truncate (or pad) the signal to a fixed length signal 'max_ms' in milliseconds
        """
        sig, sr = aud 
        num_rows, sig_len = sig.shape
        max_len = sr//1000 * max_ms

        if sig_len > max_len:
            # Truncate the singal to the given length
            sig = sig[:,:max_len]

        elif sig_len < max_len:
            # length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((pad_begin,sig, pad_end),1)

        return (sig, sr)


    @staticmethod
    def timeShift(aud, shift_limit):
        """
        shifts the signal to the left or right by some percent, values at the end
        are 'wrapped around' to the start of the transformed signal. 
        """
        sig,sr = aud
        _, sig_len = sig.shape
        shift_amt = int((random.random() * shift_limit * sig_len))
        return (sig.roll(shift_amt), sr)

    @staticmethod
    def spectroGram(aud, n_mels = 64, n_fft = 1024, hop_len = None):
        """
        generate a spectrogram
        """ 
        sig, sr = aud
        top_db = 80
        # spec has shape [channel, n_mels, time] where channel is mono, stereo etc
        spec = transforms.MelSpectrogram(sr, n_fft, hop_length = hop_len, n_mels = n_mels)(sig)
        # convert to decibels
        spec = transforms.AmplitudeToDB(top_db)(spec)
        return (spec)


    @staticmethod
    def spectroAugment(spec, max_mask_pct = 0.1, n_freq_masks = 1, n_time_masks = 1):
        """
        augment the spectrogram by masking out some sections of it in both the 
        frequency dimension ( horizontal bars) and the time dimension ( vertical bars)
        to prevent overfitting and to help the model generalise better. The masked
        sections are replaced with the mean value.
    
        """ 
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec
        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)
        
        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec =  transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)
        return aug_spec