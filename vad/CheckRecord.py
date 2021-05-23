import numpy as np
import librosa
import torch
import CLDNN

file=open("label.txt", 'w')
class CheckSpeehRecord:
    def __init__(self):
        self.mel_window_length = 30
        self.mel_window_step = 40
        self.mel_n_channels=40
        input_dim = 6
        hidden_dim = 256
        layer_dim = 3
        output_dim = 2
        dim_mid = 64

        self.model = CLDNN.CLDNNModel(input_dim, hidden_dim, layer_dim, output_dim, dim_mid)
        self.model.load_state_dict(torch.load('5.pt', map_location=torch.device('cpu')))

    def is_speech(self,record):
        wav, sampling_rate=librosa.load(record)
        dim=int(3*sampling_rate/100)
        label=[]
        for index in range(0, len(wav) - 9 * dim, dim):

            mels = librosa.feature.melspectrogram(
                wav[index:index + dim], sampling_rate,
                n_fft=int(sampling_rate * self.mel_window_length / 1000),
                hop_length=int(sampling_rate * self.mel_window_step / 1000),
                n_mels=self.mel_n_channels)
            values = librosa.core.power_to_db(mels)

            for it in range(index + dim, index + (dim * 10), dim):
                melspec = librosa.feature.melspectrogram(
                    wav[it:it + dim], sampling_rate,
                    n_fft=int(sampling_rate * self.mel_window_length / 1000),
                    hop_length=int(sampling_rate * self.mel_window_step / 1000),
                    n_mels=self.mel_n_channels)
                v = librosa.core.power_to_db(melspec)

                values = np.concatenate((values, v), axis=1)
            inputs = torch.from_numpy(values)

            inputs = inputs.unsqueeze(0)
            inputs=inputs.permute(0, 2, 1)

            with torch.no_grad():
                outputs = self.model(inputs)
            out = outputs
            label.append(np.argmax(out))

        label=np.array(label)
        for i in label:
            file.write(str(i)+'\n')


