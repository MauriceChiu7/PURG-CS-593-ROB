import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 3), nn.PReLU(), nn.Dropout(), # 4 input for the four states of the cart pole
            nn.Linear(3, 3), nn.PReLU(), nn.Dropout(),
            nn.Linear(3, 2), nn.Softmax(dim=0)  # 2 output for the control, 0 for left and 1 for right. 
                                                # Softmax gives a predicted probability distribution.
        )

    # Predict. Output is the predicted probability distribution for actions.
    def forward(self, state):
        # out = self.fc(torch.tensor(state))
        out = self.fc(state)
        return out

# class MLP(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(MLP, self).__init__()
#         self.fc = nn.Sequential(
#         nn.Linear(input_size, 1280),nn.PReLU(),nn.Dropout(),
#         nn.Linear(1280, 1024),nn.PReLU(),nn.Dropout(),
#         nn.Linear(1024, 896),nn.PReLU(),nn.Dropout(),
#         nn.Linear(896, 768),nn.PReLU(),nn.Dropout(),
#         nn.Linear(768, 512),nn.PReLU(),nn.Dropout(),
#         nn.Linear(512, 384),nn.PReLU(),nn.Dropout(),
#         nn.Linear(384, 256),nn.PReLU(), nn.Dropout(),
#         nn.Linear(256, 256),nn.PReLU(), nn.Dropout(),
#         nn.Linear(256, 128),nn.PReLU(), nn.Dropout(),
#         nn.Linear(128, 64),nn.PReLU(), nn.Dropout(),
#         nn.Linear(64, 32),nn.PReLU(),
#         nn.Linear(32, output_size))


