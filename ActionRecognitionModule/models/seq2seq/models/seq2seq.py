import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
import random
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    resnet50,
    ResNet50_Weights,
)


class FCL(nn.Module):
    """
    Fully Connected Network, we set 2 layers with the dimension number
    which is same as the dimension of the input.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # Networks
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # FCL Backbone
        x = self.fc1(x)
        x = self.dropout(F.relu(x))
        x = self.fc2(x)
        output = x

        return output


class Encoder(nn.Module):
    def __init__(
        self, input_dim, emb_dim, hid_dim, n_layers, dropout, tasks, device
    ):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.tasks = tasks

        self.fcl = FCL(input_dim, input_dim, emb_dim)

        if "rgb" in self.tasks:
            self.img_enc = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.fcl = FCL(input_dim, input_dim, input_dim)

        if "depth-aligned" in self.tasks:
            self.depth_enc = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.depth_enc.conv1 = torch.nn.Conv2d(
                1,
                64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )
            self.fcl = FCL(input_dim, input_dim, input_dim)

        self.rnn = nn.LSTM(
            emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, src, img=None, depth=None):
        embedded = self.fcl(src)

        if "rgb" in self.tasks:
            img_shape = img.shape
            img = img.reshape(-1, img_shape[2], img_shape[3], img_shape[4])
            img = rearrange(img, "b h w c -> b c h w")
            embedded_img = self.img_enc(img)
            new_emb_img = torch.reshape(embedded_img, (img_shape[0], -1, 1000))
            embedded = torch.cat((embedded, new_emb_img), 2)

        if "depth-aligned" in self.tasks:
            depth_shape = depth.shape
            depth = depth.reshape(
                -1, depth_shape[2], depth_shape[3], depth_shape[4]
            )
            depth = rearrange(depth, "b h w c -> b c h w")
            embedded_depth = self.depth_enc(depth)
            new_emb_depth = torch.reshape(
                embedded_depth, (depth_shape[0], -1, 1000)
            )
            embedded = torch.cat((embedded, new_emb_depth), 2)

        outputs, (hidden, cell) = self.rnn(embedded)

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.rnn = nn.LSTM(
            emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True
        )

        self.fc_in = FCL(emb_dim, emb_dim, emb_dim)
        self.fc_out = FCL(hid_dim, output_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = self.fc_in(input)
        input = input.unsqueeze(1)

        output, (hidden, cell) = self.rnn(input, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        self.hand_xyz = []
        for kk in range(26):
            self.hand_xyz += [4 + kk * 16, 8 + kk * 16, 12 + kk * 16]
        self.hand_xyz = torch.Tensor(self.hand_xyz).to(torch.int64)

        assert (
            encoder.hid_dim == decoder.hid_dim
        ), "Hidden dimensions of encoder and decoder must be equal!"
        assert (
            encoder.n_layers == decoder.n_layers
        ), "Encoder and decoder must have equal number of layers!"

    def forward(
        self, src, trg, task, teacher_forcing_ratio=0.5
    ):
        left_hand = (
            src["hands-left"].index_select(2, self.hand_xyz).to(self.device).float()
        )
        right_hand = (
            src["hands-right"].index_select(2, self.hand_xyz).to(self.device).float()
        )
        trg_left =  trg["hands-left"].index_select(2, self.hand_xyz).to(self.device).float()
        trg_right = trg["hands-right"].index_select(2, self.hand_xyz).to(self.device).float()

        img, depth = None, None
        if "rgb" in task:
            img = src["rgb"].to(self.device)
        if "depth-aligned" in task:
            depth = src["depth-aligned"].to(self.device)
        
        src_seq = torch.cat(
            (
                left_hand,
                right_hand,
            ),
            2,
        )

        # trg_seq shape -> (2, 30, 156)
        trg_seq = torch.cat(
            (
                trg_left,
                trg_right,
            ),
            2,
        )
        if "eye" in task:
            eye = src["eye"].to(self.device).float()
            src_seq = torch.cat(
                (src_seq, eye),
                2,
            )
        if "head-pose" in task:
            head_pose = src["rgb-info"][:, :, :16].to(self.device).float()
            src_seq = torch.cat(
                (src_seq, head_pose),
                2,
            )

        # src = [batch size, src len, dim]
        # trg = [batch size, trg len, dim]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = trg_seq.shape[0]
        trg_len = trg_seq.shape[1]
        trg_vocab_size = self.decoder.emb_dim

        # print("trg_vocab_size",trg_vocab_size)
        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(
            self.device
        )

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src_seq, img, depth)

        # first input to the decoder is the <sos> tokens
        input_seq = src_seq[:, -1][:, :156]  # Fixed input for two hands.
        # print("input",input.shape)
        for t in range(trg_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input_seq, hidden, cell)
            # place predictions in a tensor holding predictions for each token
            outputs[:, t, :] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input_seq = (
                trg_seq[:, t] if teacher_force else output
            )  # Fixed input for two hands.

        return outputs
