import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DenoiseBlock(nn.Module):
    def __init__(self,embedding_dim,num_classes,num_input_channels=1,use_decoder=False):
        super().__init__()

        self.use_decoder = use_decoder

        # Image Path (CNN)
        self.conv_path = nn.Sequential(
            nn.Conv2d(num_input_channels,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(128,256),
            nn.BatchNorm1d(256)
        )

        #Noisy Embedding Path

        self.fc_z1 = nn.Linear(256,256)
        self.bn_z1 = nn.BatchNorm1d(256)

        self.fc_z2 = nn.Linear(256,256)
        self.bn_z2 = nn.BatchNorm1d(256)

        self.fc_z3 = nn.Linear(256,256)
        self.bn_z3 = nn.BatchNorm1d(256)

        #combined downstream

        self.fc_f1 = nn.Linear(256+256,256)
        self.bn_f1 = nn.BatchNorm1d(256)

        self.fc_f2 = nn.Linear(256,256)
        self.bn_f2 = nn.BatchNorm1d(256)

        self.fc_out = nn.Linear(256,num_classes)

        #non linear decoder
        if self.use_decoder:
            self.decoder = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.Tanh()
            )


    def forward(self, x, z_prev, W_embed):
        x_feat = self.conv_path(x)

        h1 = F.relu(self.bn_z1(self.fc_z1(z_prev)))
        h2 = F.relu(self.bn_z2(self.fc_z2(h1)))
        h3 = self.bn_z3(self.fc_z3(h2))

        z_feat = h3 + h1  # Residual connection

        h_f = torch.cat([x_feat, z_feat], dim=1)
        h_f = F.relu(self.bn_f1(self.fc_f1(h_f)))
        h_f = F.relu(self.bn_f2(self.fc_f2(h_f)))
        logits = self.fc_out(h_f)

        p = F.softmax(logits, dim=1)
        z_next = p @ W_embed
        if self.use_decoder:
            z_next = self.decoder(z_next)

        return z_next, logits
    

class NoPropDT(nn.Module):
    def __init__(self, num_classes, embedding_dim, T, eta,num_input_channels=1,use_decoder=False):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.T = T
        self.eta = eta

        # Stack of Denoising Blocks
        self.blocks = nn.ModuleList([
            DenoiseBlock(embedding_dim, num_classes,num_input_channels,use_decoder=use_decoder) for _ in range(T)
        ])

        # Class-embedding matrix
        self.W_embed = nn.Parameter(torch.randn(num_classes, embedding_dim) * 0.1)

        # Classifier head
        self.classifier = nn.Linear(embedding_dim, num_classes)

        # Cosine noise schedule (ᾱ and SNR diff)
        t = torch.arange(1, T + 1, dtype=torch.float32)
        alpha_t = torch.cos(t / T * (math.pi / 2)) ** 2
        alpha_bar = torch.cumprod(alpha_t, dim=0)
        snr = alpha_bar / (1 - alpha_bar)
        snr_prev = torch.cat([torch.tensor([0.], dtype=snr.dtype), snr[:-1]], dim=0)
        snr_diff = snr - snr_prev

        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('snr_diff', snr_diff)

    def forward_denoise(self, x, z_prev, t):
        return self.blocks[t](x, z_prev, self.W_embed)[0]

    def classify(self, z):
        return self.classifier(z)

    def inference(self, x):
        B = x.size(0)
        z = torch.randn(B, self.embedding_dim, device=x.device)
        for t in range(self.T):
            z = self.forward_denoise(x, z, t)
        return self.classify(z)