# Copyright (c) 2017 Elad Hoffer
# Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch.nn as nn

import seq2seq.data.config as config
from seq2seq.models.decoder import ResidualRecurrentDecoder
from seq2seq.models.encoder import ResidualRecurrentEncoder
from seq2seq.models.seq2seq_base import Seq2Seq
import logging
from seq2seq.train.smoothing import LabelSmoothing
from seq2seq.models.sparse_ops_classification import SpExtremeClassifier_Fn# ExtremeClassifier_Fn, SpExtremeClassifier_Ref_Fn
import torch.nn.functional as F
import torch


class FusedExtremeClassifier(nn.Module):
    def __init__(self, in_features, out_features, padding_idx, smoothing) -> None:
        super(FusedExtremeClassifier, self).__init__()
        self.classifier = nn.Linear(in_features, out_features)
        nn.init.uniform_(self.classifier.weight.data, -0.1, 0.1)
        nn.init.uniform_(self.classifier.bias.data, -0.1, 0.1)
        self.smoothing = smoothing
        self.padding_idx = padding_idx

    def forward(self, input, target):
        return SpExtremeClassifier_Fn.apply(input, self.classifier.weight, self.classifier.bias, target, self.smoothing, self.padding_idx, self.training)


class GNMT(Seq2Seq):
    """
    GNMT v2 model
    """
    def __init__(self, vocab_size, hidden_size=1024, num_layers=4, dropout=0.2,
                 batch_first=False, share_embedding=True, padding_idx=0, smoothing=0.):
        """
        Constructor for the GNMT v2 model.

        :param vocab_size: size of vocabulary (number of tokens)
        :param hidden_size: internal hidden size of the model
        :param num_layers: number of layers, applies to both encoder and
            decoder
        :param dropout: probability of dropout (in encoder and decoder)
        :param batch_first: if True the model uses (batch,seq,feature) tensors,
            if false the model uses (seq, batch, feature)
        :param share_embedding: if True embeddings are shared between encoder
            and decoder
        :param padding_idx: the token idx for padding
        :param smoothing: the label smoothing factor
        """

        super(GNMT, self).__init__(batch_first=batch_first)

        if share_embedding:
            embedder = nn.Embedding(vocab_size, hidden_size,
                                    padding_idx=config.PAD)
            nn.init.uniform_(embedder.weight.data, -0.1, 0.1)
        else:
            embedder = None

        self.encoder = ResidualRecurrentEncoder(vocab_size, hidden_size,
                                                num_layers, dropout,
                                                batch_first, embedder)

        self.decoder = ResidualRecurrentDecoder(vocab_size, hidden_size,
                                                num_layers, dropout,
                                                batch_first, embedder)
        
        self.hidden_size = hidden_size
        
        if smoothing == 0.:
            logging.info(f'Building CrossEntropyLoss')
            self.criterion = nn.CrossEntropyLoss(ignore_index=padding_idx, size_average=False)
        else:
            logging.info(f'Building LabelSmoothingLoss (smoothing: {smoothing})')
            # self.criterion = LabelSmoothing(padding_idx, smoothing)
            self.criterion = FusedExtremeClassifier(hidden_size, vocab_size, padding_idx, smoothing)

    def forward(self, input_encoder, input_enc_len, input_decoder, tgt_labels, batch_first):
        context = self.encode(input_encoder, input_enc_len)
        context = (context, input_enc_len, None)
        output, _, _ = self.decode(input_decoder, context)

        if batch_first:
            B = output.size(0)
        else:
            B = output.size(1)

        loss = self.criterion(output.view(-1, self.hidden_size), tgt_labels)

        return loss, B
