#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# get_ipython().system('mkdir data && cd data && wget ')


# In[1]:


import torch
import torchvision
import cv2
import numpy as np
import os
import PIL


# In[2]:


images_dir = "formula_images_processed"
formula_list = "im2latex_formulas.norm.lst"
train_list = "im2latex_train_filter.lst"
validate_list = "im2latex_validate_filter.lst"
test_list = "im2latex_test_filter.lst"


# In[3]:


class image2latexDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, formula_list, train_list):
        self.images_dir = images_dir
        # self.image_filenames = os.listdir(self.images_dir)

        with open(formula_list, "r", encoding="utf-8", errors="ignore", newline="\n") as f1:
            self.formulas = [formula.replace("\n", "").replace("\t", " ") for formula in f1.readlines()]
        
        with open(train_list, "r", encoding="utf-8", errors="ignore", newline="\n") as f2:
            self.train_set = [t.replace("\n", "").split() for t in f2.readlines()]
    
        # assert len(self.image_filenames) == len(self.formulas) 

    def __getitem__(self, idx):
        item = self.train_set[idx]
        filename = item[0]
        formula = self.formulas[int(item[1])]
#         render_type = item[2]
        image = PIL.Image.open(self.images_dir + "/" + filename)
        return torchvision.transforms.ToTensor()(image), formula

    def __len__(self):
        return len(self.train_set)


# In[4]:


train_set = image2latexDataset(images_dir, formula_list, train_list)
validate_set = image2latexDataset(images_dir, formula_list, validate_list)
test_set = image2latexDataset(images_dir, formula_list, test_list)


# In[14]:


# print(train_set[0])
# print(len(train_set), len(validate_set), len(test_set))


# In[7]:


# train_loader = torch.utils.data.DataLoader(
#     train_set,
#     batch_size=32,
#     num_workers=4
# )

# val_loader = torch.utils.data.DataLoader(
#     validate_set,
#     batch_size=32,
#     num_workers=4
# )


# In[5]:


import math

def add_positional_features(tensor: torch.Tensor,
                            min_timescale: float = 1.0,
                            max_timescale: float = 1.0e4):
    """
    Implements the frequency-based positional encoding described
    in `Attention is all you Need
    Parameters
    ----------
    tensor : ``torch.Tensor``
        a Tensor with shape (batch_size, timesteps, hidden_dim).
    min_timescale : ``float``, optional (default = 1.0)
        The largest timescale to use.
    Returns
    -------
    The input tensor augmented with the sinusoidal frequencies.
    """
    _, timesteps, hidden_dim = tensor.size()

    timestep_range = get_range_vector(timesteps, tensor.device).data.float()
    # We're generating both cos and sin frequencies,
    # so half for each.
    num_timescales = hidden_dim // 2
    timescale_range = get_range_vector(
        num_timescales, tensor.device).data.float()

    log_timescale_increments = math.log(
        float(max_timescale) / float(min_timescale)) / float(num_timescales - 1)
    inverse_timescales = min_timescale *         torch.exp(timescale_range * -log_timescale_increments)

    # Broadcasted multiplication - shape (timesteps, num_timescales)
    scaled_time = timestep_range.unsqueeze(1) * inverse_timescales.unsqueeze(0)
    # shape (timesteps, 2 * num_timescales)
    sinusoids = torch.randn(
        scaled_time.size(0), 2*scaled_time.size(1), device=tensor.device)
    sinusoids[:, ::2] = torch.sin(scaled_time)
    sinusoids[:, 1::2] = torch.sin(scaled_time)
    if hidden_dim % 2 != 0:
        # if the number of dimensions is odd, the cos and sin
        # timescales had size (hidden_dim - 1) / 2, so we need
        # to add a row of zeros to make up the difference.
        sinusoids = torch.cat(
            [sinusoids, sinusoids.new_zeros(timesteps, 1)], 1)
    return tensor + sinusoids.unsqueeze(0)


def get_range_vector(size: int, device):
    return torch.arange(0, size, dtype=torch.long, device=device)


# In[6]:


import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.distributions.uniform import Uniform

INIT = 1e-2


class Encoder(nn.Module):
    def __init__(self, out_channels=512):
        super(Encoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 1),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 1),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1), 0),

            nn.Conv2d(256, out_channels, 3, 1, 0),
            nn.ReLU()
        )
    
    def forward(self, images):
        encoded_images = self.cnn(images)
        encoded_images = encoded_images.permute(0, 2, 3, 1)  # [B, H', W', 512]
        B, H, W, C = encoded_imgs.shape
        encoded_images = encoded_images.contiguous().view(B, H*W, C)
    
    def add_positional_features(tensor: torch.Tensor, min_timescale: float = 1.0, max_timescale: float = 1.0e4):
        """
        Implements the frequency-based positional encoding described
        in `Attention is all you Need
        Parameters
        ----------
        tensor : ``torch.Tensor``
            a Tensor with shape (batch_size, timesteps, hidden_dim).
        min_timescale : ``float``, optional (default = 1.0)
            The largest timescale to use.
        Returns
        -------
        The input tensor augmented with the sinusoidal frequencies.
        """
        _, timesteps, hidden_dim = tensor.size()

        timestep_range = get_range_vector(timesteps, tensor.device).data.float()
        # We're generating both cos and sin frequencies,
        # so half for each.
        num_timescales = hidden_dim // 2
        timescale_range = get_range_vector(
            num_timescales, tensor.device).data.float()

        log_timescale_increments = math.log(
            float(max_timescale) / float(min_timescale)) / float(num_timescales - 1)
        inverse_timescales = min_timescale *         torch.exp(timescale_range * -log_timescale_increments)

        # Broadcasted multiplication - shape (timesteps, num_timescales)
        scaled_time = timestep_range.unsqueeze(1) * inverse_timescales.unsqueeze(0)
        # shape (timesteps, 2 * num_timescales)
        sinusoids = torch.randn(
            scaled_time.size(0), 2*scaled_time.size(1), device=tensor.device)
        sinusoids[:, ::2] = torch.sin(scaled_time)
        sinusoids[:, 1::2] = torch.sin(scaled_time)
        if hidden_dim % 2 != 0:
            # if the number of dimensions is odd, the cos and sin
            # timescales had size (hidden_dim - 1) / 2, so we need
            # to add a row of zeros to make up the difference.
            sinusoids = torch.cat(
                [sinusoids, sinusoids.new_zeros(timesteps, 1)], 1)
        return tensor + sinusoids.unsqueeze(0)



class Image2LatexModel(nn.Module):
    def __init__(self, out_size, emb_size, dec_rnn_h,
                 enc_out_dim=512,  n_layer=1,
                 add_pos_feat=False, dropout=0.):
        
        super(Image2LatexModel, self).__init__()
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 1),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 1),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1), 0),

            nn.Conv2d(256, enc_out_dim, 3, 1, 0),
            nn.ReLU()
        )

        self.rnn_decoder = nn.LSTMCell(dec_rnn_h+emb_size, dec_rnn_h)
        self.embedding = nn.Embedding(out_size, emb_size)

        self.init_wh = nn.Linear(enc_out_dim, dec_rnn_h)
        self.init_wc = nn.Linear(enc_out_dim, dec_rnn_h)
        self.init_wo = nn.Linear(enc_out_dim, dec_rnn_h)

        # Attention mechanism
        self.beta = nn.Parameter(torch.Tensor(enc_out_dim))
        init.uniform_(self.beta, -INIT, INIT)
        self.W_1 = nn.Linear(enc_out_dim, enc_out_dim, bias=False)
        self.W_2 = nn.Linear(dec_rnn_h, enc_out_dim, bias=False)

        self.W_3 = nn.Linear(dec_rnn_h+enc_out_dim, dec_rnn_h, bias=False)
        self.W_out = nn.Linear(dec_rnn_h, out_size, bias=False)

        self.add_pos_feat = add_pos_feat
        self.dropout = nn.Dropout(p=dropout)
        self.uniform = Uniform(0, 1)

    def forward(self, imgs, formulas, epsilon=1.):
        """args:
        imgs: [B, C, H, W]
        formulas: [B, MAX_LEN]
        epsilon: probability of the current time step to
                 use the true previous token
        return:
        logits: [B, MAX_LEN, VOCAB_SIZE]
        """
        # encoding
        encoded_imgs = self.encode(imgs)  # [B, H*W, 512]
        # init decoder's states
        dec_states, o_t = self.init_decoder(encoded_imgs)
        max_len = formulas.size(1)
        logits = []
        for t in range(max_len):
            tgt = formulas[:, t:t+1]
            # schedule sampling
            if logits and self.uniform.sample().item() > epsilon:
                tgt = torch.argmax(torch.log(logits[-1]), dim=1, keepdim=True)
            # ont step decoding
            dec_states, O_t, logit = self.step_decoding(
                dec_states, o_t, encoded_imgs, tgt)
            logits.append(logit)
        logits = torch.stack(logits, dim=1)  # [B, MAX_LEN, out_size]
        return logits

    def encode(self, imgs):
        encoded_imgs = self.cnn_encoder(imgs)  # [B, 512, H', W']
        encoded_imgs = encoded_imgs.permute(0, 2, 3, 1)  # [B, H', W', 512]
        B, H, W, _ = encoded_imgs.shape
        encoded_imgs = encoded_imgs.contiguous().view(B, H*W, -1) # B, 100, 512
        # input image: 3, 100, 100
        # after cnn: 512, 10, 10
        # 1st channel:
        #   1st row: 1 to w'=10: 1,5,7,3,2,4,6,8,8,2
        # .....
        #  h' th row: 1 to w'
        if self.add_pos_feat:
            encoded_imgs = add_positional_features(encoded_imgs)
        return encoded_imgs

    def step_decoding(self, dec_states, o_t, enc_out, tgt):
        """Running one step decoding"""

        prev_y = self.embedding(tgt).squeeze(1)  # [B, emb_size]
        inp = torch.cat([prev_y, o_t], dim=1)  # [B, emb_size+dec_rnn_h]
        h_t, c_t = self.rnn_decoder(inp, dec_states)  # h_t:[B, dec_rnn_h]
        h_t = self.dropout(h_t)
        c_t = self.dropout(c_t)

        # context_t : [B, C]
        context_t, attn_scores = self._get_attn(enc_out, h_t)

        # [B, dec_rnn_h]
        o_t = self.W_3(torch.cat([h_t, context_t], dim=1)).tanh()
        o_t = self.dropout(o_t)

        # calculate logit
        logit = F.softmax(self.W_out(o_t), dim=1)  # [B, out_size]

        return (h_t, c_t), o_t, logit

    def _get_attn(self, enc_out, h_t):
        """Attention mechanism
        args:
            enc_out: row encoder's output [B, L=H*W, C]
            h_t: the current time step hidden state [B, dec_rnn_h]
        return:
            context: this time step context [B, C]
            attn_scores: Attention scores
        """
        # cal alpha
        alpha = torch.tanh(self.W_1(enc_out)+self.W_2(h_t).unsqueeze(1))
        alpha = torch.sum(self.beta*alpha, dim=-1)  # [B, L]
        alpha = F.softmax(alpha, dim=-1)  # [B, L]

        # cal context: [B, C]
        context = torch.bmm(alpha.unsqueeze(1), enc_out)
        context = context.squeeze(1)
        return context, alpha

    def init_decoder(self, enc_out):
        """args:
            enc_out: the output of row encoder [B, H*W, C]
          return:
            h_0, c_0:  h_0 and c_0's shape: [B, dec_rnn_h]
            init_O : the average of enc_out  [B, dec_rnn_h]
            for decoder
        """
        mean_enc_out = enc_out.mean(dim=1)
        h = self._init_h(mean_enc_out)
        c = self._init_c(mean_enc_out)
        init_o = self._init_o(mean_enc_out)
        return (h, c), init_o

    def _init_h(self, mean_enc_out):
        return torch.tanh(self.init_wh(mean_enc_out))

    def _init_c(self, mean_enc_out):
        return torch.tanh(self.init_wc(mean_enc_out))

    def _init_o(self, mean_enc_out):
        return torch.tanh(self.init_wo(mean_enc_out))


# In[7]:


import pickle as pkl
from collections import Counter

START_TOKEN = 0
PAD_TOKEN = 1
END_TOKEN = 2
UNK_TOKEN = 3

# buid sign2id


class Vocab(object):
    def __init__(self):
        self.sign2id = {"<s>": START_TOKEN, "</s>": END_TOKEN,
                        "<pad>": PAD_TOKEN, "<unk>": UNK_TOKEN}
        self.id2sign = dict((idx, token)
                            for token, idx in self.sign2id.items())
        self.length = 4

    def add_sign(self, sign):
        if sign not in self.sign2id:
            self.sign2id[sign] = self.length
            self.id2sign[self.length] = sign
            self.length += 1

    def __len__(self):
        return self.length


def build_vocab(min_count=10):
    """
    traverse training formulas to make vocab
    and store the vocab in the file
    """
    vocab = Vocab()
    counter = Counter()

    formulas_file = formula_list
    with open(formula_list, "r", encoding="utf-8", errors="ignore", newline="\n") as f1:
        formulas = [formula.replace("\n", "").replace("\t", " ") for formula in f1.readlines()]
        
    with open(train_list, "r", encoding="utf-8", errors="ignore", newline="\n") as f2:
        for line in f2:
            img_filename, idx = line.strip('\n').split()
            idx = int(idx)
            formula = formulas[idx].split()
            counter.update(formula)

    for word, count in counter.most_common():
        if count >= min_count:
            vocab.add_sign(word)
    vocab_file = 'vocab.pkl'
    print("Writing Vocab File in ", vocab_file)
    with open(vocab_file, 'wb') as w:
        pkl.dump(vocab, w)

vocab = build_vocab()


# In[8]:


def load_vocab():
    with open('vocab.pkl', 'rb') as f:
        vocab = pkl.load(f)
    print("Load vocab including {} words!".format(len(vocab)))
    return vocab

v = load_vocab()


# In[32]:


class BeamSearch:
    """
    Implements the beam search algorithm for decoding the most likely sequences.
    Parameters
    ----------
    end_index : ``int``
        The index of the "stop" or "end" token in the target vocabulary.
    max_steps : ``int``, optional (default = 50)
        The maximum number of decoding steps to take, i.e. the maximum length
        of the predicted sequences.
    beam_size : ``int``, optional (default = 10)
        The width of the beam used.
    per_node_beam_size : ``int``, optional (default = beam_size)
        The maximum number of candidates to consider per node, at each step in the search.
        If not given, this just defaults to ``beam_size``. Setting this parameter
        to a number smaller than ``beam_size`` may give better results, 
        as it can introduce more diversity into the search. 
        See `Beam Search Strategies for Neural Machine Translation.
        Freitag and Al-Onaizan, 2017 <http://arxiv.org/abs/1702.01806>`_.
    """

    def __init__(self,
                 end_index: int,
                 max_steps: int = 50,
                 beam_size: int = 10,
                 per_node_beam_size: int = None) -> None:
        self._end_index = end_index
        self.max_steps = max_steps
        self.beam_size = beam_size
        self.per_node_beam_size = per_node_beam_size or beam_size

    def search(self, start_predictions, start_state, step):
        """
        Given a starting state and a step function, apply beam search to find the
        most likely target sequences.
        Notes
        -----
        If your step function returns ``-inf`` for some log probabilities
        (like if you're using a masked log-softmax) then some of the "best"
        sequences returned may also have ``-inf`` log probability. Specifically
        this happens when the beam size is smaller than the number of actions
        with finite log probability (non-zero probability) returned by the step function.
        Therefore if you're using a mask you may want to check the results from ``search``
        and potentially discard sequences with non-finite log probability.
        Parameters
        ----------
        start_predictions : ``torch.Tensor``
            A tensor containing the initial predictions with shape ``(batch_size,)``.
            Usually the initial predictions are just the index of the "start" token
            in the target vocabulary.
        start_state : ``dict``
            The initial state passed to the ``step`` function. 
            Each value of the state dict should be a tensor of shape ``(batch_size, *)``, 
            where ``*`` means any other number of dimensions.
        step : ``function``
            A function that is responsible for computing the next most likely tokens,
            given the current state and the predictions from the last time step.
            The function should accept two arguments. The first being a tensor
            of shape ``(group_size,)``, representing the index of the predicted
            tokens from the last time step, and the second being the current state.
            The ``group_size`` will be ``batch_size * beam_size``, except in the initial
            step, for which it will just be ``batch_size``.
            The function is expected to return a tuple, where the first element
            is a tensor of shape ``(group_size, target_vocab_size)`` containing
            the log probabilities of the tokens for the next step, and the second
            element is the updated state. The tensor in the state should have shape
            ``(group_size, *)``, where ``*`` means any other number of dimensions.
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of ``(predictions, log_probabilities)``, where ``predictions``
            has shape ``(batch_size, beam_size, max_steps)`` and ``log_probabilities``
            has shape ``(batch_size, beam_size)``.
        """
        batch_size = start_predictions.size()[0]

        # List of (batch_size, beam_size) tensors. One for each time step. Does not
        # include the start symbols, which are implicit.
        predictions = []

        # List of (batch_size, beam_size) tensors. One for each time step. None for
        # the first.  Stores the index n for the parent prediction, i.e.
        # predictions[t-1][i][n], that it came from.
        backpointers = []

        # Calculate the first timestep. This is done outside the main loop
        # because we are going from a single decoder input (the output from the
        # encoder) to the top `beam_size` decoder outputs. On the other hand,
        # within the main loop we are going from the `beam_size` elements of the
        # beam to `beam_size`^2 candidates from which we will select the top
        # `beam_size` elements for the next iteration.
        # shape: (batch_size, num_classes)
        start_class_log_probabilities, state = step(
            start_predictions, start_state)

        num_classes = start_class_log_probabilities.size()[1]

        # shape: (batch_size, beam_size), (batch_size, beam_size)
        start_top_log_probabilities, start_predicted_classes = \
            start_class_log_probabilities.topk(self.beam_size)
        if self.beam_size == 1 and (start_predicted_classes == self._end_index).all():
            print("Empty sequences predicted. You may want to "
                  "increase the beam size or ensure "
                  "your step function is working properly.")
            return start_predicted_classes.unsqueeze(-1), start_top_log_probabilities

        # The log probabilities for the last time step.
        # shape: (batch_size, beam_size)
        last_log_probabilities = start_top_log_probabilities

        # shape: [(batch_size, beam_size)]
        predictions.append(start_predicted_classes)

        # Log probability tensor that mandates that the end token is selected.
        # shape: (batch_size * beam_size, num_classes)
        log_probs_after_end = start_class_log_probabilities.new_full(
            (batch_size * self.beam_size, num_classes),
            float("-inf")
        )
        log_probs_after_end[:, self._end_index] = 0.

        # Set the same state for each element in the beam.
        for key, state_tensor in state.items():
            _, *last_dims = state_tensor.size()
            # shape: (batch_size * beam_size, *)
            state[key] = state_tensor.\
                unsqueeze(1).\
                expand(batch_size, self.beam_size, *last_dims).\
                reshape(batch_size * self.beam_size, *last_dims)

        for timestep in range(self.max_steps - 1):
            # shape: (batch_size * beam_size,)
            last_predictions = predictions[-1].reshape(
                batch_size * self.beam_size)

            # If every predicted token from the last step is `self._end_index`,
            # then we can stop early.
            if (last_predictions == self._end_index).all():
                break

            # Take a step. This get the predicted log probs of the next classes
            # and updates the state.
            # shape: (batch_size * beam_size, num_classes)
            class_log_probabilities, state = step(last_predictions, state)

            # shape: (batch_size * beam_size, num_classes)
            last_predictions_expanded = last_predictions.unsqueeze(-1).expand(
                batch_size * self.beam_size,
                num_classes
            )

            # Here we are finding any beams where we predicted the end token in
            # the previous timestep and replacing the distribution with a
            # one-hot distribution, forcing the beam to predict the end token
            # this timestep as well.
            # shape: (batch_size * beam_size, num_classes)
            cleaned_log_probabilities = torch.where(
                last_predictions_expanded == self._end_index,
                log_probs_after_end,
                class_log_probabilities
            )

            # shape (both): (batch_size * beam_size, per_node_beam_size)
            top_log_probabilities, predicted_classes = \
                cleaned_log_probabilities.topk(self.per_node_beam_size)

            # Here we expand the last log probabilities to (batch_size * beam_size, per_node_beam_size)
            # so that we can add them to the current log probs for this timestep.
            # This lets us maintain the log probability of each element on the beam.
            # shape: (batch_size * beam_size, per_node_beam_size)
            expanded_last_log_probabilities = last_log_probabilities.\
                unsqueeze(2).\
                expand(batch_size, self.beam_size, self.per_node_beam_size).\
                reshape(batch_size * self.beam_size, self.per_node_beam_size)

            # shape: (batch_size * beam_size, per_node_beam_size)
            summed_top_log_probabilities = top_log_probabilities + \
                expanded_last_log_probabilities

            # shape: (batch_size, beam_size * per_node_beam_size)
            reshaped_summed = summed_top_log_probabilities.\
                reshape(batch_size, self.beam_size * self.per_node_beam_size)

            # shape: (batch_size, beam_size * per_node_beam_size)
            reshaped_predicted_classes = predicted_classes.\
                reshape(batch_size, self.beam_size * self.per_node_beam_size)

            # Keep only the top `beam_size` beam indices.
            # shape: (batch_size, beam_size), (batch_size, beam_size)
            restricted_beam_log_probs, restricted_beam_indices = reshaped_summed.topk(
                self.beam_size)

            # Use the beam indices to extract the corresponding classes.
            # shape: (batch_size, beam_size)
            restricted_predicted_classes = reshaped_predicted_classes.gather(
                1, restricted_beam_indices)

            predictions.append(restricted_predicted_classes)

            # shape: (batch_size, beam_size)
            last_log_probabilities = restricted_beam_log_probs

            # The beam indices come from a `beam_size * per_node_beam_size` dimension where the
            # indices with a common ancestor are grouped together. Hence
            # dividing by per_node_beam_size gives the ancestor. (Note that this is integer
            # division as the tensor is a LongTensor.)
            # shape: (batch_size, beam_size)
            backpointer = restricted_beam_indices // self.per_node_beam_size

            backpointers.append(backpointer)

            # Keep only the pieces of the state tensors corresponding to the
            # ancestors created this iteration.
            for key, state_tensor in state.items():
                _, *last_dims = state_tensor.size()
                # shape: (batch_size, beam_size, *)
                expanded_backpointer = backpointer.\
                    view(batch_size, self.beam_size, *([1] * len(last_dims))).\
                    expand(batch_size, self.beam_size, *last_dims)

                # shape: (batch_size * beam_size, *)
                state[key] = state_tensor.\
                    reshape(batch_size, self.beam_size, *last_dims).\
                    gather(1, expanded_backpointer).\
                    reshape(batch_size * self.beam_size, *last_dims)

        if not torch.isfinite(last_log_probabilities).all():
            print("Infinite log probabilities encountered. "
                  "Some final sequences may not make sense. "
                  "This can happen when the beam size is "
                  "larger than the number of valid (non-zero "
                  "probability) transitions that the step function produces.")

        # Reconstruct the sequences.
        # shape: [(batch_size, beam_size, 1)]
        reconstructed_predictions = [predictions[-1].unsqueeze(2)]

        # shape: (batch_size, beam_size)
        cur_backpointers = backpointers[-1]

        for timestep in range(len(predictions) - 2, 0, -1):
            # shape: (batch_size, beam_size, 1)
            cur_preds = predictions[timestep].gather(
                1, cur_backpointers).unsqueeze(2)

            reconstructed_predictions.append(cur_preds)

            # shape: (batch_size, beam_size)
            cur_backpointers = backpointers[timestep -
                                            1].gather(1, cur_backpointers)

        # shape: (batch_size, beam_size, 1)
        final_preds = predictions[0].gather(1, cur_backpointers).unsqueeze(2)

        reconstructed_predictions.append(final_preds)

        # shape: (batch_size, beam_size, max_steps)
        all_predictions = torch.cat(
            list(reversed(reconstructed_predictions)), 2)

        return all_predictions, last_log_probabilities


# In[9]:


class LatexProducer(object):
    """
    Model wrapper, implementing batch greedy decoding and
    batch beam search decoding
    """

    def __init__(self, model, vocab, beam_size=5, max_len=64, use_cuda=True):
        """args:
            the path to model checkpoint
        """
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model = model.to(self.device)
        self._sign2id = vocab.sign2id
        self._id2sign = vocab.id2sign
        self.max_len = max_len
        self.beam_size = beam_size
        self._beam_search = BeamSearch(END_TOKEN, max_len, beam_size)

    def __call__(self, imgs):
        """args:
            imgs: images need to be decoded
            beam_size: if equal to 1, use greedy decoding
           returns:
            formulas list of batch_size length
        """
        if self.beam_size == 1:
            results = self._greedy_decoding(imgs)
        else:
            results = self._batch_beam_search(imgs)
        return results

    def _greedy_decoding(self, imgs):
        imgs = imgs.to(self.device)
        self.model.eval()

        enc_outs = self.model.encode(imgs)
        dec_states, O_t = self.model.init_decoder(enc_outs)

        batch_size = imgs.size(0)
        # storing decoding results
        formulas_idx = torch.ones(
            batch_size, self.max_len, device=self.device).long() * PAD_TOKEN
        # first decoding step's input
        tgt = torch.ones(
            batch_size, 1, device=self.device).long() * START_TOKEN
        with torch.no_grad():
            for t in range(self.max_len):
                dec_states, O_t, logit = self.model.step_decoding(
                    dec_states, O_t, enc_outs, tgt)

                tgt = torch.argmax(logit, dim=1, keepdim=True)
                formulas_idx[:, t:t + 1] = tgt
        results = self._idx2formulas(formulas_idx)
        return results

    def _simple_beam_search_decoding(self, imgs):
        """simpple beam search decoding (not support batch)"""
        self.model.eval()
        beam_results = [
            self._bs_decoding(img.unsqueeze(0))
            for img in imgs
        ]
        return beam_results

    def _idx2formulas(self, formulas_idx):
        """convert formula id matrix to formulas list"""
        results = []
        for id_ in formulas_idx:
            id_list = id_.tolist()
            result = []
            for sign_id in id_list:
                if sign_id != END_TOKEN:
                    result.append(self._id2sign[sign_id])
                else:
                    break
            results.append(" ".join(result))
        return results

    def _bs_decoding(self, img):
        """
        beam search decoding not support batch
        args:
            img: [1, C, H, W]
            beam_size: int
        return:
            formulas in str format
        """
        self.model.eval()
        img = img.to(self.device)

        # encoding
        # img = img.unsqueeze(0)  # [1, C, H, W]
        enc_outs = self.model.encode(img)  # [1, H*W, OUT_C]

        # prepare data for decoding
        enc_outs = enc_outs.expand(self.beam_size, -1, -1)
        # [Beam_size, dec_rnn_h]
        dec_states, O_t = self.model.init_decoder(enc_outs)

        # store top k ids (k is less or equal to beam_size)
        # in first decoding step, all they are  start token
        topk_ids = torch.ones(
            self.beam_size, device=self.device).long() * START_TOKEN
        topk_log_probs = torch.Tensor([0.0] + [-1e10] * (self.beam_size - 1))
        topk_log_probs = topk_log_probs.to(self.device)
        seqs = torch.ones(
            self.beam_size, 1, device=self.device).long() * START_TOKEN
        # store complete sequences and corrosponing scores
        complete_seqs = []
        complete_seqs_scores = []
        k = self.beam_size
        vocab_size = len(self._sign2id)
        with torch.no_grad():
            for t in range(self.max_len):
                dec_states, O_t, logit = self.model.step_decoding(
                    dec_states, O_t, enc_outs, topk_ids.unsqueeze(1))
                log_probs = torch.log(logit)  # [k, vocab_size]

                log_probs += topk_log_probs.unsqueeze(1)
                topk_log_probs, topk_ids = torch.topk(log_probs.view(-1), k)

                beam_index = topk_ids // vocab_size
                topk_ids = topk_ids % vocab_size

                seqs = torch.cat(
                    [seqs.index_select(0, beam_index), topk_ids.unsqueeze(1)],
                    dim=1
                )

                complete_inds = [
                    ind for ind, next_word in enumerate(topk_ids)
                    if next_word == END_TOKEN
                ]
                if t == (self.max_len-1):  # last_step, end all seqs
                    complete_inds = list(range(len(topk_ids)))

                incomplete_inds = list(
                    set(range(len(topk_ids))) - set(complete_inds)
                )
                if len(complete_inds) > 0:
                    complete_seqs.extend(seqs[complete_inds])
                    complete_seqs_scores.extend(topk_log_probs[complete_inds])
                k -= len(complete_inds)
                if k == 0:  # all beam finished
                    break

                # prepare for next step
                seqs = seqs[incomplete_inds]
                topk_ids = topk_ids[incomplete_inds]
                topk_log_probs = topk_log_probs[incomplete_inds]

                enc_outs = enc_outs[:k]
                seleted = beam_index[incomplete_inds]
                O_t = O_t[seleted]
                dec_states = (dec_states[0][seleted],
                              dec_states[1][seleted])

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i][1:]
        result = self._idx2formulas(seq.unsqueeze(0))[0]
        return result

    def _batch_beam_search(self, imgs):
        self.model.eval()
        imgs = imgs.to(self.device)
        enc_outs = self.model.encode(imgs)  # [batch_size, H*W, OUT_C]
        # enc_outs = enc_outs.expand(self.beam_size, -1, -1)
        dec_states, O_t = self.model.init_decoder(enc_outs)

        batch_size = imgs.size(0)
        start_predictions = torch.ones(
            batch_size, device=self.device).long() * START_TOKEN
        state = {}
        state['h_t'] = dec_states[0]
        state['c_t'] = dec_states[1]
        state['o_t'] = O_t
        state['enc_outs'] = enc_outs
        all_top_k_predictions, log_probabilities = self._beam_search.search(
            start_predictions, state, self._take_step)

        all_top_predictions = all_top_k_predictions[:, 0, :]
        all_top_predictions = self._idx2formulas(all_top_predictions)
        print(all_top_predictions, "\n")
        return all_top_predictions

    def _take_step(self, last_predictions, state):
        dec_states = (state['h_t'], state['c_t'])
        O_t = state['o_t']
        enc_outs = state['enc_outs']

        last_predictions = last_predictions.unsqueeze(1)
        with torch.no_grad():
            dec_states, O_t, logit = self.model.step_decoding(
                dec_states, O_t, enc_outs, last_predictions)

        # update state
        state['h_t'] = dec_states[0]
        state['c_t'] = dec_states[1]
        state['o_t'] = O_t
        return (torch.log(logit), state)


# In[10]:


import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli


def collate_fn(sign2id, batch):
    size = batch[0][0].size()
    batch = [img_formula for img_formula in batch
             if img_formula[0].size() == size]
    batch.sort(key=lambda img_formula: len(img_formula[1].split()),
               reverse=True)

    imgs, formulas = zip(*batch)
    formulas = [formula.split() for formula in formulas]
    tgt4training = formulas2tensor([['<s>']+formula for formula in formulas], sign2id)
    tgt4cal_loss = formulas2tensor([formula+['</s>'] for formula in formulas], sign2id)
    imgs = torch.stack(imgs, dim=0)
    return imgs, tgt4training, tgt4cal_loss


def formulas2tensor(formulas, sign2id):
    """convert formula to tensor"""

    batch_size = len(formulas)
    max_len = len(formulas[0])
    tensors = torch.ones(batch_size, max_len, dtype=torch.long) * PAD_TOKEN
    for i, formula in enumerate(formulas):
        for j, sign in enumerate(formula):
            tensors[i][j] = sign2id.get(sign, UNK_TOKEN)
    return tensors


def count_parameters(model):
    """count model parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# def tile(x, count, dim=0):
#     """
#     Tiles x on dimension dim count times.
#     """
#     perm = list(range(len(x.size())))
#     if dim != 0:
#         perm[0], perm[dim] = perm[dim], perm[0]
#         x = x.permute(perm).contiguous()
#     out_size = list(x.size())
#     out_size[0] *= count
#     batch = x.size(0)
#     x = x.view(batch, -1).transpose(0, 1).repeat(count, 1).transpose(0, 1)          .contiguous()          .view(*out_size)
#     if dim != 0:
#         x = x.permute(perm).contiguous()
#     return x


def load_formulas(filename):
    formulas = dict()
    with open(filename) as f:
        for idx, line in enumerate(f):
            formulas[idx] = line.strip()
    print("Loaded {} formulas from {}".format(len(formulas), filename))
    return formulas


def cal_loss(logits, targets):
    """args:
        logits: probability distribution return by model
                [B, MAX_LEN, voc_size]
        targets: target formulas
                [B, MAX_LEN]
    """
    padding = torch.ones_like(targets) * PAD_TOKEN
    mask = (targets != padding)

    targets = targets.masked_select(mask)
    logits = logits.masked_select(
        mask.unsqueeze(2).expand(-1, -1, logits.size(2))
    ).contiguous().view(-1, logits.size(2))
    logits = torch.log(logits)

    assert logits.size(0) == targets.size(0)

    loss = F.nll_loss(logits, targets)
    return loss


def get_checkpoint(ckpt_dir):
    """return full path if there is ckpt in ckpt_dir else None"""
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError("No checkpoint found in {}".format(ckpt_dir))

    ckpts = [f for f in os.listdir(ckpt_dir) if f.startswith('ckpt')]
    if not ckpts:
        raise FileNotFoundError("No checkpoint found in {}".format(ckpt_dir))

    last_ckpt, max_epoch = None, 0
    for ckpt in ckpts:
        epoch = int(ckpt.split('-')[1])
        if epoch > max_epoch:
            max_epoch = epoch
            last_ckpt = ckpt
    full_path = os.path.join(ckpt_dir, last_ckpt)
    print("Get checkpoint from {} for training".format(full_path))
    return full_path


def schedule_sample(prev_logit, prev_tgt, epsilon):
    prev_out = torch.argmax(prev_logit, dim=1, keepdim=True)
    prev_choices = torch.cat([prev_out, prev_tgt], dim=1)  # [B, 2]
    batch_size = prev_choices.size(0)
    prob = Bernoulli(torch.tensor([epsilon]*batch_size).unsqueeze(1))
    # sampling
    sample = prob.sample().long().to(prev_tgt.device)
    next_inp = torch.gather(prev_choices, 1, sample)
    return next_inp


def cal_epsilon(k, step, method):
    """
    Reference:
        Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks
        See details in https://arxiv.org/pdf/1506.03099.pdf
    """
    assert method in ['inv_sigmoid', 'exp', 'teacher_forcing']

    if method == 'exp':
        return k**step
    elif method == 'inv_sigmoid':
        return k/(k+math.exp(step/k))
    else:
        return 1.


# In[11]:


from os.path import join
from torch.nn.utils import clip_grad_norm_

class Trainer(object):
    def __init__(self, optimizer, model, lr_scheduler,
                 train_loader, val_loader, args,
                 use_cuda=True, init_epoch=1, last_epoch=15):

        self.optimizer = optimizer
        self.model = model
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args

        self.step = 0
        self.epoch = init_epoch
        self.total_step = (init_epoch-1)*len(train_loader)
        self.last_epoch = last_epoch
        self.best_val_loss = 1e18
        self.device = torch.device("cuda" if use_cuda else "cpu")

    def train(self):
        mes = "Epoch {}, step:{}/{} {:.2f}%, Loss:{:.4f}, Perplexity:{:.4f}"

        while self.epoch <= self.last_epoch:
            self.model.train()
            losses = 0.0
            for imgs, tgt4training, tgt4cal_loss in self.train_loader:
                step_loss = self.train_step(imgs, tgt4training, tgt4cal_loss)
                losses += step_loss

                # log message
                if self.step % self.args.print_freq == 0:
                    avg_loss = losses / self.args.print_freq
                    print(mes.format(
                        self.epoch, self.step, len(self.train_loader),
                        100 * self.step / len(self.train_loader),
                        avg_loss,
                        2**avg_loss
                    ))
                    losses = 0.0

            # one epoch Finished, calcute val loss
            val_loss = self.validate()
            self.lr_scheduler.step(val_loss)

            self.save_model('ckpt-{}-{:.4f}'.format(self.epoch, val_loss))
            self.epoch += 1
            self.step = 0

    def train_step(self, imgs, tgt4training, tgt4cal_loss):
        self.optimizer.zero_grad()

        imgs = imgs.to(self.device)
        tgt4training = tgt4training.to(self.device)
        tgt4cal_loss = tgt4cal_loss.to(self.device)
        epsilon = cal_epsilon(
            self.args.decay_k, self.total_step, self.args.sample_method)
        logits = self.model(imgs, tgt4training, epsilon)

        # calculate loss
        loss = cal_loss(logits, tgt4cal_loss)
        self.step += 1
        self.total_step += 1
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.args.clip)
        self.optimizer.step()

        return loss.item()

    def validate(self):
        self.model.eval()
        val_total_loss = 0.0
        mes = "Epoch {}, validation average loss:{:.4f}, Perplexity:{:.4f}"
        with torch.no_grad():
            for imgs, tgt4training, tgt4cal_loss in self.val_loader:
                imgs = imgs.to(self.device)
                tgt4training = tgt4training.to(self.device)
                tgt4cal_loss = tgt4cal_loss.to(self.device)

                epsilon = cal_epsilon(
                    self.args.decay_k, self.total_step, self.args.sample_method)
                logits = self.model(imgs, tgt4training, epsilon)
                loss = cal_loss(logits, tgt4cal_loss)
                val_total_loss += loss
            avg_loss = val_total_loss / len(self.val_loader)
            print(mes.format(
                self.epoch, avg_loss, 2**avg_loss
            ))
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_model('best_ckpt')
        return avg_loss

    def save_model(self, model_name):
        if not os.path.isdir(self.args.save_dir):
            os.makedirs(self.args.save_dir)
        save_path = join(self.args.save_dir, model_name+'.pt')
        print("Saving checkpoint to {}".format(save_path))

        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_sche': self.lr_scheduler.state_dict(),
            'epoch': self.epoch,
            'args': self.args
        }, save_path)


# In[21]:


# get_ipython().system('pip install distance')
# get_ipython().system('pip install nltk')


# In[12]:


import numpy as np
import nltk
import distance


def score_files(path_ref, path_hyp):
    """Loads result from file and score it
    Args:
        path_ref: (string) formulas of reference
        path_hyp: (string) formulas of prediction.
    Returns:
        scores: (dict)
    """
    # load formulas
    formulas_ref = load_formulas(path_ref)
    formulas_hyp = load_formulas(path_hyp)

    assert len(formulas_ref) == len(formulas_hyp)

    # tokenize
    refs = [ref.split(' ') for _, ref in formulas_ref.items()]
    hyps = [hyp.split(' ') for _, hyp in formulas_hyp.items()]

    # score
    return {
        "BLEU-4": bleu_score(refs, hyps)*100,
        "EM": exact_match_score(refs, hyps)*100,
        "Edit": edit_distance(refs, hyps)*100
    }


def exact_match_score(references, hypotheses):
    """Computes exact match scores.
    Args:
        references: list of list of tokens (one ref)
        hypotheses: list of list of tokens (one hypothesis)
    Returns:
        exact_match: (float) 1 is perfect
    """
    exact_match = 0
    for ref, hypo in zip(references, hypotheses):
        if np.array_equal(ref, hypo):
            exact_match += 1

    return exact_match / float(max(len(hypotheses), 1))


def bleu_score(references, hypotheses):
    """Computes bleu score.
    Args:
        references: list of list (one hypothesis)
        hypotheses: list of list (one hypothesis)
    Returns:
        BLEU-4 score: (float)
    """
    references = [[ref] for ref in references]  # for corpus_bleu func
    BLEU_4 = nltk.translate.bleu_score.corpus_bleu(
        references, hypotheses,
        weights=(0.25, 0.25, 0.25, 0.25)
    )
    return BLEU_4


def edit_distance(references, hypotheses):
    """Computes Levenshtein distance between two sequences.
    Args:
        references: list of list of token (one hypothesis)
        hypotheses: list of list of token (one hypothesis)
    Returns:
        1 - levenshtein distance: (higher is better, 1 is perfect)
    """
    d_leven, len_tot = 0, 0
    for ref, hypo in zip(references, hypotheses):
        d_leven += distance.levenshtein(ref, hypo)
        len_tot += float(max(len(ref), len(hypo)))

    return 1. - d_leven / len_tot


# In[13]:


import torch.optim as optim
from torch.utils.data import DataLoader
from collections import namedtuple
from torch.optim.lr_scheduler import ReduceLROnPlateau
from functools import partial


def train(args):
    max_epoch = args.epoches
    from_check_point = args.from_check_point
    if from_check_point:
        checkpoint_path = get_checkpoint(args.save_dir)
        checkpoint = torch.load(checkpoint_path)
        args = checkpoint['args']
    print("Training args:", args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Building vocab
    print("Load vocab...")
    vocab = load_vocab()

    use_cuda = True if args.cuda and torch.cuda.is_available() else False
    device = torch.device("cuda" if use_cuda else "cpu")

#     # data loader
    print("Construct data loader...")
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        collate_fn=partial(collate_fn, vocab.sign2id),
        pin_memory=True if use_cuda else False,
        num_workers=4)
    val_loader = DataLoader(
        validate_set,
        batch_size=args.batch_size,
        collate_fn=partial(collate_fn, vocab.sign2id),
        pin_memory=True if use_cuda else False,
        num_workers=4)

    # construct model
    print("Construct model")
    vocab_size = len(vocab)
    model = Image2LatexModel(
        vocab_size, args.emb_dim, args.dec_rnn_h,
        add_pos_feat=args.add_position_features,
        dropout=args.dropout
    )
    model = model.to(device)
    print("Model Settings:")
    print(model)

    # construct optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    lr_scheduler = ReduceLROnPlateau(
        optimizer,
        "min",
        factor=args.lr_decay,
        patience=args.lr_patience,
        verbose=True,
        min_lr=args.min_lr)

    if from_check_point:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        lr_scheduler.load_state_dict(checkpoint['lr_sche'])
        # init trainer from checkpoint
        trainer = Trainer(optimizer, model, lr_scheduler,
                          train_loader, val_loader, args,
                          use_cuda=use_cuda,
                          init_epoch=epoch, last_epoch=max_epoch)
    else:
        trainer = Trainer(optimizer, model, lr_scheduler,
                          train_loader, val_loader, args,
                          use_cuda=use_cuda,
                          init_epoch=1, last_epoch=args.epoches)
    # begin training
    trainer.train()


# In[17]:


# TODO
Arguments = namedtuple("Arguments", ["emb_dim", "dec_rnn_h", "add_position_features","max_len",
                       "dropout", "cuda", "batch_size", "epoches", 
                       "lr", "min_lr", "sample_method", "decay_k",
                        "lr_decay", "lr_patience", "clip", "save_dir",
                       "print_freq", "seed", "from_check_point"]
            )

args = Arguments(80, 512, True, 150, 0.2, True, 32, 5, 3e-4,
                         3e-5, "teacher_forcing", 1., 0.5, 3, 2.0,
                         os.getcwd(), 100, 2020, True)


# In[ ]:


# train(args)


# In[2]:


# torch.cuda.is_available()


# In[ ]:

# from tqdm import tqdm

args_cuda = True
args_model_path = "best_ckpt.pt"
args_batch_size = 32
args_result_path = "result.txt"
args_ref_path = "ref.txt"
args_max_len = 64
args_beam_size = 5


checkpoint = torch.load(join(args_model_path))
model_args = checkpoint['args']

vocab = load_vocab()
use_cuda = True if args_cuda and torch.cuda.is_available() else False

data_loader = DataLoader(
    test_set,
    batch_size=args_batch_size,
    collate_fn=partial(collate_fn, vocab.sign2id),
    pin_memory=True if use_cuda else False,
    num_workers=4
)

model = Image2LatexModel(
    len(vocab), model_args.emb_dim, model_args.dec_rnn_h,
    add_pos_feat=model_args.add_position_features,
    dropout=model_args.dropout
)
model.load_state_dict(checkpoint['model_state_dict'])

result_file = open(args_result_path, 'w')
ref_file = open(args_ref_path, 'w')

latex_producer = LatexProducer(
    model, vocab, max_len=args_max_len,
    use_cuda=use_cuda, beam_size=args_beam_size)

for imgs, tgt4training, tgt4cal_loss in data_loader:
    # try:
    reference = latex_producer._idx2formulas(tgt4cal_loss)
    results = latex_producer(imgs)
        # print(results, reference)
    # except RuntimeError:
        # break

    result_file.write('\n'.join(results))
    ref_file.write('\n'.join(reference))

result_file.close()
ref_file.close()
score = score_files(args_result_path, args_ref_path)
print("beam search result:", score)

