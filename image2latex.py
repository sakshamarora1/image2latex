import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.distributions.uniform import Uniform

import numpy as np
import os
import PIL
from collections import Counter


images_dir = "formula_images_processed"
formula_list = "im2latex_formulas.norm.lst"
train_list = "im2latex_train_filter.lst"
validate_list = "im2latex_validate_filter.lst"
test_list = "im2latex_test_filter.lst"


START_TOKEN = 0
PAD_TOKEN = 1
END_TOKEN = 2
UNK_TOKEN = 3


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

vocab = build_vocab()


class Image2LatexDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, formula_list, train_list):
        self.images_dir = images_dir

        with open(formula_list, "r", encoding="utf-8", errors="ignore", newline="\n") as f1:
            self.formulas = [formula.replace("\n", "").replace("\t", " ") for formula in f1.readlines()]
        
        with open(train_list, "r", encoding="utf-8", errors="ignore", newline="\n") as f2:
            self.train_set = [t.replace("\n", "").split() for t in f2.readlines()]
    
    def __getitem__(self, idx):
        item = self.train_set[idx]
        filename = item[0]
        formula = self.formulas[int(item[1])]
#         render_type = item[2]
        image = PIL.Image.open(self.images_dir + "/" + filename)
        return torchvision.transforms.ToTensor()(image), formula

    def __len__(self):
        return len(self.train_set)


train_set = Image2LatexDataset(images_dir, formula_list, train_list)
validate_set = Image2LatexDataset(images_dir, formula_list, validate_list)
test_set = Image2LatexDataset(images_dir, formula_list, test_list)


INIT = 1e-2


class Encoder(nn.Module):
    def __init__(self, out_channels=512, add_pos_feat=True):
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

        self.add_pos_feat = add_pos_feat
    
    def forward(self, images):
        encoded_images = self.cnn(images)
        encoded_images = encoded_images.permute(0, 2, 3, 1)  # [B, H', W', 512]
        B, H, W, C = encoded_imgs.shape
        encoded_images = encoded_images.contiguous().view(B, H*W, C)
        if self.add_pos_feat:
            encoded_images = self.add_positional_features(encoded_images)
        return encoded_images
    
    def add_positional_features(self, tensor: torch.Tensor, min_timescale: float = 1.0, max_timescale: float = 1.0e4):
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

        timestep_range = torch.arange(0, timesteps, dtype=torch.long, device= tensor.device).data.float()
        # We're generating both cos and sin frequencies,
        # so half for each.
        num_timescales = hidden_dim // 2
        timescale_range = torch.arange(0, num_timescales, dtype=torch.long, device= tensor.device).data.float()

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


class Decoder(nn.Module):
    def __init__(self, encoder_outdim, decoder_rnn_hidden, embed_size, output_size):
        super(Decoder, self).__init__()

        self.rnn_decoder = nn.LSTMCell(decoder_rnn_hidden+embed_size, decoder_rnn_hidden)
        self.embedding = nn.Embedding(output_size, embed_size)

        self.init_wh = nn.Linear(encoder_outdim, decoder_rnn_hidden)
        self.init_wc = nn.Linear(encoder_outdim, decoder_rnn_hidden)
        self.init_wo = nn.Linear(encoder_outdim, decoder_rnn_hidden)

    def forward(self, encoder_output):
        mean_encoder_output = encoder_output.mean(dim=1)
        h = nn.Tanh()(self.init_wh(mean_encoder_output))
        c = nn.Tanh()(self.init_wc(mean_encoder_output))
        o = nn.Tanh()(self.init_wo(mean_encoder_output))


class Image2LatexModel(nn.Module):
    def __init__(self):
        super(Image2LatexModel, self).__init__()



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

def cal_loss(pred, targets):
    """args:
        pred: probability distribution return by model
                [B, MAX_LEN, voc_size]
        targets: target formulas
                [B, MAX_LEN]
    """
    padding = torch.ones_like(targets) * PAD_TOKEN
    mask = (targets != padding)

    targets = targets.masked_select(mask)
    pred = pred.masked_select(
        mask.unsqueeze(2).expand(-1, -1, pred.size(2))
    ).contiguous().view(-1, pred.size(2))
    pred = torch.log(pred)

    assert pred.size(0) == targets.size(0)

    loss = F.nll_loss(pred, targets)
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


class Model:
    def __init__(self, optimizer, model, lr_scheduler,
                 train_loader, val_loader, args,
                 use_cuda=True, max_epoch=25):

        self.optimizer = optimizer
        self.model = model
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args

        self.step = 0
        self.epoch = 0
        self.total_step = 0
        self.last_epoch = max_epoch
        self.best_val_loss = None
        self.device = torch.device("cuda" if use_cuda else "cpu")

    def train(self):
        message = "Epoch {}, step:{}/{} {:.2f}%, Loss:{:.4f}, Perplexity:{:.4f}"

        while self.epoch <= self.last_epoch:
            self.model.train()
            losses = 0.0
            for imgs, tgt4training, tgt4cal_loss in self.train_loader:
                self.optimizer.zero_grad()

                imgs = imgs.to(self.device)
                tgt4training = tgt4training.to(self.device)
                tgt4cal_loss = tgt4cal_loss.to(self.device)
                epsilon = cal_epsilon(
                    self.args.decay_k, self.total_step, self.args.sample_method)
                pred = self.model(imgs, tgt4training, epsilon)

                # calculate loss
                loss = cal_loss(pred, tgt4cal_loss)
                self.step += 1
                self.total_step += 1
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.args.clip)
                self.optimizer.step()

                step_loss = loss.item()
                losses += step_loss

                # log message
                if self.step % self.args.print_freq == 0:
                    avg_loss = losses / self.args.print_freq
                    print(message.format(
                        self.epoch, self.step, len(self.train_loader),
                        100 * self.step / len(self.train_loader),
                        avg_loss,
                        2**avg_loss
                    ))
                    losses = 0.0

            val_loss = self.validate()
            self.lr_scheduler.step(val_loss)

            self.save_model('ckpt-{}-{:.4f}'.format(self.epoch, val_loss))
            self.epoch += 1
            self.step = 0

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
                pred = self.model(imgs, tgt4training, epsilon)
                loss = cal_loss(pred, tgt4cal_loss)
                val_total_loss += loss
            avg_loss = val_total_loss / len(self.val_loader)
            print(mes.format(
                self.epoch, avg_loss, 2**avg_loss
            ))
        if self.best_val_loss is None or avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_model('best_ckpt')
        return avg_loss
    
    def predict(self):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for imgs, tgt4training, tgt4cal_loss in self.val_loader:
                imgs = imgs.to(self.device)
                tgt4training = tgt4training.to(self.device)
                tgt4cal_loss = tgt4cal_loss.to(self.device)

                epsilon = cal_epsilon(
                    self.args.decay_k, self.total_step, self.args.sample_method)
                pred = self.model(imgs, tgt4training, epsilon)
                # TODO
                predictions.append(pred)
            # return        

    def save_model(self, model_name):
        if not os.path.isdir(self.args.save_dir):
            os.makedirs(self.args.save_dir)
        save_path = join(self.args.save_dir, model_name+'.pt')
        print("Saving checkpoint to {}".format(save_path))

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_sche': self.lr_scheduler.state_dict(),
            'args': self.args
        }, save_path)


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

    use_cuda = True if args.cuda and torch.cuda.is_available() else False
    device = torch.device("cuda" if use_cuda else "cpu")

    # data loader
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
        # init model from checkpoint
        model = Model(optimizer, model, lr_scheduler,
                          train_loader, val_loader, args,
                          use_cuda=use_cuda,
                          init_epoch=epoch, last_epoch=max_epoch)
    else:
        model = Model(optimizer, model, lr_scheduler,
                          train_loader, val_loader, args,
                          use_cuda=use_cuda,
                          init_epoch=1, last_epoch=args.epoches)

    model.train()


args_cuda = True
args_model_path = "best_ckpt.pt"
args_batch_size = 32
args_result_path = "result.txt"

def test():
    checkpoint = torch.load(join(args_model_path))
    model_args = checkpoint['args']
    use_cuda = True if args_cuda and torch.cuda.is_available() else False

    data_loader = DataLoader(
        test_set,
        batch_size=args_batch_size,
        collate_fn=partial(collate_fn, vocab.sign2id),
        pin_memory=True if use_cuda else False,
        num_workers=4
    )

    model = Model(optimizer, model, lr_scheduler,
        train_loader, val_loader, args,
        use_cuda=use_cuda,
        init_epoch=epoch, last_epoch=max_epoch
    )

    predictions = model.predict()
    # TODO
    # Get the index of corresponding formulas and match them simply if they are same of not
    # Write the predicted formula's index to result file along with image name