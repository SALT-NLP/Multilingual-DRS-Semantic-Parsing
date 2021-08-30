from __future__ import unicode_literals, print_function, division
from io import open
import types
import argparse
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from utils import readfile
from utils import data2instance
from utils import readpretrain
from tag import Tag

from joint_mask import OuterMask
from joint_mask import RelationMask
from joint_mask import VariableMask

trn_file = "../PMB2/gold/train_input.txt"
dev_file = "../PMB2/gold/dev_input.txt"
tst_file = "../PMB2/gold/test_input.txt"
trn_gold_file="train.gold"
dev_gold_file="output_dev/dev.gold"
tst_gold_file="output_tst/tst.gold"
pretrain_file = "../PMB2/gold/sskip.100.vectors"
tag_info_file = "../PMB2/gold/tag.txt"

dev_out_dir = "output_dev/"
tst_out_dir = "output_tst/"
model_dir = "output_model/"
log_dir="output_log"

parser = argparse.ArgumentParser(description='Semantic Parsing')
parser.add_argument("-s", "--savemodel", help="save model during evaluation", action="store_true", default=False)
parser.add_argument("-t", "--test", help="only test model", action="store_true", default=False)
parser.add_argument("-r", "--reload", help="reload model", action="store_true", default=False)
parser.add_argument("-mp", "--minusPos", help="reload model", action="store_true", default=False)
parser.add_argument("-md", "--minusDependency", help="reload model", action="store_true", default=False)
parser.add_argument("-mw", "--minusWordEmbedding", help="reload model", action="store_true", default=False)
parser.add_argument('-model', type=str, default='output_model/1.model')
args = parser.parse_args()

WORD_EMBEDDING_DIM = 64
PRETRAIN_EMBEDDING_DIM = 100
LEMMA_EMBEDDING_DIM = 32
POS_TAG_DIM=32
DEP_TAG_DIM=32
TAG_DIM = 128
INPUT_DIM = 100
ENCODER_HIDDEN_DIM = 256
DECODER_INPUT_DIM = 128
ATTENTION_HIDDEN_DIM = 256
PRINT_EVERY=1000
EVALUATE_EVERY_EPOCH=1
LEARNING_RATE=0.0005
ENCODER_LAYER=2
DECODER_LAYER=1
ENCODER_DROUPOUT_RATE=0.1
DECODER_DROPOUT_RATE=0.1

UNK = "<UNK>"

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, word_size, word_dim, pretrain_size, pretrain_dim, pretrain_embeddings, lemma_size, lemma_dim,
                 postag_size, postag_dim,deptag_size,deptag_dim,input_dim, hidden_dim, n_layers=1, dropout_p=0.0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.postag_dim=postag_dim
        self.deptag_dim=deptag_dim

        self.word_embeds = nn.Embedding(word_size, word_dim)
        self.pretrain_embeds = nn.Embedding(pretrain_size, pretrain_dim)
        self.pretrain_embeds.weight = nn.Parameter(pretrain_embeddings, False)
        self.lemma_embeds = nn.Embedding(lemma_size, lemma_dim)
        self.postag_embeds = nn.Embedding(postag_size, postag_dim)
        self.deptag_embeds=nn.Embedding(deptag_size, deptag_dim)
        self.dropout = nn.Dropout(self.dropout_p)

        if args.minusPos and args.minusDependency:
            self.embeds2input = nn.Linear(word_dim + pretrain_dim + lemma_dim, input_dim)
        elif args.minusPos and args.minusWordEmbedding:
            self.embeds2input = nn.Linear(deptag_dim, input_dim)
        elif args.minusDependency and args.minusWordEmbedding:
            self.embeds2input = nn.Linear(postag_dim, input_dim)
        elif args.minusWordEmbedding:
            self.embeds2input = nn.Linear( postag_dim + deptag_dim, input_dim)
        elif args.minusPos:
            self.embeds2input = nn.Linear(word_dim + pretrain_dim + lemma_dim + deptag_dim, input_dim)
        elif args.minusDependency:
            self.embeds2input = nn.Linear(word_dim + pretrain_dim + lemma_dim + postag_dim, input_dim)
        else:
            self.embeds2input = nn.Linear(word_dim + pretrain_dim + lemma_dim + postag_dim + deptag_dim, input_dim)

        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=self.n_layers, bidirectional=True)

    def forward(self, sentence, hidden, train=True):
        word_embedded = self.word_embeds(sentence[0])
        pretrain_embedded = self.pretrain_embeds(sentence[1])
        lemma_embedded = self.lemma_embeds(sentence[2])
        postag_embedded = self.postag_embeds(sentence[3])
        deptag_embedded =self.deptag_embeds(sentence[4])
        if train:
            word_embedded = self.dropout(word_embedded)
            lemma_embedded = self.dropout(lemma_embedded)
            self.lstm.dropout = self.dropout_p
            postag_embedded = self.dropout(postag_embedded)
            deptag_embedded=self.dropout(deptag_embedded)
        if args.minusPos and args.minusDependency:
            embeds = self.tanh(self.embeds2input(
                torch.cat((word_embedded, pretrain_embedded, lemma_embedded),
                          1))).view(len(sentence[0]), 1, -1)
        elif args.minusPos and args.minusWordEmbedding:
            embeds = self.tanh(self.embeds2input(deptag_embedded)).view(len(sentence[0]), 1, -1)
        elif args.minusDependency and args.minusWordEmbedding:
            embeds = self.tanh(self.embeds2input(postag_embedded)).view(len(sentence[0]), 1, -1)
        elif args.minusWordEmbedding:
            embeds = self.tanh(self.embeds2input(
                torch.cat((postag_embedded, deptag_embedded),1))).view(len(sentence[0]), 1, -1)
        elif args.minusPos:
            embeds = self.tanh(self.embeds2input(
                torch.cat((word_embedded, pretrain_embedded, lemma_embedded, deptag_embedded),
                          1))).view(len(sentence[0]), 1, -1)

        elif args.minusDependency:
            embeds = self.tanh(self.embeds2input(
                torch.cat((word_embedded, pretrain_embedded, lemma_embedded, postag_embedded),
                          1))).view(len(sentence[0]), 1, -1)

        else:
            embeds = self.tanh(self.embeds2input(torch.cat((word_embedded, pretrain_embedded, lemma_embedded,postag_embedded,deptag_embedded), 1))).view(len(sentence[0]), 1, -1)

        output, hidden = self.lstm(embeds, hidden)
        return output, hidden

    def initHidden(self):
        result = (torch.zeros(2 * self.n_layers, 1, self.hidden_dim,dtype=torch.float, device=device),
                  torch.zeros(2 * self.n_layers, 1, self.hidden_dim,dtype=torch.float, device=device))
        return result



class AttnDecoderRNN(nn.Module):
    def __init__(self, outer_mask_pool, rel_mask_pool, var_mask_pool, tags_info, tag_dim, input_dim, feat_dim,
                 encoder_hidden_dim, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.outer_mask_pool = outer_mask_pool
        self.rel_mask_pool = rel_mask_pool
        self.var_mask_pool = var_mask_pool
        self.total_rel = 0

        self.tags_info = tags_info
        self.tag_size = tags_info.tag_size
        self.all_tag_size = tags_info.all_tag_size

        self.tag_dim = tag_dim
        self.input_dim = input_dim
        self.feat_dim = feat_dim
        self.hidden_dim = encoder_hidden_dim * 2

        self.n_layers = n_layers
        self.dropout_p = dropout_p

        self.dropout = nn.Dropout(self.dropout_p)
        self.tag_embeds = nn.Embedding(self.tags_info.tag_size, self.tag_dim)

        self.struct2input = nn.Linear(self.hidden_dim, self.tag_dim)

        self.struct2rel = nn.Linear(self.hidden_dim, self.tag_dim)
        self.rel2var = nn.Linear(self.hidden_dim, self.tag_dim)

        self.lstm = nn.LSTM(self.tag_dim, self.hidden_dim, num_layers=self.n_layers)

        self.feat = nn.Linear(self.hidden_dim + self.tag_dim, self.feat_dim)
        self.feat_tanh = nn.Tanh()

        self.out = nn.Linear(self.feat_dim, self.tag_size)

        self.selective_matrix = torch.randn(1, self.hidden_dim, self.hidden_dim).to(device)

    def forward(self, sentence_variable, inputs, hidden, encoder_output, least, train, mask_variable, opt):

        if opt == 1:
            return self.forward_1(inputs, hidden, encoder_output, train, mask_variable)
        elif opt == 2:
            return self.forward_2(sentence_variable, inputs, hidden, encoder_output, least, train, mask_variable)
        elif opt == 3:
            return self.forward_3(inputs, hidden, encoder_output, train, mask_variable)
        else:
            assert False, "unrecognized option"

    def forward_1(self, input, hidden, encoder_output, train, mask_variable):
        if train:
            self.lstm.dropout = self.dropout_p
            embedded = self.tag_embeds(input).unsqueeze(1)
            embedded = self.dropout(embedded)

            output, hidden = self.lstm(embedded, hidden)

            attn_weights = F.softmax(
                torch.bmm(output.transpose(0, 1), encoder_output.transpose(0, 1).transpose(1, 2)).view(output.size(0),
                                                                                                       -1), 1)
            attn_hiddens = torch.bmm(attn_weights.unsqueeze(0), encoder_output.transpose(0, 1))
            feat_hiddens = self.feat_tanh(
                self.feat(torch.cat((attn_hiddens, embedded.transpose(0, 1)), 2).view(output.size(0), -1)))

            global_score = self.out(feat_hiddens)

            log_softmax_output = F.log_softmax(global_score + (mask_variable - 1) * 1e10, 1)

            return log_softmax_output, output
        else:

            self.lstm.dropout = 0.0
            tokens = []
            self.outer_mask_pool.reset()
            hidden_rep = []
            while True:
                mask = self.outer_mask_pool.get_step_mask()
                mask_variable = torch.tensor(mask, requires_grad=False,dtype=torch.float, device=device).unsqueeze(0)

                embedded = self.tag_embeds(input).view(1, 1, -1)
                output, hidden = self.lstm(embedded, hidden)
                hidden_rep.append(output)

                attn_weights = F.softmax(
                    torch.bmm(output, encoder_output.transpose(0, 1).transpose(1, 2)).view(output.size(0), -1), 1)
                attn_hiddens = torch.bmm(attn_weights.unsqueeze(0), encoder_output.transpose(0, 1))
                feat_hiddens = self.feat_tanh(
                    self.feat(torch.cat((attn_hiddens, embedded), 2).view(embedded.size(0), -1)))

                global_score = self.out(feat_hiddens)

                score = global_score + (mask_variable - 1) * 1e10

                _, input = torch.max(score, 1)
                idx = input.view(-1).detach().tolist()[0]

                tokens.append(idx)
                self.outer_mask_pool.update(-2, idx)

                if idx == self.tags_info.tag_to_ix[self.tags_info.EOS]:
                    break
            return torch.tensor(tokens,dtype=torch.long, device=device), torch.cat(hidden_rep, 0), hidden

    def forward_2(self, sentence_variable, inputs, hidden, encoder_output, least, train, mask_variable):
        if train:
            self.lstm.dropout = self.dropout_p
            List = []
            for condition, input in inputs:
                List.append(self.struct2rel(condition).view(1, 1, -1))
                if type(input) == types.NoneType:
                    pass
                else:
                    for item in input:
                        if item<self.tag_size:
                            List.append(self.tag_embeds(torch.tensor([item], dtype=torch.long, device=device)).unsqueeze(1))
                        else:
                            inputList=self.struct2input(encoder_output[item-self.tag_size])
                            List.append(inputList.unsqueeze(1))
            embedded = torch.cat(List, 0)
            embedded = self.dropout(embedded)

            output, hidden = self.lstm(embedded, hidden)

            selective_score = torch.bmm(torch.bmm(output.transpose(0, 1), self.selective_matrix),
                                        encoder_output.transpose(0, 1).transpose(1, 2)).view(output.size(0), -1)

            attn_weights = F.softmax(
                torch.bmm(output.transpose(0, 1), encoder_output.transpose(0, 1).transpose(1, 2)).view(output.size(0),
                                                                                                       -1), 1)
            attn_hiddens = torch.bmm(attn_weights.unsqueeze(0), encoder_output.transpose(0, 1))
            feat_hiddens = self.feat_tanh(
                self.feat(torch.cat((attn_hiddens, embedded.transpose(0, 1)), 2).view(output.size(0), -1)))

            global_score = self.out(feat_hiddens)

            total_score = torch.cat((global_score, selective_score), 1)

            log_softmax_output = F.log_softmax(total_score + (mask_variable - 1) * 1e10, 1)

            return log_softmax_output, output

        else:
            self.lstm.dropout = 0.0
            tokens = []
            rel = 0
            hidden_reps = []
            self.rel_mask_pool.reset(sentence_variable[0].size(0))
            embedded = self.struct2rel(inputs).view(1, 1, -1)
            idx=-1
            idchoice=-1
            while True:
                if least:
                    mask = self.rel_mask_pool.get_step_mask(least,idchoice)
                    least=False
                else:
                    mask = self.rel_mask_pool.get_step_mask(least, idchoice)

                mask_variable = torch.tensor(mask, requires_grad=False, dtype=torch.float, device=device).unsqueeze(0)

                output, hidden = self.lstm(embedded, hidden)
                hidden_reps.append(output)

                selective_score = torch.bmm(torch.bmm(output, self.selective_matrix),
                                            encoder_output.transpose(0, 1).transpose(1, 2)).view(output.size(0), -1)
                attn_weights = F.softmax(
                    torch.bmm(output, encoder_output.transpose(0, 1).transpose(1, 2)).view(output.size(0), -1), 1)
                attn_hiddens = torch.bmm(attn_weights.unsqueeze(0), encoder_output.transpose(0, 1))
                feat_hiddens = self.feat_tanh(
                    self.feat(torch.cat((attn_hiddens, embedded), 2).view(embedded.size(0), -1)))

                global_score = self.out(feat_hiddens)

                total_score = torch.cat((global_score, selective_score), 1)


                output = total_score + (mask_variable - 1) * 1e10

                _, input = torch.max(output, 1)
                idchoice = input.view(-1).detach().tolist()[0]
                if idx==17:
                    assert(idchoice>=self.tags_info.tag_size)

                if idchoice >= self.tags_info.tag_size:
                    tokens.append(idchoice)
                    idx = idchoice
                else:
                    idx=idchoice
                    tokens.append(idx)
                if len(tokens)>1 and tokens[-2]==17:
                    assert(tokens[-1]>=self.tags_info.tag_size)

                if idx == self.tags_info.tag_to_ix[self.tags_info.EOS]:
                    break
                elif rel > 61 or self.total_rel > 121:
                    break
                rel += 1
                self.total_rel += 1
                if idx<self.tags_info.tag_size:
                    embedded = self.tag_embeds(input).view(1, 1, -1)
                else:
                    inputList = self.struct2input(encoder_output[idx - self.tag_size])
                    embedded=inputList.unsqueeze(1)
            assert(len(tokens)==len(hidden_reps))

            return torch.tensor(tokens,dtype=torch.long, device=device), torch.cat(hidden_reps, 0), hidden

    def forward_3(self, inputs, hidden, encoder_output, train, mask_variable):
        if train:
            self.lstm.dropout = self.dropout_p

            List = []
            for condition, input in inputs:
                List.append(self.rel2var(condition).view(1, 1, -1))
                List.append(self.tag_embeds(input).unsqueeze(1))
            embedded = torch.cat(List, 0)
            embedded = self.dropout(embedded)

            output, hidden = self.lstm(embedded, hidden)

            attn_weights = F.softmax(
                torch.bmm(output.transpose(0, 1), encoder_output.transpose(0, 1).transpose(1, 2)).view(output.size(0),
                                                                                                       -1), 1)
            attn_hiddens = torch.bmm(attn_weights.unsqueeze(0), encoder_output.transpose(0, 1))
            feat_hiddens = self.feat_tanh(
                self.feat(torch.cat((attn_hiddens, embedded.transpose(0, 1)), 2).view(output.size(0), -1)))

            global_score = self.out(feat_hiddens)

            score = global_score

            log_softmax_output = F.log_softmax(score + (mask_variable - 1) * 1e10, 1)

            return log_softmax_output, output
        else:
            self.lstm.dropout = 0.0
            tokens = []
            embedded = self.rel2var(inputs).view(1, 1, -1)
            while True:
                output, hidden = self.lstm(embedded, hidden)

                attn_weights = F.softmax(
                    torch.bmm(output, encoder_output.transpose(0, 1).transpose(1, 2)).view(output.size(0), -1), 1)
                attn_hiddens = torch.bmm(attn_weights.unsqueeze(0), encoder_output.transpose(0, 1))
                feat_hiddens = self.feat_tanh(
                    self.feat(torch.cat((attn_hiddens, embedded), 2).view(embedded.size(0), -1)))

                global_score = self.out(feat_hiddens)

                mask = self.var_mask_pool.get_step_mask()
                mask_variable = torch.tensor(mask,dtype=torch.float, device=device)

                score = global_score + (mask_variable - 1) * 1e10

                _, input = torch.max(score, 1)
                embedded = self.tag_embeds(input).view(1, 1, -1)

                idx = input.view(-1).detach().tolist()[0]
                assert idx < self.tags_info.tag_size
                if idx == self.tags_info.tag_to_ix[self.tags_info.EOS]:
                    break

                tokens.append(idx)
                self.var_mask_pool.update(idx)

            return torch.tensor(tokens,dtype=torch.long, device=device),  hidden


def train(sentence_variable, input_variables, gold_variables, mask_variables, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, back_prop=True):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    target_length1 = 0
    target_length2 = 0
    target_length3 = 0

    encoder_hidden = encoder.initHidden()
    encoder_output, encoder_hidden = encoder(sentence_variable, encoder_hidden)

    ################structure
    decoder_hidden1 = (torch.cat((encoder_hidden[0][-2], encoder_hidden[0][-1]), 1).unsqueeze(0),
                       torch.cat((encoder_hidden[1][-2], encoder_hidden[1][-1]), 1).unsqueeze(0))
    decoder_input1 = input_variables[0]
    decoder_output1, hidden_rep1 = decoder(None, decoder_input1, decoder_hidden1, encoder_output, least=None,
                                           train=True, mask_variable=mask_variables[0], opt=1)
    gold_variable1 = gold_variables[0]
    loss1 = criterion(decoder_output1, gold_variable1)
    target_length1 += gold_variable1.size(0)

    ################ relation
    decoder_hidden2 = (torch.cat((encoder_hidden[0][-2], encoder_hidden[0][-1]), 1).unsqueeze(0),
                       torch.cat((encoder_hidden[1][-2], encoder_hidden[1][-1]), 1).unsqueeze(0))
    decoder_input2 = []
    structs = input_variables[0].view(-1).detach().tolist()  # SOS DRS( P1( DRS( P2( DRS( ) ) ) ) )
    p = 0
    for i in range(len(structs)):
        if structs[i] ==4:
            decoder_input2.append((hidden_rep1[i], input_variables[1][p]))
            p += 1
    assert p == len(input_variables[1])
    decoder_output2, hidden_rep2 = decoder(sentence_variable, decoder_input2, decoder_hidden2, encoder_output,
                                           least=None, train=True, mask_variable=mask_variables[1], opt=2)
    loss2 = criterion(decoder_output2, gold_variables[1])
    target_length2 += gold_variables[1].size(0)

    ################ variable
    decoder_hidden3 = (torch.cat((encoder_hidden[0][-2], encoder_hidden[0][-1]), 1).unsqueeze(0),
                       torch.cat((encoder_hidden[1][-2], encoder_hidden[1][-1]), 1).unsqueeze(0))

    decoder_input3 = []
    ##### decoder hidden is like
    #####   DRS( india( say( CAUSE( TOPIC(
    #####   DRS( THING( security( increase( PATIENT( ATTRIBUTE( FOR(
    #####   DRS( TOPIC( possible( TOPIC( militant( strike( country( in( thwart( AGENT( THEME(

    i = 0
    p = 0
    for j in range(len(input_variables[1])):
        i += 1
        if type(input_variables[1][j]) == types.NoneType:
            pass
        else:
            for k in range(len(input_variables[1][j])):
                if k<len(input_variables[1][j])-1 and input_variables[1][j][k+1]==17:
                    i+=1
                    continue
                if input_variables[1][j][k]==17:
                    i+=1
                    continue
                decoder_input3.append((hidden_rep2[i], input_variables[2][p]))
                i += 1
                p += 1
    assert p == len(input_variables[2])
    decoder_output3, hidden_rep3 = decoder(None, decoder_input3, decoder_hidden3, encoder_output, least=None,
                                           train=True, mask_variable=mask_variables[2], opt=3)
    loss3 = criterion(decoder_output3, gold_variables[2])
    target_length3 += gold_variables[2].size(0)

    loss = loss1 + loss2 + loss3
    if back_prop:
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

    return loss1.item() / target_length1, loss2.item() / target_length2, loss3.item() / target_length3


def decode(sentence_variable, encoder, decoder):
    encoder_hidden = encoder.initHidden()
    encoder_output, encoder_hidden = encoder(sentence_variable, encoder_hidden,train=False)

    ####### struct
    decoder_hidden1 = (torch.cat((encoder_hidden[0][-2], encoder_hidden[0][-1]), 1).unsqueeze(0),
                       torch.cat((encoder_hidden[1][-2], encoder_hidden[1][-1]), 1).unsqueeze(0))

    decoder_input1 = torch.tensor([0],dtype=torch.long, device=device)
    decoder_output1, hidden_rep1, decoder_hidden1 = decoder(None, decoder_input1, decoder_hidden1, encoder_output,
                                                            least=None, train=False, mask_variable=None, opt=1)
    structs = decoder_output1.view(-1).detach().tolist()

    ####### relation
    decoder_hidden2 = (torch.cat((encoder_hidden[0][-2], encoder_hidden[0][-1]), 1).unsqueeze(0),
                       torch.cat((encoder_hidden[1][-2], encoder_hidden[1][-1]), 1).unsqueeze(0))
    decoder.rel_mask_pool.reset(sentence_variable[0].size(0))
    decoder.total_rel = 0
    relations = []
    hidden_rep2_list = []
    for i in range(len(structs)):
        if structs[i] == 4:  # prev output, and hidden_rep1[i+1] is the input representation of prev output.
            least = False
            if (structs[i] == 4 and structs[i + 1] == 2):
                least = True
            decoder.rel_mask_pool.set_sdrs(structs[i] == 3)
            decoder_output2, hidden_rep2, decoder_hidden2 = decoder(sentence_variable, hidden_rep1[i + 1],
                                                                    decoder_hidden2, encoder_output, least=least,
                                                                    train=False, mask_variable=None, opt=2)#???
            relations.append(decoder_output2.view(-1).data.tolist())
            hidden_rep2_list.append(hidden_rep2)
            decoder.rel_mask_pool.reset(sentence_variable[0].size(0))
    ####### variable
    decoder_hidden3 = (torch.cat((encoder_hidden[0][-2], encoder_hidden[0][-1]), 1).unsqueeze(0),
                       torch.cat((encoder_hidden[1][-2], encoder_hidden[1][-1]), 1).unsqueeze(0))
    # p_max
    p_max = 0
    for tok in structs:
        if tok >= decoder.tags_info.p_rel_start and tok < decoder.tags_info.p_tag_start:
            p_max += 1

    decoder.var_mask_pool.reset(p_max)
    structs_p = 0
    struct_rel_tokens = []
    var_tokens = []
    for i in range(len(structs)):
        if structs[i] == 1:  # EOS
            continue
        assert not (structs[i]==17)
        decoder.var_mask_pool.update(structs[i])
        struct_rel_tokens.append(structs[i])
        if structs[i] == 4:
            for j in range(len(relations[structs_p])):
                if relations[structs_p][j] == 1 or j==len(relations[structs_p])-1:  # EOS
                    continue
                if j==len(relations[structs_p])-2 and relations[structs_p][j]==17:
                    continue
                struct_rel_tokens.append(relations[structs_p][j])
                decoder.var_mask_pool.update(relations[structs_p][j])####????
                if j<len(relations[structs_p])-3 and relations[structs_p][j+1]==17:
                    continue
                if relations[structs_p][j]==17:
                    continue
                if j+1>=len(hidden_rep2_list[structs_p]):
                    print(relations[structs_p])
                decoder_output3, decoder_hidden3 = decoder(None, hidden_rep2_list[structs_p][j + 1], decoder_hidden3,
                                                           encoder_output, least=None, train=False, mask_variable=None,
                                                           opt=3)
                var_tokens.append(decoder_output3.view(-1).data.tolist())
                decoder.var_mask_pool.update(2)
                struct_rel_tokens.append(2)
            structs_p += 1
    assert structs_p == len(relations)

    return struct_rel_tokens, var_tokens


######################################################################
# The whole training process looks like this:
#
# -  Start a timer
# -  Initialize optimizers and criterion
# -  Create set of training pairs
# -  Start empty losses array for plotting
#
# Then we call ``train`` many times and occasionally print the progress (%
# of examples, time so far, estimated time) and average loss.
#

def trainIters(trn_instances, dev_instances, tst_instances, encoder, decoder, print_every=100, evaluate_every=1000,
               learning_rate=0.001,log_dir=None):
    print_loss_total = 0.0  # Reset every print_every
    print_loss_total1 = 0.0
    print_loss_total2 = 0.0
    print_loss_total3 = 0.0

    criterion = nn.NLLLoss()

    check_point = {}
    if args.reload:
        check_point = torch.load(args.model)
        encoder.load_state_dict(check_point["encoder"])
        decoder.load_state_dict(check_point["decoder"])
        encoder = encoder.to(device)
        decoder = decoder.to(device)

    encoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=learning_rate,
                                   weight_decay=1e-4)
    decoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, decoder.parameters()), lr=learning_rate,
                                   weight_decay=1e-4)

    if args.reload:
        encoder_optimizer.load_state_dict(check_point["encoder_optimizer"])
        decoder_optimizer.load_state_dict(check_point["decoder_optimizer"])

        for state in encoder_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        for state in decoder_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    # =============================== training_data

    sentence_variables = []

    input1_variables = []
    input2_variables = []
    input3_variables = []

    gold1_variables = []
    gold2_variables = []
    gold3_variables = []

    mask1_variables = []
    mask2_variables = []
    mask3_variables = []

    for instance in trn_instances:
        sentence_variables.append([])
        sentence_variables[-1].append(instance[0].to(device))
        sentence_variables[-1].append(instance[1].to(device))
        sentence_variables[-1].append(instance[2].to(device))
        sentence_variables[-1].append(instance[7].to(device))
        sentence_variables[-1].append(instance[8].to(device))
        sentence_variables[-1].append(instance[9])

        input1_variables.append(torch.tensor([0] + instance[3],dtype=torch.long, device=device))
        gold1_variables.append(torch.tensor(instance[3] + [1],dtype=torch.long, device=device))

        p = 0
        all_relations = []
        input2_variable = []
        for i in range(len(instance[3])):
            idx = instance[3][i]
            if idx == 4:
                all_relations = all_relations + instance[4][p]
                all_relations.append([-2, 1])
                if len(instance[4][p]) == 0:
                    input2_variable.append(None)
                else:
                    input2_variable.append([idx if type==-2 else type + decoder.tags_info.tag_size for type, idx in instance[4][p]])
                p += 1
        input2_variables.append(input2_variable)

        sel_gen_relations = []
        for type, idx in all_relations:
            if type == -2:
                sel_gen_relations.append(idx)
            else:
                sel_gen_relations.append(type + decoder.tags_info.tag_size)
        gold2_variables.append(torch.tensor(sel_gen_relations,dtype=torch.long, device=device))
        assert p == len(instance[4])

        p = 0
        all_variables = []
        input3_variable = []
        assert(instance[5][-1]==2)
        for i in range(len(instance[5])-1):
            idx = instance[5][i]
            idy= instance[5][i+1]
            if (idx>=14 and idx<=16) or (idx >= decoder.tags_info.global_start and idx < decoder.tags_info.p_rel_start) or \
                    (idx >= decoder.tags_info.tag_size and not idy==17):
                all_variables = all_variables + instance[6][p]
                all_variables.append(1)
                input3_variable.append(torch.tensor(instance[6][p], dtype=torch.long, device=device))
                p += 1
        input3_variables.append(input3_variable)
        gold3_variables.append(torch.tensor(all_variables,dtype=torch.long, device=device))
        assert p == len(instance[6])

        ##### mask1
        p_max = 0
        decoder.outer_mask_pool.reset()
        mask1 = []
        mask1.append(decoder.outer_mask_pool.get_step_mask())
        for idx in instance[3]:
            assert mask1[-1][idx] == decoder.outer_mask_pool.need
            if idx >= decoder.tags_info.p_rel_start and idx < decoder.tags_info.p_tag_start:
                p_max += 1
            decoder.outer_mask_pool.update(-2, idx)
            mask1.append(decoder.outer_mask_pool.get_step_mask())

        ##### mask2
        decoder.rel_mask_pool.reset(len(instance[0]))
        mask2 = []
        p = 0
        for i in range(len(instance[3])):
            idx = instance[3][i]
            if idx == 4:
                least = False
                if (idx == 4 and instance[3][i + 1] == 2):
                    least = True
                decoder.rel_mask_pool.set_sdrs(idx == 3)
                decoder.rel_mask_pool.reset(len(instance[0]))
                temp_mask = []
                temp_mask.append(decoder.rel_mask_pool.get_step_mask(least,-1))
                if least==True:
                    least=False
                for k in range(len(instance[4][p])):
                    temp_idx = instance[4][p][k][0]
                    if temp_idx == -2:
                        temp_idx = instance[4][p][k][1]
                    else:
                        temp_idx += decoder.tags_info.tag_size
                    if not temp_idx==17:
                        assert(temp_mask[k][temp_idx]==1)
                    if not temp_mask[k][temp_idx] == decoder.rel_mask_pool.need:
                        print(instance[4][p][k][1])
                    assert temp_mask[k][temp_idx] == decoder.rel_mask_pool.need

                    temp_mask.append(decoder.rel_mask_pool.get_step_mask(least, temp_idx))
                mask2 = mask2 + temp_mask
                p += 1
        assert p == len(instance[4])

        #### mask3
        decoder.var_mask_pool.reset(p_max)
        mask3 = []
        p = 0
        for i in range(len(instance[5])-1):
            idx = instance[5][i]
            idy=instance[5][i+1]
            decoder.var_mask_pool.update(idx)
            if (idx>=14 and idx<=16) or (idx >= decoder.tags_info.global_start and idx < decoder.tags_info.p_rel_start) or \
                    (idx >= decoder.tags_info.tag_size and not idy==17):
                for idxx in instance[6][p]:
                    mask3.append(decoder.var_mask_pool.get_step_mask())
                    decoder.var_mask_pool.update(idxx)
                    assert mask3[-1][idxx] == decoder.var_mask_pool.need
                mask3.append(decoder.var_mask_pool.get_step_mask())
                p += 1
        assert p == len(instance[6])

        mask1_variables.append(torch.tensor(mask1, requires_grad=False, dtype=torch.float, device=device))
        mask2_variables.append(torch.tensor(mask2, requires_grad=False, dtype=torch.float, device=device))
        mask3_variables.append(torch.tensor(mask3, requires_grad=False, dtype=torch.float, device=device))


    # ==================================
    dev_sentence_variables = []

    dev_input1_variables = []
    dev_input2_variables = []
    dev_input3_variables = []

    dev_gold1_variables = []
    dev_gold2_variables = []
    dev_gold3_variables = []

    dev_mask1_variables = []
    dev_mask2_variables = []
    dev_mask3_variables = []

    for instance in dev_instances:
        dev_sentence_variables.append([])
        dev_sentence_variables[-1].append(instance[0].to(device))
        dev_sentence_variables[-1].append(instance[1].to(device))
        dev_sentence_variables[-1].append(instance[2].to(device))
        dev_sentence_variables[-1].append(instance[7].to(device))
        dev_sentence_variables[-1].append(instance[8].to(device))
        dev_sentence_variables[-1].append(instance[9])


        dev_input1_variables.append(torch.tensor([0] + instance[3], dtype=torch.long, device=device))
        dev_gold1_variables.append(torch.tensor(instance[3] + [1], dtype=torch.long, device=device))

        p = 0
        all_relations = []
        dev_input2_variable = []
        for i in range(len(instance[3])):
            idx = instance[3][i]
            if idx == 4:
                all_relations = all_relations + instance[4][p]
                all_relations.append([-2, 1])
                if len(instance[4][p]) == 0:
                    dev_input2_variable.append(None)
                else:
                    dev_input2_variable.append(
                        [idx if type == -2 else type + decoder.tags_info.tag_size for type, idx in instance[4][p]])
                p += 1
        dev_input2_variables.append(dev_input2_variable)
        sel_gen_relations = []
        for type, idx in all_relations:
            if type == -2:
                sel_gen_relations.append(idx)
            else:
                sel_gen_relations.append(type + decoder.tags_info.tag_size)
        dev_gold2_variables.append(torch.tensor(sel_gen_relations,dtype=torch.long, device=device))
        assert p == len(instance[4])

        p = 0
        all_variables = []
        dev_input3_variable = []
        for i in range(len(instance[5])-1):
            idx = instance[5][i]
            idy = instance[5][i+1]
            if (idx>=14 and idx<=16) or (idx >= decoder.tags_info.global_start and idx < decoder.tags_info.p_rel_start) or \
                    (idx >= decoder.tags_info.tag_size and not idy==17):
                all_variables = all_variables + instance[6][p]
                all_variables.append(1)
                dev_input3_variable.append(torch.tensor(instance[6][p],dtype=torch.long, device=device))
                p += 1
        dev_input3_variables.append(dev_input3_variable)
        dev_gold3_variables.append(torch.tensor(all_variables,dtype=torch.long, device=device))
        assert p == len(instance[6])

        ##### mask1
        p_max = 0
        decoder.outer_mask_pool.reset()
        mask1 = []
        mask1.append(decoder.outer_mask_pool.get_step_mask())
        for idx in instance[3]:
            assert mask1[-1][idx] == decoder.outer_mask_pool.need
            if idx >= decoder.tags_info.p_rel_start and idx < decoder.tags_info.p_tag_start:
                p_max += 1
            decoder.outer_mask_pool.update(-2, idx)
            mask1.append(decoder.outer_mask_pool.get_step_mask())

        ##### mask2
        decoder.rel_mask_pool.reset(len(instance[0]))
        mask2 = []
        p = 0
        for i in range(len(instance[3])):
            idx = instance[3][i]
            if idx == 4:
                least = False
                if (idx == 4 and instance[3][i + 1] == 2):
                    least = True
                decoder.rel_mask_pool.set_sdrs(idx == 3)
                decoder.rel_mask_pool.reset(len(instance[0]))
                temp_mask = []
                temp_mask.append(decoder.rel_mask_pool.get_step_mask(least, -1))
                if least==True:
                    least=False
                for k in range(len(instance[4][p])):
                    temp_idx = instance[4][p][k][0]
                    if temp_idx == -2:
                        temp_idx = instance[4][p][k][1]
                    else:
                        temp_idx += decoder.tags_info.tag_size
                    assert temp_mask[k][temp_idx] == decoder.rel_mask_pool.need
                    temp_mask.append(decoder.rel_mask_pool.get_step_mask(least, temp_idx))
                mask2 = mask2 + temp_mask
                p += 1
        assert p == len(instance[4])

        #### mask3
        decoder.var_mask_pool.reset(p_max)
        mask3 = []
        p = 0
        for i in range(len(instance[5]) - 1):
            idx = instance[5][i]
            idy = instance[5][i + 1]
            decoder.var_mask_pool.update(idx)
            if (idx >= 14 and idx <= 16) or (
                    idx >= decoder.tags_info.global_start and idx < decoder.tags_info.p_rel_start) or \
                    (idx >= decoder.tags_info.tag_size and not idy == 17):
                for idxx in instance[6][p]:
                    mask3.append(decoder.var_mask_pool.get_step_mask())
                    decoder.var_mask_pool.update(idxx)
                    assert mask3[-1][idxx] == decoder.var_mask_pool.need
                mask3.append(decoder.var_mask_pool.get_step_mask())
                p += 1
        assert p == len(instance[6])

        dev_mask1_variables.append(torch.tensor(mask1,dtype=torch.float, device=device))
        dev_mask2_variables.append(torch.tensor(mask2,dtype=torch.float, device=device))
        dev_mask3_variables.append(torch.tensor(mask3,dtype=torch.float, device=device))
    # ====================================== test
    tst_sentence_variables = []

    for instance in tst_instances:
        tst_sentence_variable = []
        tst_sentence_variable.append(instance[0].to(device))
        tst_sentence_variable.append(instance[1].to(device))
        tst_sentence_variable.append(instance[2].to(device))
        tst_sentence_variable.append(instance[7].to(device))
        tst_sentence_variable.append(instance[8].to(device))
        tst_sentence_variable.append(instance[9])

        tst_sentence_variables.append(tst_sentence_variable)

    # ======================================
    idx = -1
    iter = 0
    if args.reload:
        iter = check_point["iter"]
        idx = check_point["idx"]


    while True:
        if use_cuda:
            torch.cuda.empty_cache()
        idx += 1
        iter += 1
        if idx == len(trn_instances):
            idx = 0
        loss1, loss2, loss3 = train(sentence_variables[idx],
                                    (input1_variables[idx], input2_variables[idx], input3_variables[idx]),
                                    (gold1_variables[idx], gold2_variables[idx], gold3_variables[idx]),
                                    (mask1_variables[idx], mask2_variables[idx], mask3_variables[idx]), encoder,
                                    decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total1 += loss1
        print_loss_total2 += loss2
        print_loss_total3 += loss3
        print_loss_total += (loss1 + loss2 + loss3)

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0

            print_loss_avg1 = print_loss_total1 / print_every
            print_loss_total1 = 0
            print_loss_avg2 = print_loss_total2 / print_every
            print_loss_total2 = 0
            print_loss_avg3 = print_loss_total3 / print_every
            print_loss_total3 = 0
            print('epoch %.6f : %.10f s1: %.10f s2: %.10f s3: %.10f' % (
            iter * 1.0 / len(trn_instances), print_loss_avg, print_loss_avg1, print_loss_avg2, print_loss_avg3))
            log_dir.add_scalar('trainLoss', print_loss_avg, iter * 1.0 / len(trn_instances))

        if iter % (evaluate_every*len(trn_instances)) == 0:
            with torch.no_grad():
                dev_idx = 0
                dev_loss = 0.0
                dev_loss1 = 0.0
                dev_loss2 = 0.0
                dev_loss3 = 0.0
                if args.savemodel:
                    torch.save(
                        {"iter": iter, "idx": idx, "encoder": encoder.state_dict(), "decoder": decoder.state_dict(),
                         "encoder_optimizer": encoder_optimizer.state_dict(),
                         "decoder_optimizer": decoder_optimizer.state_dict()},
                        model_dir + str(int(iter / len(trn_instances))) + ".model")

                while dev_idx < len(dev_instances):
                    if use_cuda:
                        torch.cuda.empty_cache()
                    a, b, c = train(dev_sentence_variables[dev_idx], (
                        dev_input1_variables[dev_idx], dev_input2_variables[dev_idx], dev_input3_variables[dev_idx]), (
                                        dev_gold1_variables[dev_idx], dev_gold2_variables[dev_idx],
                                        dev_gold3_variables[dev_idx]), (
                                        dev_mask1_variables[dev_idx], dev_mask2_variables[dev_idx],
                                        dev_mask3_variables[dev_idx]), encoder, decoder, encoder_optimizer,
                                    decoder_optimizer,
                                    criterion, back_prop=False)
                    dev_loss1 += a
                    dev_loss2 += b
                    dev_loss3 += c
                    dev_loss += (a + b + c)
                    dev_idx += 1
                print('epoch %.6f : dev loss %.10f, s1: %.10f, s2: %.10f, s3: %.10f ' % (
                    iter * 1.0 / len(trn_instances),dev_loss / len(dev_instances), dev_loss1 / len(dev_instances), dev_loss2 / len(dev_instances),
                    dev_loss3 / len(dev_instances)))
                evaluate(dev_sentence_variables, encoder, decoder,
                         dev_out_dir + str(int(iter / len(trn_instances))) + ".drs")
                log_dir.add_scalar('devLoss', dev_loss / len(dev_instances), iter * 1.0 / len(trn_instances))
                evaluate(tst_sentence_variables, encoder, decoder,
                         tst_out_dir + str(int(iter / len(trn_instances))) + ".drs")


def evaluate(sentence_variables, encoder, decoder, path):
    out = open(path, "w",encoding='utf-8')
    for idx in range(len(sentence_variables)):
        if use_cuda:
            torch.cuda.empty_cache()
        structs, tokens = decode(sentence_variables[idx], encoder, decoder)
        p = 0
        output = []
        flag=0
        for i in range(len(structs)):
            if structs[i] < decoder.tags_info.tag_size and not structs[i]==17:
                output.append(decoder.tags_info.ix_to_tag[structs[i]])
            elif structs[i]==17:
                flag=1
                if structs[i + 1] < decoder.tags_info.tag_size:
                    print(structs)
                assert(structs[i+1]>=decoder.tags_info.tag_size)
            else:
                if flag==0:
                    output.append(sentence_variables[idx][5][structs[i] - decoder.tags_info.tag_size]+'(')
                else:
                    output[-1]=output[-1][:-1]+'~'+sentence_variables[idx][5][structs[i] - decoder.tags_info.tag_size]+'('
                    flag=0
            if (structs[i] >=14 and structs[i]<=16) or (structs[i] >= decoder.tags_info.global_start and structs[i] < decoder.tags_info.p_rel_start) \
                    or (structs[i] >= decoder.tags_info.tag_size and not structs[i+1]==17):
                if p>=len(tokens):
                    print(structs)
                    print(tokens)
                for idy in tokens[p]:
                    output.append(decoder.tags_info.ix_to_tag[idy])
                p += 1
        assert p == len(tokens)
        out.write(" ".join(output) + "\n")
        out.flush()
    out.close()


def test(dev_instances, tst_instances, encoder, decoder):
    # ====================================== test
    dev_sentence_variables = []

    for instance in dev_instances:
        dev_sentence_variable = []
        dev_sentence_variable.append(instance[0].to(device))
        dev_sentence_variable.append(instance[1].to(device))
        dev_sentence_variable.append(instance[2].to(device))
        dev_sentence_variable.append(instance[7].to(device))
        dev_sentence_variable.append(instance[8].to(device))
        dev_sentence_variable.append(instance[9])
        dev_sentence_variables.append(dev_sentence_variable)



    # ====================================== test
    tst_sentence_variables = []

    for instance in tst_instances:
        tst_sentence_variable = []
        tst_sentence_variable.append(instance[0].to(device))
        tst_sentence_variable.append(instance[1].to(device))
        tst_sentence_variable.append(instance[2].to(device))
        tst_sentence_variable.append(instance[7].to(device))
        tst_sentence_variable.append(instance[8].to(device))
        tst_sentence_variable.append(instance[9])

        tst_sentence_variables.append(tst_sentence_variable)


    check_point = {}
    check_point = torch.load(args.model)
    encoder.load_state_dict(check_point["encoder"])
    decoder.load_state_dict(check_point["decoder"])
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    evaluate(dev_sentence_variables, encoder, decoder, dev_out_dir + "dev.drs")
    evaluate(tst_sentence_variables, encoder, decoder, tst_out_dir + "tst.drs")
#####################################################################################




def main():

    trn_data = readfile(trn_file)
    word_to_ix = {UNK: 0}
    lemma_to_ix = {UNK: 0}
    ix_to_lemma = [UNK]
    postag_to_ix = {UNK: 0,'punc':1}
    dep_to_ix={UNK:0}
    for sentence, _, lemmas, posTags, dps, _, tags in trn_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
        for lemma in lemmas:
            if lemma not in lemma_to_ix:
                lemma_to_ix[lemma] = len(lemma_to_ix)
                ix_to_lemma.append(lemma)
        for posTag in posTags:
            if posTag not in postag_to_ix:
                postag_to_ix[posTag]=len(postag_to_ix)
        for tag, head ,dep in dps:
            if tag not in dep_to_ix:
                dep_to_ix[tag]=len(dep_to_ix)

    #############################################
    ## tags
    tags_info = Tag(tag_info_file, ix_to_lemma)
    outer_mask_pool = OuterMask(tags_info)
    rel_mask_pool = RelationMask(tags_info)
    var_mask_pool = VariableMask(tags_info)
    ##############################################
    ##
    # mask_info = Mask(tags)
    #############################################
    pretrain_to_ix = {UNK: 0}
    pretrain_embeddings = [[0. for i in range(100)]]  # for UNK
    pretrain_data = readpretrain(pretrain_file)

    for one in pretrain_data[1:]:
        pretrain_to_ix[one[0]] = len(pretrain_to_ix)
        pretrain_embeddings.append([float(a) for a in one[1:]])
    assert (len(pretrain_to_ix) == int(pretrain_data[0][0]) + 1)
    assert (len(pretrain_embeddings[0]) == int(pretrain_data[0][1]))
    print("pretrain dict size:", len(pretrain_to_ix))

    dev_data = readfile(dev_file)
    tst_data = readfile(tst_file)

    print("word dict size: ", len(word_to_ix))
    print("lemma dict size: ", len(lemma_to_ix))
    print("global tag (w/o variables) dict size: ", tags_info.p_rel_start)
    print("global tag (w variables) dict size: ", tags_info.tag_size)

    #print(pretrain_to_ix.items())
    ###########################################################
    # prepare training instance
    trn_instances = data2instance(trn_data, [(word_to_ix, 0), (pretrain_to_ix, 0), (lemma_to_ix, 0), tags_info,(postag_to_ix,0),(dep_to_ix,0)],trn_gold_file)
    print("trn size: " + str(len(trn_instances)))
    ###########################################################
    # prepare development instance
    dev_instances = data2instance(dev_data, [(word_to_ix, 0), (pretrain_to_ix, 0), (lemma_to_ix, 0), tags_info,(postag_to_ix,0),(dep_to_ix,0)],dev_gold_file)
    print("dev size: " + str(len(dev_instances)))
    ###########################################################
    # prepare test instance
    tst_instances = data2instance(tst_data, [(word_to_ix, 0), (pretrain_to_ix, 0), (lemma_to_ix, 0), tags_info,(postag_to_ix,0),(dep_to_ix,0)],tst_gold_file)
    print("tst size: " + str(len(tst_instances)))

    encoder = EncoderRNN(len(word_to_ix), WORD_EMBEDDING_DIM, len(pretrain_to_ix), PRETRAIN_EMBEDDING_DIM,
                         torch.tensor(pretrain_embeddings,dtype=torch.float, device=device),
                         len(lemma_to_ix), LEMMA_EMBEDDING_DIM, len(postag_to_ix), POS_TAG_DIM, len(dep_to_ix), DEP_TAG_DIM,INPUT_DIM,
                         ENCODER_HIDDEN_DIM, n_layers=ENCODER_LAYER, dropout_p=ENCODER_DROUPOUT_RATE).to(device)
    attn_decoder = AttnDecoderRNN(outer_mask_pool, rel_mask_pool, var_mask_pool, tags_info, TAG_DIM, DECODER_INPUT_DIM,
                                  ENCODER_HIDDEN_DIM, ATTENTION_HIDDEN_DIM, n_layers=DECODER_LAYER,
                                  dropout_p=DECODER_DROPOUT_RATE).to(device)

    if args.test:
        test(dev_instances, tst_instances, encoder, attn_decoder)
    else:
        writer = SummaryWriter(log_dir)
        trainIters(trn_instances, dev_instances, tst_instances, encoder, attn_decoder, print_every=PRINT_EVERY,
                   evaluate_every=EVALUATE_EVERY_EPOCH, learning_rate=LEARNING_RATE,log_dir=writer)
        writer.close()


if __name__ == "__main__":
    main()
