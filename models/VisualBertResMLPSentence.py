'''
Description     : VisualBertResMLP + Transformer based sentence generation model.
Paper           : Surgical-VQA: Visual Question Answering in Surgical Scenes Using Transformers
Author          : Lalithkumar Seenivasan, Mobarakol Islam, Adithya Krishna, Hongliang Ren
Lab             : MMLAB, National University of Singapore
Acknowledgement : Code adopted from the official implementation of VisualBertModel from 
                  huggingface/transformers (https://github.com/huggingface/transformers.git) and modified.
'''

import numpy as np

import torch
from torch import nn
from transformers import VisualBertConfig
from models.VisualBertResMLP import VisualBertResMLPModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
channel_number = 512


'''
Encoder Transformer: VisualBertResMLP Encoder
'''
class VisualBertResMLPEncoder(nn.Module):
    def __init__(self, vocab_size, layers, n_heads, token_size = 26):
        super(VisualBertResMLPEncoder, self).__init__()
        VBconfig = VisualBertConfig(vocab_size= vocab_size, visual_embedding_dim = 512, num_hidden_layers = layers, num_attention_heads = n_heads, hidden_size = 2048)
        self.VisualBertResMLPEncoder = VisualBertResMLPModel(VBconfig, token_size = token_size)


    def forward(self, inputs, visual_embeds):
        # prepare visual embedding
        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long).to(device)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float).to(device)

        # append visual features to text
        inputs.update({
                        "visual_embeds": visual_embeds,
                        "visual_token_type_ids": visual_token_type_ids,
                        "visual_attention_mask": visual_attention_mask,
                        "output_attentions": True
                        })
                        
        inputs['input_ids'] = inputs['input_ids'].to(device)
        inputs['token_type_ids'] = inputs['token_type_ids'].to(device)
        inputs['attention_mask'] = inputs['attention_mask'].to(device)
        inputs['visual_token_type_ids'] = inputs['visual_token_type_ids'].to(device)
        inputs['visual_attention_mask'] = inputs['visual_attention_mask'].to(device)

        # Encoder output
        outputs = self.VisualBertResMLPEncoder(**inputs)
        
        return outputs


'''
Decoder transformer
'''
class ScaledDotProductAttention(nn.Module):
    def __init__(self, QKVdim):
        super(ScaledDotProductAttention, self).__init__()
        self.QKVdim = QKVdim

    def forward(self, Q, K, V, attn_mask):
        """
        :param Q: [batch_size, n_heads, -1(len_q), QKVdim]
        :param K, V: [batch_size, n_heads, -1(len_k=len_v), QKVdim]
        :param attn_mask: [batch_size, n_heads, len_q, len_k]
        """
        # scores: [batch_size, n_heads, len_q, len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.QKVdim)
        # Fills elements of self tensor with value where mask is True.
        scores.to(device).masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)  # [batch_size, n_heads, len_q, len_k]
        context = torch.matmul(attn, V).to(device)  # [batch_size, n_heads, len_q, QKVdim]
        return context, attn


class Multi_Head_Attention(nn.Module):
    def __init__(self, Q_dim, K_dim, QKVdim, n_heads=8, dropout=0.1):
        super(Multi_Head_Attention, self).__init__()
        self.W_Q = nn.Linear(Q_dim, QKVdim * n_heads)
        self.W_K = nn.Linear(K_dim, QKVdim * n_heads)
        self.W_V = nn.Linear(K_dim, QKVdim * n_heads)
        self.n_heads = n_heads
        self.QKVdim = QKVdim
        self.embed_dim = Q_dim
        self.dropout = nn.Dropout(p=dropout)
        self.W_O = nn.Linear(self.n_heads * self.QKVdim, self.embed_dim)

    def forward(self, Q, K, V, attn_mask):
        """
        In self-encoder attention:
                Q = K = V: [batch_size, num_pixels=26, encoder_dim=2048]
                attn_mask: [batch_size, len_q=26, len_k=26]
        In self-decoder attention:
                Q = K = V: [batch_size, max_len=20, embed_dim=512]
                attn_mask: [batch_size, len_q=20, len_k=20]
        encoder-decoder attention:
                Q: [batch_size, 20, 512] from decoder
                K, V: [batch_size, 26, 2048] from encoder
                attn_mask: [batch_size, len_q=20, len_k=26]
        return _, attn: [batch_size, n_heads, len_q, len_k]
        """
        residual, batch_size = Q, Q.size(0)
        # q_s: [batch_size, n_heads=8, len_q, QKVdim] k_s/v_s: [batch_size, n_heads=8, len_k, QKVdim]
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.QKVdim).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.QKVdim).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.QKVdim).transpose(1, 2)
        # attn_mask: [batch_size, self.n_heads, len_q, len_k]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        # attn: [batch_size, n_heads, len_q, len_k]
        # context: [batch_size, n_heads, len_q, QKVdim]
        context, attn = ScaledDotProductAttention(self.QKVdim)(q_s, k_s, v_s, attn_mask)
        # context: [batch_size, n_heads, len_q, QKVdim] -> [batch_size, len_q, n_heads * QKVdim]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.QKVdim).to(device)
        # output: [batch_size, len_q, embed_dim]
        output = self.W_O(context)
        output = self.dropout(output)
        return nn.LayerNorm(self.embed_dim).to(device)(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, embed_dim, d_ff, dropout):
        '''
        PosewiseFeedForwardNet
        embed_dim = 300
        d_ff      = dim_size
        dropout`  = 0.1
        '''
        super(PoswiseFeedForwardNet, self).__init__()
        """
        Two fc layers can also be described by two cnn with kernel_size=1.
        """
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=embed_dim, kernel_size=1)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_dim = embed_dim

    def forward(self, inputs):
        """
        encoder: inputs: [batch_size, len_q=26, embed_dim=2048]
        decoder: inputs: [batch_size, max_len=20, embed_dim=512]
        """
        residual = inputs
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        output = self.dropout(output)
        return nn.LayerNorm(self.embed_dim).to(device)(output + residual)


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, dropout, n_heads):
        '''
        Decoder layer
        embed_dim   = 300
        droput      = 0.1
        n_heads     = 6
        '''
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = Multi_Head_Attention(Q_dim=embed_dim, K_dim=embed_dim, QKVdim=64, n_heads=n_heads, dropout=dropout)
        self.dec_enc_attn = Multi_Head_Attention(Q_dim=embed_dim, K_dim=2048, QKVdim=64, n_heads=n_heads, dropout=dropout)
        self.pos_ffn = PoswiseFeedForwardNet(embed_dim=embed_dim, d_ff=2048, dropout=dropout)
        
    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        """
        :param dec_inputs: [batch_size, max_len=20, embed_dim=512]
        :param enc_outputs: [batch_size, num_pixels=26, 2048]
        :param dec_self_attn_mask: [batch_size, 20, 20]
        :param dec_enc_attn_mask: [batch_size, 20, 26]
        """
        # print(dec_inputs.shape, enc_outputs.shape, dec_self_attn_mask.shape, dec_enc_attn_mask.shape)

        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):
    def __init__(self, n_layers, vocab_size, embed_dim, dropout, n_heads, answer_len):
        '''
        Transformer decoder
        n_layers    = 6
        vocab_size  = tokenizer length
        embed_fim   = 300
        dropout     = 0.1
        n_heads     = 6
        answer_len  = 20
        '''
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.anwer_len = answer_len
        self.tgt_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding.from_pretrained(self.get_position_embedding_table(embed_dim), freeze=True)
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList([DecoderLayer(embed_dim, dropout, n_heads) for _ in range(n_layers)])
        self.projection = nn.Linear(embed_dim, vocab_size, bias=False)

    def get_position_embedding_table(self, embed_dim):
        def cal_angle(position, hid_idx):
            return position / np.power(10000, 2 * (hid_idx // 2) / embed_dim)
        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_idx) for hid_idx in range(embed_dim)]

        embedding_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(self.anwer_len)])
        embedding_table[:, 0::2] = np.sin(embedding_table[:, 0::2])  # dim 2i
        embedding_table[:, 1::2] = np.cos(embedding_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(embedding_table).to(device)

    def get_attn_pad_mask(self, seq_q, seq_k):
        batch_size, len_q = seq_q.size()
        batch_size, len_k = seq_k.size()
        # In wordmap, <pad>:0
        # pad_attn_mask: [batch_size, 1, len_k], one is masking
        pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
        return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

    def get_attn_subsequent_mask(self, seq):
        attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
        subsequent_mask = np.triu(np.ones(attn_shape), k=1)
        subsequent_mask = torch.from_numpy(subsequent_mask).byte().to(device)
        return subsequent_mask

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        :param encoder_out: [batch_size, num_pixels=26, 2048]
        :param encoded_captions: [batch_size, 20]
        :param caption_lengths: [batch_size, 1]
        """
        batch_size = encoder_out.size(0)
        token_size = encoder_out.size(1)
        # Sort input data by decreasing lengths.
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        '''# dec_outputs: [batch_size, max_len=20, embed_dim=512]
        # dec_self_attn_pad_mask: [batch_size, len_q=20, len_k=20], 1 if id=0(<pad>)
        # dec_self_attn_subsequent_mask: [batch_size, 20, 20], Upper triangle of an array with 1.
        # dec_self_attn_mask for self-decoder attention, the position whose val > 0 will be masked.
        # dec_enc_attn_mask for encoder-decoder attention.
        # e.g. 9488, 23, 53, 74, 0, 0  |  dec_self_attn_mask:
        # 0 1 1 1 2 2
        # 0 0 1 1 2 2
        # 0 0 0 1 2 2
        # 0 0 0 0 2 2
        # 0 0 0 0 1 2
        # 0 0 0 0 1 1'''
        dec_outputs = self.tgt_emb(encoded_captions) + self.pos_emb(torch.LongTensor([list(range(self.anwer_len))]*batch_size).to(device))
        dec_outputs = self.dropout(dec_outputs)
        dec_self_attn_pad_mask = self.get_attn_pad_mask(encoded_captions, encoded_captions)
        dec_self_attn_subsequent_mask = self.get_attn_subsequent_mask(encoded_captions)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        dec_enc_attn_mask = (torch.tensor(np.zeros((batch_size, self.anwer_len, token_size))).to(device) == torch.tensor(np.ones((batch_size, self.anwer_len, token_size))).to(device))

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # attn: [batch_size, n_heads, len_q, len_k]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, encoder_out, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        predictions = self.projection(dec_outputs)
        return predictions, encoded_captions, decode_lengths, sort_ind, dec_self_attns, dec_enc_attns


'''
VisualBertResMLP Encoder + Transformer decoder
'''
class VisualBertResMLPSentence(nn.Module):

    def __init__(self, vocab_size, embed_dim, encoder_layers, decoder_layers, dropout=0.1, n_heads=8, token_size = 26, answer_len = 20):
        '''
        VisualBertResMLP Encoder + Transformer decoder
        vocab_size     = tokenizer length
        embed_dim      = 300
        encoder_layers = 6
        decoder_layers = 6
        dropout        = 0.1
        n_heads        = 6
        answer_len     = 20
        '''
        super(visualBertResMLPSentence, self).__init__()
        
        self.encoder = VisualBertResMLPEncoder(vocab_size, encoder_layers, n_heads, token_size)
        self.decoder = Decoder(decoder_layers, vocab_size, embed_dim, dropout, n_heads, answer_len)
        self.embedding = self.decoder.tgt_emb

    def load_pretrained_embeddings(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def forward(self, inputs, visual_embeds, encoded_captions, caption_lengths):
        # Vision and text encoder output
        encoder_outputs = self.encoder(inputs, visual_embeds)
        
        # predict answer using decoder model
        predictions, encoded_captions, decode_lengths, sort_ind, dec_self_attns, dec_enc_attns = self.decoder(encoder_outputs['last_hidden_state'], encoded_captions, caption_lengths)
        alphas = {"enc_self_attns": encoder_outputs['attentions'], "dec_self_attns": dec_self_attns, "dec_enc_attns": dec_enc_attns}
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
