import time
import codecs
import argparse
import numpy as np

import torch.optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from torch import nn
from transformers import BertTokenizer
from torch.utils.data  import DataLoader
from nltk.translate.bleu_score import corpus_bleu
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu

from utils import *
from transformer import *
from dataloaders.dataloader_VQA_miccai18 import *

def train(args, train_loader, model, criterion, optimizer, epoch, tokenizer):
    """
    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: loss layer
    :param optimizer: optimizer to update model's weights
    :param epoch: epoch number
    """

    model.train()  # train mode (dropout and batchnorm is used)
    
    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    # top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    for i, (imgs, q, a) in enumerate(train_loader,0):

        data_time.update(time.time() - start)

        # prepare questions
        questions = []
        for question in q: questions.append(question)
        inputs = tokenizer(questions, return_tensors="pt", padding="max_length", max_length=25)

        # prepare answers
        answers = []
        for answer in a: answers.append(answer)
        answers_GT = tokenizer(answers, return_tensors="pt", padding="max_length", max_length=20)
        answers_GT_ID = answers_GT.input_ids
        answers_GT_len = torch.sum(answers_GT.attention_mask, dim=1).unsqueeze(1)

        # GPU / CPU
        imgs = imgs.to(device)
        answers_GT_ID = answers_GT_ID.to(device)
        answers_GT_len = answers_GT_len.to(device)

        #Visual Question and Answering                          
        scores, answer_GT_ID_sorted, decode_lengths, alphas, sort_ind = model(inputs, imgs, answers_GT_ID, answers_GT_len)

        
        # pack_padded_sequence is an easy trick to do this, for target, we remove 1st <start> element
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(answer_GT_ID_sorted[:, 1:], decode_lengths, batch_first=True).data
        
        # Calculate loss
        loss = criterion(scores, targets)
        
        #later integrate attention loss
        # dec_alphas = alphas["dec_enc_attns"]
        # alpha_trans_c = args.alpha_c / (args.n_heads * args.decoder_layers)
        
        # for layer in range(args.decoder_layers):  # args.decoder_layers = len(dec_alphas)
        #     cur_layer_alphas = dec_alphas[layer]  # [batch_size, n_heads, 20, 26]
        #     for h in range(args.n_heads):
        #         cur_head_alpha = cur_layer_alphas[:, h, :, :]
        #         loss += alpha_trans_c * ((1. - cur_head_alpha.sum(dim=1)) ** 2).mean()
        

        # Back prop.
        optimizer.zero_grad()
        loss.backward()
        # if args.grad_clip is not None: clip_gradient(optimizer, args.grad_clip) # clip_grad
        optimizer.step()

        # Keep track of metrics
        # top5 = accuracy(scores, targets, 5)
        # top5accs.update(top5, sum(decode_lengths))
        losses.update(loss.item(), sum(decode_lengths))
        batch_time.update(time.time() - start)
        start = time.time()
        if i % args.print_freq == 0:
            print("Epoch: {}/{} step: {}/{} Loss: {:.6f} AVG_Loss: {:.6f} Batch_time: {:.6f}s".format(epoch+1, args.epochs, i+1, len(train_loader), losses.val, losses.avg, batch_time.val))


def validate(args, val_loader, model, criterion, tokenizer):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param model: model
    :param criterion: loss layer
    :return: score_dict {'Bleu_1': 0., 'Bleu_2': 0., 'Bleu_3': 0., 'Bleu_4': 0., 'METEOR': 0., 'ROUGE_L': 0., 'CIDEr': 1.}
    """
    model.eval()  # eval mode (no dropout or batchnorm)

    batch_time = AverageMeter()
    losses = AverageMeter()
    # top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    with torch.no_grad():
        
        for i, (imgs, q, a) in enumerate(val_loader,0):
            
            # prepare questions
            questions = []
            for question in q: questions.append(question)
            inputs = tokenizer(questions, return_tensors="pt", padding="max_length", max_length=25)

            # prepare answers
            answers = []
            for answer in a: answers.append(answer)
            answers_GT = tokenizer(answers, return_tensors="pt", padding="max_length", max_length=20)
            answers_GT_ID = answers_GT.input_ids
            answers_GT_len = torch.sum(answers_GT.attention_mask, dim=1).unsqueeze(1)
            
            # GPU / CPU
            imgs = imgs.to(device)
            answers_GT_ID = answers_GT_ID.to(device)
            answers_GT_len = answers_GT_len.to(device)

            #Visual Question and Answering
            scores, answer_GT_ID_sorted, decode_lengths, alphas, sort_ind = model(inputs, imgs, answers_GT_ID, answers_GT_len)

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this, for target, remove the <start> in the first element
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(answer_GT_ID_sorted[:, 1:], decode_lengths, batch_first=True).data

            # Calculate loss
            loss = criterion(scores, targets)
            
            
            # dec_alphas = alphas["dec_enc_attns"]
            # alpha_trans_c = args.alpha_c / (args.n_heads * args.decoder_layers)
            # for layer in range(args.decoder_layers):  # args.decoder_layers = len(dec_alphas)
            #     cur_layer_alphas = dec_alphas[layer]  # [batch_size, n_heads, 20, 196]
            #     for h in range(args.n_heads):
            #         cur_head_alpha = cur_layer_alphas[:, h, :, :]
            #         loss += alpha_trans_c * ((1. - cur_head_alpha.sum(dim=1)) ** 2).mean()
            

            # Keep track of metrics
            # top5 = accuracy(scores, targets, 5)
            # top5accs.update(top5, sum(decode_lengths))
            losses.update(loss.item(), sum(decode_lengths))
            batch_time.update(time.time() - start)
            start = time.time()

            # references
            answer_GT_sorted = tokenizer.batch_decode(answer_GT_ID_sorted, skip_special_tokens= True)
            for answer_GT_sorted_i in answer_GT_sorted: references.append([answer_GT_sorted_i.split()])

            # print(references)

            # Hypotheses
            _, predicted_answer_id = torch.max(scores_copy, dim=2)
            predicted_answer = tokenizer.batch_decode(predicted_answer_id, skip_special_tokens= True)
            for pa in predicted_answer: hypotheses.append(pa.split())

            # print(hypotheses)
            # print(decode_lengths)

    # Calculate BLEU1~4
    metrics = {}
    metrics["Bleu_1"] = corpus_bleu(references, hypotheses, weights=(0.50, 0.50, 0.00, 0.00))
    metrics["Bleu_2"] = corpus_bleu(references, hypotheses, weights=(0.50, 0.50, 0.00, 0.00))
    metrics["Bleu_3"] = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0.00))
    metrics["Bleu_4"] = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

    print("EVA LOSS: {:.6f} BLEU-1 {:.6f} BLEU2 {:.6f} BLEU3 {:.6f} BLEU-4 {:.6f}".format
          (losses.avg, metrics["Bleu_1"],  metrics["Bleu_2"],  metrics["Bleu_3"],  metrics["Bleu_4"]))

    return metrics


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='VisualQuestionAnswer')
    
    # Model parameters
    parser.add_argument('--emb_dim',        type=int,   default=300,    help='dimension of word embeddings.')
    parser.add_argument('--n_heads',        type=int,   default=8,      help='Multi-head attention.')
    parser.add_argument('--dropout',        type=float, default=0.1,    help='dropout')
    parser.add_argument('--encoder_layers', type=int,   default=6,      help='the number of layers of encoder in Transformer.')
    parser.add_argument('--decoder_layers', type=int,   default=6,      help='the number of layers of decoder in Transformer.')
    
    # Training parameters
    parser.add_argument('--epochs',         type=int,   default=100,    help='number of epochs to train for (if early stopping is not triggered).')
    parser.add_argument('--lr',             type=float, default=0.0001, help='learning rate.')
    parser.add_argument('--batch_size',     type=int,   default=20,     help='batch_size')
    parser.add_argument('--workers',        type=int,   default=1,      help='for data-loading; right now, only 1 works with h5pys.')
    parser.add_argument('--stop_criteria',  type=int,   default=25,     help='training stop if epochs_since_improvement == stop_criteria')
    parser.add_argument('--print_freq',     type=int,   default=50,     help='print training/validation stats every __ batches.')
    
    # parser.add_argument('--grad_clip',      type=float, default=5.,   help='clip gradients at an absolute value of.')
    parser.add_argument('--alpha_c',        type=float, default=1.,   help='regularization parameter for doubly stochastic attention, as in the paper.')
    
    parser.add_argument('--checkpoint',     default=None, help='path to checkpoint, None if none.')
    parser.add_argument('--embedding_path', default=None, help='path to pre-trained word Embedding.')
    args = parser.parse_args()

    checkpoint_dir = 'checkpoints/v6/'
    
    # train and test dataloader
    train_seq = [2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15]
    val_seq = [1, 5, 16]
    folder_head = 'dataset/instruments18/seq_'
    folder_tail = '/vqa/complex/*.txt'

    # load checkpoint, these parameters can't be modified
    final_args = {"emb_dim": args.emb_dim, "n_heads": args.n_heads, "dropout": args.dropout, 
                "encoder_layers": args.encoder_layers, "decoder_layers": args.decoder_layers}


    # GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
    cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
    print('device =',device)

    # best model initialize
    start_epoch = 0
    best_bleu4 = 0.  # BLEU-4 score right now
    epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
    
    
    # Initialize / load checkpoint
    if args.checkpoint is None:

        model = Transformer(vocab_size=4000, embed_dim=args.emb_dim, encoder_layers=args.encoder_layers,
                              decoder_layers=args.decoder_layers, dropout=args.dropout, n_heads=args.n_heads)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # model.fine_tune_embeddings(True)

    else:
        checkpoint = torch.load(args.checkpoint, map_location=str(device))
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['metrics']["Bleu_4"]
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        # model.fine_tune_embeddings(True)
        # load final_args from checkpoint
        final_args = checkpoint['final_args']
        for key in final_args.keys(): args.__setattr__(key, final_args[key])


    # Move to GPU, if available
    model = model.to(device)
    print("encoder_layers {} decoder_layers {} n_heads {} dropout {} lr {} alpha_c {}".format(args.encoder_layers, 
                args.decoder_layers, args.n_heads, args.dropout, args.lr, args.alpha_c))    
    print(model)

    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)

    # dataset and dataloader
    transform = transforms.Compose([
                transforms.Resize((300,256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ])

    train_dataset = SurgicalSentenceVQADataset(train_seq, folder_head, folder_tail, transform=transform)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size= args.batch_size, shuffle=True)

    val_dataset = SurgicalSentenceVQADataset(val_seq, folder_head, folder_tail, transform=transform)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size= 40, shuffle=False)

    # custom trained tokenizer
    tokenizer = BertTokenizer.from_pretrained('./dataset/bertvocab/v1/bert-miccai18/bert-miccai18-vocab.txt')

    for epoch in range(start_epoch, args.epochs):

        # if epochs_since_improvement == args.stop_criteria:
        #     print("the model has not improved in the last {} epochs".format(args.stop_criteria))
        #     break
        if epochs_since_improvement > 0 and epochs_since_improvement % 5 == 0:
            adjust_learning_rate(optimizer, 0.8)

        # training
        train(args, train_loader=train_dataloader, model = model, criterion=criterion, optimizer=optimizer, epoch=epoch, tokenizer = tokenizer)
        
        # validation
        metrics = validate(args, val_loader=val_dataloader, model = model, criterion=criterion, tokenizer = tokenizer)
        
        # Check if there was an improvement
        is_best = metrics["Bleu_4"] > best_bleu4
        best_bleu4 = max(metrics["Bleu_4"], best_bleu4)
        
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, metrics, is_best, final_args)