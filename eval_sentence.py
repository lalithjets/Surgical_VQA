#!/usr/bin/env python3


import argparse
from tqdm import tqdm

import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from nltk.translate.bleu_score import corpus_bleu

from utils import *
# from dataloaders.dataloader_VQA_miccai18_3 import *
# from dataloaders.dataloader_VQAMed19_3 import *
from dataloaders.dataloaderSentence import *
from models.VisualBertResMLPSentence import *


import pandas as pd

'''
    !pip -q install pycocoevalcap
'''
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

def seed_everything(seed=27):
    '''
    Set random seed for reproducible experiments
    Inputs: seed number 
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_transformer(args, val_dataloader, tokenizer, model, answer_len, vocab_size):
    """
    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """

    model.eval()

    beam_size = args.beam_size
    Caption_End = False

    references = list()
    hypotheses = list()
    labels = list()
    predictions = list()
    file_names = list()

    with torch.no_grad():

        for i, (file_name, visual_features, q, a) in enumerate(tqdm(val_dataloader),0):
            
            k = beam_size

            # prepare questions
            questions = []
            for question in (q*k): questions.append(question)
            
            # print(q(0))
            # prepare answers
            answers = []
            for answer in a: answers.append(answer)

            # GPU / CPU
            visual_features = visual_features.to(device)
            # print(visual_features.shape)
            visual_features = visual_features.expand(k, visual_features.size(1), visual_features.size(2))

            # Tensor to store top k previous words at each step; now they're just <start>
            # Important: [1, 20] (eg: [[<start> <start> <start> ...]]) will not work, since it contains the position encoding
            k_prev_words = torch.LongTensor([tokenizer.convert_tokens_to_ids(['[CLS]'])*answer_len] * k).to(device)  # (k, 20)                              #   K, 25
            
            # Tensor to store top k sequences; now they're just <start>
            seqs = torch.LongTensor([tokenizer.convert_tokens_to_ids(['[CLS]'])] * k).to(device)  # (k, 1)                                          #   K, 1
            
            # Tensor to store top k sequences' scores; now they're just 0
            top_k_scores = torch.zeros(k, 1).to(device)                                                                                             #   k, 1
            
            # Lists to store completed sequences and scores
            complete_seqs = []
            complete_seqs_scores = []
            step = 1


            # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
            while True:
                inputs = tokenizer(questions, return_tensors="pt", padding="max_length", max_length=25)

                # print(inputs)
                answers_len = torch.LongTensor([answer_len]).repeat(k, 1)  # [s, 1]                                                                             #   k, 1
                
                #Visual Question and Answering
                scores, _, _, _, _ = model(inputs, visual_features, k_prev_words, answers_len)                                                                 #   k, 25, 4000
                scores = scores[:, step-1, :].squeeze(1)  # [s, 1, vocab_size] -> [s, vocab_size]                                                   #   k, 4000
                scores = F.log_softmax(scores, dim=1)                                                                                               #   k, 4000 
                
                # top_k_scores: [s, 1]
                scores = top_k_scores.expand_as(scores) + scores  # [s, vocab_size]
                
                # For the first step, all k points will have the same scores (since same k previous words, h, c)
                if step == 1:
                    top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
                else:
                    # Unroll and find top scores, and their unrolled indices
                    top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

                # Convert unrolled indices to actual indices of scores
                prev_word_inds = top_k_words // vocab_size  # (s)
                next_word_inds = top_k_words % vocab_size  # (s)

                # Add new words to sequences
                seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
                
                # Which sequences are incomplete (didn't reach <end>)?
                incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != tokenizer.convert_tokens_to_ids(['[SEP]'][0])]
                complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
                
                # Set aside complete sequences
                if len(complete_inds) > 0:
                    Caption_End = True
                    complete_seqs.extend(seqs[complete_inds].tolist())
                    complete_seqs_scores.extend(top_k_scores[complete_inds])
                k -= len(complete_inds)  # reduce beam length accordingly
                
                # Proceed with incomplete sequences
                if k == 0:
                    break

                seqs = seqs[incomplete_inds]
                visual_features = visual_features[prev_word_inds[incomplete_inds]]
                questions = [questions[i] for i in incomplete_inds]

                # encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
                top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                
                # Important: this will not work, since decoder has self-attention
                # k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1).repeat(k, 20)
                k_prev_words = k_prev_words[incomplete_inds]
                k_prev_words[:, :step+1] = seqs  # [s, 20]
                
                # k_prev_words[:, step] = next_word_inds[incomplete_inds]  # [s, 20]
                # Break if things have been going on too long
                if step > 18:
                    break
                step += 1

            # choose the caption which has the best_score.
            assert Caption_End
            indices = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[indices]
            
            # references
            references.append([answers[0].replace('-',' - ').lower().split()])
            s = ''
            for item in answers[0].replace('-',' - ').lower().split(): s += item + ' '
            labels.append([s])

            # prediction
            predicted_answer = tokenizer.batch_decode([seq], skip_special_tokens= True)
            hypotheses.append(predicted_answer[0].lower().split())
            file_names.append(file_name)
            s = ''
            for item in predicted_answer[0].lower().split(): s += item + ' '
            predictions.append([s])

            
            # print('predicted', predicted_answer[0].lower().split())

    '''
        To create text and csv files of GT and prediction
    '''
    # if os.path.exists(args.checkpoint_dir + 'eval_text_files_' + str(args.beam_size)) == False:
    #     os.mkdir(args.checkpoint_dir + 'eval_text_files_' + str(args.beam_size) ) 
    # file1 = open(args.checkpoint_dir + 'eval_text_files_3/references.txt', 'w')
    # file1.write(str(references))
    # file1.close()

    # file1 = open(args.checkpoint_dir + 'eval_text_files_3/hypotheses.txt', 'w')
    # file1.write(str(hypotheses))
    # file1.close()

    # file1 = open(args.checkpoint_dir + 'eval_text_files_3/labels.txt', 'w')
    # file1.write(str(labels))
    # file1.close()

    # file1 = open(args.checkpoint_dir + 'eval_text_files_3/predictions.txt', 'w')
    # file1.write(str(predictions))
    # file1.close()

    df = pd.DataFrame(columns=["Img", "Ground Truth", "Prediction"])
    for i in range(len(references)):
        ref_str = ''
        hyp_str = ''
        for item in references[i][0]: ref_str += item+ ' '
        for item in hypotheses[i]: hyp_str += item + ' '
        df = df.append({'Img': file_names[i], 'Ground Truth': ref_str, 'Prediction': hyp_str}, ignore_index=True)
    
    df.to_csv(args.checkpoint_dir + args.checkpoint_dir.split('/')[1] + '_' + args.checkpoint_dir.split('/')[2] + '_eval.csv')

    # Calculate BLEU1~4
    metrics = {}
    metrics["Bleu_1"] = corpus_bleu(references, hypotheses, weights=(1.00, 0.00, 0.00, 0.00))
    metrics["Bleu_2"] = corpus_bleu(references, hypotheses, weights=(0.50, 0.50, 0.00, 0.00))
    metrics["Bleu_3"] = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0.00))
    metrics["Bleu_4"] = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

    labels = dict(zip(np.arange(len(labels)).astype(np.float), labels))
    predictions = dict(zip(np.arange(len(predictions)).astype(np.float), predictions))
    (cider_avg, cider_per_sentence) = Cider().compute_score(labels, predictions)
    (meteor_avg, meteor_per_sentence) = Meteor().compute_score(labels, predictions)

    print("BLEU-1 {:.6f} BLEU2 {:.6f} BLEU3 {:.6f} BLEU-4 {:.6f} CIDEr {:.6f} Meteor {:.6f}".format
          (metrics["Bleu_1"],  metrics["Bleu_2"],  metrics["Bleu_3"],  metrics["Bleu_4"], cider_avg, meteor_avg))

    return metrics


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='VisualQuestionAnswer')
    parser.add_argument('--beam_size',      type=int,                   default=1,                          help='beam_size.')
    parser.add_argument('--checkpoint_dir', default= 'checkpoints/sen_v3_3_1x1/med_vqa/',        help='m18_1.2/med_vqa')
    parser.add_argument('--checkpoint',     default='checkpoints/sen_v3_2_1x1/c80/epoch_42.pth.tar',        help='model checkpoint.')
    parser.add_argument('--dataset_type',   default= 'c80',                                                 help='m18/c80/med_vqa/m18_vid/c80_vid')
    parser.add_argument('--transformer_ver',default= 'v3',                                                  help='v1/v2/v3/vbrs')
    parser.add_argument('--tokenizer_ver',  default= 'v2',                                                  help='v1/v2/v3')
    parser.add_argument('--patch_size',     default= 1,                                                     help='1/2/3/4/5')
    parser.add_argument('--temporal_size',  default= 3,                                                     help='1/2/3/4/5')
    
    args = parser.parse_args()

    if args.dataset_type == 'm18':
        '''
        Train and test for miccai dataset
        '''
        if args.tokenizer_ver == 'v2':
            tokenizer = BertTokenizer.from_pretrained('./dataset/bertvocab/v2/bert-miccai18/')
        elif args.tokenizer_ver == 'v3':
            tokenizer = BertTokenizer.from_pretrained('./dataset/bertvocab/v3/bert-miccai18/', do_lower_case=True)
        answer_len = 20
        dataset_ver = 'complex1.2'
        val_seq = [1, 5, 16]
        folder_head = 'dataset/instruments18/seq_'
        folder_tail = '/vqa/'+dataset_ver+'/*.txt'
        
        val_dataset = SurgicalSentenceVQADataset(val_seq, folder_head, folder_tail, patch_size = args.patch_size)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size= 1, shuffle=False)

    elif args.dataset_type == 'c80':
        '''
        Train and test for cholec dataset
        '''
        if args.tokenizer_ver == 'v2':
            tokenizer = BertTokenizer.from_pretrained('./dataset/bertvocab/v2/bert-cholec80/')
        elif args.tokenizer_ver == 'v3':
            tokenizer = BertTokenizer.from_pretrained('./dataset/bertvocab/v3/bert-cholec80/', do_lower_case=True)
        answer_len = 20
        dataset_ver = 'complex2'
        val_seq = [5, 11, 12, 17, 19, 26, 27, 31]
        folder_head = 'dataset/cholec80/'+dataset_ver+'/'
        folder_tail = '/*.txt'

        val_dataset = SurgicalSentenceC80VQADataset(val_seq, folder_head, folder_tail, patch_size = args.patch_size)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size= 1, shuffle=False)

    elif args.dataset_type == 'med_vqa':
        '''
        Train and test for MED_VQA_S
        '''
        if args.tokenizer_ver == 'v2':
            tokenizer = BertTokenizer.from_pretrained('./dataset/bertvocab/v2/bert-medvqa/')
        elif args.tokenizer_ver == 'v3':
            tokenizer = BertTokenizer.from_pretrained('./dataset/bertvocab/v3/bert-medvqa/', do_lower_case=True)
        answer_len = 50
        val_folder = 'dataset/VQA-Med/ImageClef-2019-VQA-Med-Validation/'
        val_img_folder = 'Val_images/'
        val_QA = 'QAPairsByCategory/C4_Abnormality_val.txt'

        val_dataset = VQA_Med19Sentence(val_folder, val_img_folder, val_QA, patch_size = args.patch_size)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size= 1, shuffle=False)

    elif args.dataset_type == 'm18_vid':
        '''
        Train and test for miccai video dataset
        '''
        if args.tokenizer_ver == 'v2':
            tokenizer = BertTokenizer.from_pretrained('./dataset/bertvocab/v2/bert-miccai18/')
        elif args.tokenizer_ver == 'v3':
            tokenizer = BertTokenizer.from_pretrained('./dataset/bertvocab/v3/bert-miccai18/', do_lower_case=True)
        answer_len = 20
        dataset_ver = 'complex1.2'
        val_seq = [1, 5, 16]
        folder_head = 'dataset/instruments18/seq_'
        folder_tail = '/vqa/'+dataset_ver+'/*.txt'
        
        val_dataset = SurgicalSentenceVideoVQADataset(val_seq, folder_head, folder_tail, temporal_size = args.temporal_size, patch_size = args.patch_size)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size= 1, shuffle=False)
    
    elif args.dataset_type == 'c80_vid':
        '''
        Train and test for cholec video dataset
        '''
        if args.tokenizer_ver == 'v2':
            tokenizer = BertTokenizer.from_pretrained('./dataset/bertvocab/v2/bert-cholec80/')
        elif args.tokenizer_ver == 'v3':
            tokenizer = BertTokenizer.from_pretrained('./dataset/bertvocab/v3/bert-cholec80/', do_lower_case=True)
        answer_len = 20
        dataset_ver = 'complex2'
        val_seq = [5, 11, 12, 17, 19, 26, 27, 31]
        folder_head = 'dataset/cholec80/'+dataset_ver+'/'
        folder_tail = '/*.txt'

        val_dataset = SurgicalSentenceC80VideoVQADataset(val_seq, folder_head, folder_tail, temporal_size = args.temporal_size, patch_size = args.patch_size)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size= 1, shuffle=False)

    # fix randoms
    seed_everything()

    # cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    print(device)

    checkpoint = torch.load(args.checkpoint, map_location=str(device)) # + str(epoch) + ".pth.tar", map_location=str(device))
    model = checkpoint['model']
    model = model.to(device)
    # print(model)

    metrics = evaluate_transformer(args, val_dataloader, tokenizer, model, answer_len, len(tokenizer))
        # print("Epoch {} : BLEU-1 {:.6f} BLEU-2 {:.6f} BLEU-3 {:.6f} BLEU-4 {:.6f}".format(epoch, metrics["Bleu_1"],  metrics["Bleu_2"],  metrics["Bleu_3"],  metrics["Bleu_4"]))




     # DataLoader
    # transform = transforms.Compose([
    #                                 transforms.Resize((300,256)),
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    #                                 ])
    
    # # MICCAI18 dataset
    # val_seq = [1, 5, 16]
    # folder_head = 'dataset/instruments18/seq_'
    # folder_tail = '/vqa/complex/*.txt'
    # answer_len = 20
    # val_dataset = SurgicalSentenceVQADataset(val_seq, folder_head, folder_tail, transform=transform)
    # val_dataloader = DataLoader(dataset=val_dataset, batch_size= 1, shuffle=False)
    # tokenizer = BertTokenizer.from_pretrained('./dataset/bertvocab/v1/bert-miccai18/bert-miccai18-vocab.txt')

    # MEDVQA dataset
    # val_folder = 'dataset/VQA-Med/ImageClef-2019-VQA-Med-Validation/'
    # val_img_folder = 'Val_images/'
    # val_QA = 'QAPairsByCategory/C4_Abnormality_val.txt'
    # val_dataset = VQA_Med19Sentence(val_folder, val_img_folder, val_QA, patch_size = args.patch_size)
    # val_dataloader = DataLoader(dataset=val_dataset, batch_size= 1, shuffle=False)
    # answer_len = 50
    # tokenizer = BertTokenizer.from_pretrained('./dataset/bertvocab/v2/bert-medvqa/', do_lower_case=True)
    # vocab_size = len(tokenizer)
    # print(vocab_size)