'''
Description     : Train classification model.
Paper           : Surgical-VQA: Visual Question Answering in Surgical Scenes Using Transformers
Author          : Lalithkumar Seenivasan, Mobarakol Islam, Adithya Krishna, Hongliang Ren
Lab             : MMLAB, National University of Singapore
'''

import os
import argparse
import pandas as pd
from lib2to3.pytree import convert

from torch import nn
from torch import optim
import torch.utils.data
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from transformers import BertTokenizer
from torch.utils.data  import DataLoader

from utils import *
from dataloaders.dataloaderClassification import *
from models.VisualBertClassification import VisualBertClassification
from models.VisualBertResMLPClassification import VisualBertResMLPClassification

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


'''
Seed randoms
'''
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


def train(args, train_dataloader, model, criterion, optimizer, epoch, tokenizer, device):
    
    model.train()
    
    total_loss = 0.0    
    label_true = None
    label_pred = None
    label_score = None
    
    
    for i, (_, visual_features, q, labels) in enumerate(train_dataloader,0):

        # prepare questions
        questions = []
        for question in q: questions.append(question)
        inputs = tokenizer(questions, return_tensors="pt", padding="max_length", max_length=args.question_len)

    
        # GPU / CPU
        visual_features = visual_features.to(device)
        labels = labels.to(device)
                
        outputs = model(inputs, visual_features)
        loss = criterion(outputs, labels)

        # zero the parameter gradients
        optimizer.zero_grad()        
        loss.backward()
        optimizer.step()

        # print statistics
        total_loss += loss.item()
        
        scores, predicted = torch.max(F.softmax(outputs, dim=1).data, 1)    
        label_true = labels.data.cpu() if label_true == None else torch.cat((label_true, labels.data.cpu()), 0)
        label_pred = predicted.data.cpu() if label_pred == None else torch.cat((label_pred, predicted.data.cpu()), 0)
        label_score = scores.data.cpu() if label_score == None else torch.cat((label_score, scores.data.cpu()), 0)

    # loss and acc
    acc, c_acc = calc_acc(label_true, label_pred), calc_classwise_acc(label_true, label_pred)
    precision, recall, fscore = calc_precision_recall_fscore(label_true, label_pred)
    print('Train: epoch: %d loss: %.6f | Acc: %.6f | Precision: %.6f | Recall: %.6f | FScore: %.6f' %(epoch, total_loss, acc, precision, recall, fscore))
    return acc


def validate(args, val_loader, model, criterion, epoch, tokenizer, device, save_output = False):
    
    model.eval()

    total_loss = 0.0    
    label_true = None
    label_pred = None
    label_score = None
    file_names = list()
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for i, (file_name, visual_features, q, labels) in enumerate(val_loader,0):
            # prepare questions
            questions = []
            for question in q: questions.append(question)
            inputs = tokenizer(questions, return_tensors="pt", padding="max_length", max_length=args.question_len)

            # GPU / CPU
            visual_features = visual_features.to(device)
            labels = labels.to(device)
                    
            outputs = model(inputs, visual_features)
            loss = criterion(outputs,labels)

            total_loss += loss.item()
        
            scores, predicted = torch.max(F.softmax(outputs, dim=1).data, 1)    
            label_true = labels.data.cpu() if label_true == None else torch.cat((label_true, labels.data.cpu()), 0)
            label_pred = predicted.data.cpu() if label_pred == None else torch.cat((label_pred, predicted.data.cpu()), 0)
            label_score = scores.data.cpu() if label_score == None else torch.cat((label_score, scores.data.cpu()), 0)
            for f in file_name: file_names.append(f)
            
    acc = calc_acc(label_true, label_pred) 
    c_acc = 0.0
    # c_acc = calc_classwise_acc(label_true, label_pred)
    precision, recall, fscore = calc_precision_recall_fscore(label_true, label_pred)

    print('Test: epoch: %d loss: %.6f | Acc: %.6f | Precision: %.6f | Recall: %.6f | FScore: %.6f' %(epoch, total_loss, acc, precision, recall, fscore))

    if save_output:
        '''
            Saving predictions
        '''
        if os.path.exists(args.checkpoint_dir + 'text_files') == False:
            os.mkdir(args.checkpoint_dir + 'text_files' ) 
        file1 = open(args.checkpoint_dir + 'text_files/labels.txt', 'w')
        file1.write(str(label_true))
        file1.close()

        file1 = open(args.checkpoint_dir + 'text_files/predictions.txt', 'w')
        file1.write(str(label_pred))
        file1.close()

        if args.dataset_type == 'med_vqa':
            if args.dataset_cat == 'cat1': 
                convert_arr = ['cta - ct angiography', 'no', 'us - ultrasound', 'xr - plain film', 'noncontrast', 'yes', 't2', 'ct w/contrast (iv)', 'mr - flair', 'mammograph', 'ct with iv contrast', 
                            'gi and iv', 't1', 'mr - t2 weighted', 'mr - t1w w/gadolinium', 'contrast', 'iv', 'an - angiogram', 'mra - mr angiography/venography', 'nm - nuclear medicine', 'mr - dwi diffusion weighted', 
                            'ct - gi & iv contrast', 'ct noncontrast', 'mr - other pulse seq.', 'ct with gi and iv contrast', 'flair', 'mr - t1w w/gd (fat suppressed)', 'ugi - upper gi', 'mr - adc map (app diff coeff)', 
                            'bas - barium swallow', 'pet - positron emission', 'mr - pdw proton density', 'mr - t1w - noncontrast', 'be - barium enema', 'us-d - doppler ultrasound', 'mr - stir', 'mr - flair w/gd', 
                            'ct with gi contrast', 'venogram', 'mr t2* gradient,gre,mpgr,swan,swi', 'mr - fiesta', 'ct - myelogram', 'gi', 'sbft - small bowel', 'pet-ct fusion']
            elif args.dataset_cat == 'cat2':
                convert_arr = ['axial', 'longitudinal', 'coronal', 'lateral', 'ap', 'sagittal', 'mammo - mlo', 'pa', 'mammo - cc', 'transverse', 'mammo - mag cc', 'frontal', 'oblique', '3d reconstruction', 'decubitus', 'mammo - xcc']
            else:
                convert_arr = ['lung, mediastinum, pleura', 'skull and contents', 'genitourinary', 'spine and contents', 'musculoskeletal', 'heart and great vessels', 'vascular and lymphatic', 'gastrointestinal', 'face, sinuses, and neck', 'breast']
        elif args.dataset_type == 'c80':
            convert_arr = ['no', 'calot triangle dissection', 'yes', '1', '2', 'gallbladder dissection', 
                            'clipping cutting', 'gallbladder retraction', '0', 'cleaning coagulation', 
                            'gallbladder packaging', 'preparation', '3']
        elif args.dataset_type == 'm18':
            convert_arr = ['kidney', 'Idle', 'Grasping', 'Retraction', 'Tissue_Manipulation',
                            'Tool_Manipulation', 'Cutting', 'Cauterization', 'Suction', 
                            'Looping', 'Suturing', 'Clipping', 'Staple', 'Ultrasound_Sensing',
                            'left-top', 'right-top', 'left-bottom', 'right-bottom']

        df = pd.DataFrame(columns=["Img", "Ground Truth", "Prediction"])
        for i in range(len(label_true)):
            df = df.append({'Img': file_names[i], 'Ground Truth': convert_arr[label_true[i]], 'Prediction': convert_arr[label_pred[i]]}, ignore_index=True)
        
        df.to_csv(args.checkpoint_dir + args.checkpoint_dir.split('/')[1] + '_' + args.checkpoint_dir.split('/')[2] + '_eval.csv')
    
    return (acc, c_acc, precision, recall, fscore)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='VisualQuestionAnswerClassification')
    
    # Model parameters
    parser.add_argument('--emb_dim',        type=int,   default=300,                                help='dimension of word embeddings.')
    parser.add_argument('--n_heads',        type=int,   default=8,                                  help='Multi-head attention.')
    parser.add_argument('--dropout',        type=float, default=0.1,                                help='dropout')
    parser.add_argument('--encoder_layers', type=int,   default=6,                                  help='the number of layers of encoder in Transformer.')
    
    # Training parameters
    parser.add_argument('--epochs',         type=int,   default=80,                                 help='number of epochs to train for (if early stopping is not triggered).') #80, 26
    parser.add_argument('--batch_size',     type=int,   default=64,                                 help='batch_size')
    parser.add_argument('--workers',        type=int,   default=1,                                  help='for data-loading; right now, only 1 works with h5pys.')
    parser.add_argument('--print_freq',     type=int,   default=100,                                help='print training/validation stats every __ batches.')
    
    # existing checkpoint
    parser.add_argument('--checkpoint',     default=None,                                           help='path to checkpoint, None if none.')
    
    parser.add_argument('--lr',             type=float, default=0.00001,                            help='0.000005, 0.00001, 0.000005')
    parser.add_argument('--checkpoint_dir', default= 'checkpoints/clf_v1_2_5x5/m18f3/',    help='med_vqa_c$version$/m18/c80//m18_vid$temporal_size$/c80_vid$temporal_size$') #clf_v1_2_1x1/med_vqa_c3
    parser.add_argument('--dataset_type',   default= 'm18',                                     help='med_vqa/m18/c80/m18_vid/c80_vid')
    parser.add_argument('--dataset_cat',    default= 'None',                                        help='cat1/cat2/cat3')
    parser.add_argument('--transformer_ver',default= 'vbrm',                                        help='vb/vbrm')
    parser.add_argument('--tokenizer_ver',  default= 'v2',                                          help='v2/v3')
    parser.add_argument('--patch_size',     default= 5,                                             help='1/2/3/4/5')
    parser.add_argument('--temporal_size',  default= 3,                                             help='1/2/3/4/5')
    parser.add_argument('--question_len',   default= 25,                                            help='25')
    parser.add_argument('--num_class',      default= 2,                                             help='25')
    parser.add_argument('--validate',       default=False,                                          help='When only validation required False/True')
    args = parser.parse_args()

    # load checkpoint, these parameters can't be modified
    final_args = {"emb_dim": args.emb_dim, "n_heads": args.n_heads, "dropout": args.dropout, "encoder_layers": args.encoder_layers}
    
    seed_everything()
    
    # GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
    cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
    print('device =', device)

    # best model initialize
    start_epoch = 1
    best_epoch = [0]
    best_results = [0.0]
    epochs_since_improvement = 0

    # dataset
    if args.dataset_type == 'med_vqa':
        '''
        Train and test dataloader for MED_VQA
        '''
        # tokenizer
        tokenizer = None
        if args.tokenizer_ver == 'v2': tokenizer = BertTokenizer.from_pretrained('./dataset/bertvocab/v2/bert-medvqa/')
        elif args.tokenizer_ver == 'v3': tokenizer = BertTokenizer.from_pretrained('./dataset/bertvocab/v3/bert-medvqa/', do_lower_case=True)
        
        # data location
        train_folder = 'dataset/VQA-Med/ImageClef-2019-VQA-Med-Training/'
        val_folder = 'dataset/VQA-Med/ImageClef-2019-VQA-Med-Validation/'
        train_img_folder = 'Train_images/'
        val_img_folder = 'Val_images/'

        # dataloader
        train_dataset = MedVQAClassification(train_folder, train_img_folder, args.dataset_cat, patch_size = args.patch_size, validation=False)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size= args.batch_size, shuffle=True)
        val_dataset = MedVQAClassification(val_folder, val_img_folder, args.dataset_cat, patch_size = args.patch_size, validation=True)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size= args.batch_size, shuffle=False)

        # num_classes
        if args.dataset_cat == 'cat1': args.num_class = 45
        elif args.dataset_cat == 'cat2': args.num_class = 16
        elif args.dataset_cat == 'cat3': args.num_class = 10

    elif args.dataset_type == 'm18':
        '''
        Train and test dataloader for EndoVis18
        '''
        # tokenizer
        tokenizer = None
        if args.tokenizer_ver == 'v2': tokenizer = BertTokenizer.from_pretrained('./dataset/bertvocab/v2/bert-miccai18/')
        elif args.tokenizer_ver == 'v3': tokenizer = BertTokenizer.from_pretrained('./dataset/bertvocab/v3/bert-miccai18/', do_lower_case=True)
        
        # data location
        train_seq = [2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15]
        val_seq = [1, 5, 16]
        # train_seq = [1, 2, 3, 5, 6, 7, 9, 10, 14, 15, 16]
        # val_seq = [4, 11, 12]
        folder_head = 'dataset/instruments18/seq_'
        folder_tail = '/vqa/simple/*.txt'
        
        # dataloader
        train_dataset = EndoVis18VQAClassification(train_seq, folder_head, folder_tail, patch_size = args.patch_size)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size= args.batch_size, shuffle=True)
        val_dataset = EndoVis18VQAClassification(val_seq, folder_head, folder_tail, patch_size = args.patch_size)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size= args.batch_size, shuffle=False)

        # num_classes
        args.num_class = 18

    elif args.dataset_type == 'm18_vid':
        '''
        Train and test dataloader for EndoVis18 temporal
        '''
        # tokenizer
        tokenizer = None
        if args.tokenizer_ver == 'v2': tokenizer = BertTokenizer.from_pretrained('./dataset/bertvocab/v2/bert-miccai18/')
        elif args.tokenizer_ver == 'v3': tokenizer = BertTokenizer.from_pretrained('./dataset/bertvocab/v3/bert-miccai18/', do_lower_case=True)
        
        # data location
        train_seq = [2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15]
        val_seq = [1, 5, 16]
        folder_head = 'dataset/instruments18/seq_'
        folder_tail = '/vqa/simple/*.txt'
        
        # dataloader
        train_dataset = EndoVis18VidVQAClassification(train_seq, folder_head, folder_tail, patch_size = args.patch_size, temporal_size=args.temporal_size)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size= args.batch_size, shuffle=True)
        val_dataset = EndoVis18VidVQAClassification(val_seq, folder_head, folder_tail, patch_size = args.patch_size, temporal_size=args.temporal_size)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size= args.batch_size, shuffle=False)

        # num_classes
        args.num_class = 18

    elif args.dataset_type == 'c80':
        '''
        Train and test for cholec dataset
        '''
        # tokenizer
        if args.tokenizer_ver == 'v2': tokenizer = BertTokenizer.from_pretrained('./dataset/bertvocab/v2/bert-cholec80/')
        elif args.tokenizer_ver == 'v3': tokenizer = BertTokenizer.from_pretrained('./dataset/bertvocab/v3/bert-cholec80/', do_lower_case=True)
        
        # dataloader
        train_seq = [1, 2, 3, 4, 6, 7, 8, 9, 10, 13, 14, 15, 16, 18, 20, 21, 22, 23, 24, 25, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40]
        val_seq = [5, 11, 12, 17, 19, 26, 27, 31]
        folder_head = 'dataset/cholec80/simple2/'
        folder_tail = '/*.txt'

        # dataloader
        train_dataset = Cholec80VQAClassification(train_seq, folder_head, folder_tail, patch_size = args.patch_size)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size= args.batch_size, shuffle=True)
        val_dataset = Cholec80VQAClassification(val_seq, folder_head, folder_tail, patch_size = args.patch_size)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size= args.batch_size, shuffle=False)

        # num_classes
        args.num_class = 13

    elif args.dataset_type == 'c80_vid':
        '''
        Train and test dataloader for c80 temporal
        '''
        # tokenizer
        tokenizer = None
        if args.tokenizer_ver == 'v2': tokenizer = BertTokenizer.from_pretrained('./dataset/bertvocab/v2/bert-cholec80/')
        elif args.tokenizer_ver == 'v3': tokenizer = BertTokenizer.from_pretrained('./dataset/bertvocab/v3/bert-cholec80/', do_lower_case=True)
        
        # data location
        train_seq = [1, 2, 3, 4, 6, 7, 8, 9, 10, 13, 14, 15, 16, 18, 20, 21, 22, 23, 24, 25, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40]
        val_seq = [5, 11, 12, 17, 19, 26, 27, 31]
        folder_head = 'dataset/cholec80/simple2/'
        folder_tail = '/*.txt'
        
        # dataloader
        train_dataset = Cholec80VQAVidClassification(train_seq, folder_head, folder_tail, patch_size = args.patch_size, temporal_size=args.temporal_size)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size= args.batch_size, shuffle=True)
        val_dataset = Cholec80VQAVidClassification(val_seq, folder_head, folder_tail, patch_size = args.patch_size, temporal_size=args.temporal_size)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size= args.batch_size, shuffle=False)

        # num_classes
        args.num_class = 13
    

    # Initialize / load checkpoint
    if args.checkpoint is None:
        # model
        if args.transformer_ver == 'vb':
            model = VisualBertClassification(vocab_size=len(tokenizer), layers=args.encoder_layers, n_heads=args.n_heads, num_class = args.num_class)
        elif args.transformer_ver == 'vbrm':
            model = VisualBertResMLPClassification(vocab_size=len(tokenizer), layers=args.encoder_layers, n_heads=args.n_heads, num_class = args.num_class, token_size = int(args.question_len+(args.patch_size * args.patch_size)))
        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    else:
        checkpoint = torch.load(args.checkpoint, map_location=str(device))
        start_epoch = checkpoint['epoch']
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_Acc = checkpoint['Acc']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        final_args = checkpoint['final_args']
        for key in final_args.keys(): args.__setattr__(key, final_args[key])


    # Move to GPU, if available
    model = model.to(device)
    print(final_args)    
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('model params: ', pytorch_total_params)
    # print(model)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # validation
    if args.validate:
        test_acc, test_c_acc, test_precision, test_recall, test_fscore = validate(args, val_loader=val_dataloader, model = model, criterion=criterion, epoch=(args.epochs-1), tokenizer = tokenizer, device = device)
    else:     
        for epoch in range(start_epoch, args.epochs):

            if epochs_since_improvement > 0 and epochs_since_improvement % 5 == 0:
                adjust_learning_rate(optimizer, 0.8)
            
            # train
            train_acc = train(args, train_dataloader=train_dataloader, model = model, criterion=criterion, optimizer=optimizer, epoch=epoch, tokenizer = tokenizer, device = device)

            # validation
            test_acc, test_c_acc, test_precision, test_recall, test_fscore = validate(args, val_loader=val_dataloader, model = model, criterion=criterion, epoch=epoch, tokenizer = tokenizer, device = device)
            
            if test_acc >= best_results[0]:
                epochs_since_improvement = 0
                
                best_results[0] = test_acc
                best_epoch[0] = epoch
                # print('Best epoch: %d | Best acc: %.6f' %(best_epoch[0], best_results[0]))
                save_clf_checkpoint(args.checkpoint_dir, epoch, epochs_since_improvement, model, optimizer, best_results[0], final_args)
                        
            else:
                epochs_since_improvement += 1
                print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
            
            if train_acc >= 1.0: break
