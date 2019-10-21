import random
import os
import argparse

def Config():
    parser = argparse.ArgumentParser()
    #Seed for random initializations
    parser.add_argument("--seed", default=123, type=int, required=False, dest='SEED', help="random seed to initialize the network")

    #Data settings
    parser.add_argument("--lang", default='en', type=str, required=False, choices=['en', 'it', 'de'], dest='LANG',help="dataset language to be trained for (en (or) it (or) de)")
    
    # RAW data location
    parser.add_argument('--raw_data', default='./data/', type=str, required=False, dest='RAW_DATA_DIR', help="location of raw data downloaded for WoZ2.0")

    #location to store preprocessed data
    parser.add_argument('--data_dir', default='./data/', type=str, required=False, dest='DATA_DIR', help='location of preprocessed dataset')

    #Embedding details
    parser.add_argument('--use_pt', default=False, type=str, required=False, dest='USE_PRETRAINED_EMB', help='True to use pre-trained embedding')

    parser.add_argument('--train_emb', default=True, type=str, required=False, dest='TRAIN_EMBEDDING', help='True to train embedding or False to freeze embedding')
    parser.add_argument('--emb_dim', default=128, type=int, required=False, dest='EMBEDDING_DIM', help='embedding dimension')

    #Model details
    parser.add_argument('--rnn', default='lstm', type=str, required=False, choices=['lstm', 'gru'], dest='RNN', help='rnn type to use (lstm ot gru)')
    parser.add_argument('--rnn_dim', default=64, type=int, required=False, dest='HIDDEM_DIM', help='hidden dimension of rnn')
    parser.add_argument('--sh_enc', default=True, type=str, required=False, dest='SHARED_ENCODER', help='True for shared encoder for all slots')

    #Training details
    parser.add_argument('--batch', default=32, type=int, required=False, dest='BATCH_SIZE', help='Batch size to train and evaluate')
    parser.add_argument('--max_epoch', default=150, type=int, required=False, dest='MAX_EPOCH', help='Maximum epoxh of the model')
    parser.add_argument('--lr', default=0.001, type=int, required=False, dest='LEARNING_RATE', help='learning rate of the model') 
    parser.add_argument('--metric', default='joint_goal', type=str, required=False, dest='METRIC_TO_EVALUATE', help='metric to judge the best model') 
    parser.add_argument('--p', default=0.5, type=int, required=False, dest='PROB_THRESHOLD', help='probability threshold for the requested slot prediction') 
    parser.add_argument('--save_dir', default='./exp/', type=str, required=False, dest='SAVE_DIR', help='Directory to save the trained models') 

    args = parser.parse_args()

    #choose none_token based on the language
    NONE_TOKEN = {'en': 'none', 'it': 'nessuno', 'de': 'keine'}
    args.NONE_TOKEN = NONE_TOKEN[args.LANG]

    # args.SEED = get_seed()
    args.DATA_DIR = os.path.join(args.DATA_DIR, args.LANG, 'preprocessed')
    args.SAVE_DIR = os.path.join(args.SAVE_DIR, args.LANG, str(args.SEED))

    if args.USE_PRETRAINED_EMB:
        args.PRETRAINED_EMB = os.path.join(args.DATA_DIR, 'preprocessed/emb.json')

    return args

# def get_seed():
#     seed = random.randint(1,10000)
#     if os.path.isdir(os.path.join(args.SAVE_DIR, args.LANG, str(args.SEED))):
#         seed = get_seed()
#     return seed