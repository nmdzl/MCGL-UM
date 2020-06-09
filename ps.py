
import argparse

# Training settings
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--acc_file', type=str, default='acc/sample.csv',
                        help='the file to save the accuracy')
    parser.add_argument('--baseline', type=int, default=1,
                        help='Choose GCN baseline. 1 refers to GCN. 2 refers to GCN*.')
    parser.add_argument('--batch_decay', type=float, default=600,
                        help='the decay of batch size -- in MC_co_decay file')
    parser.add_argument('--batch_min', type=float, default=0.85,
                        help='the minimum ratio of the batch size dropout -- in MC_co_decay file')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Set the batch size of MCGL')
    parser.add_argument('--begin_ite', type=int, default=10,
                        help='the iteration that begin to change graph structure -- in MC_cotraining file')
    parser.add_argument('--dataset', type=str, default='cora',
                        help='Choose from {cora, citeseer, pubmed, ms_academic}')
    parser.add_argument('--decay_pro', type=float, default=0.1,
                        help='the probability decay for pair')
    parser.add_argument('--decay_rate', type=float, default=0.2,
                        help='the minimum ratio of the batch size dropout -- in MC_cotraining file')
    parser.add_argument('--degree', type=float, default=0,
                       help='The neighbor number degree')
    parser.add_argument('--depth', type=int, default=2,
                        help='the depth of GCN model')
    parser.add_argument('--dropout', type=float, default=0.7,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--epochs', type=int, default=3000,
                        help='Number of epochs to train.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--hard_thre', type=float, default=0.8,
                        help='when the confidence of a node classified to class-i, then give it hard label -- in sftn file')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--iterations', type=int, default=10000,
                        help='Number of iterations to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--MCTSdep', type=str, default=3,
                        help='the tree depth when change graph structure')
    parser.add_argument('--model', type=str, default='MLP',
                        help='Set training model. Default is MLP.')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disable CUDA training.')
    parser.add_argument('--noise_rate', type=float, default=1.0,
                        help='Reduce the noise rate to some point. Set it as 1.0 to keep the original noise rate.')
    parser.add_argument('--opt_batch', type=int, default=100,
                        help='the batch when begin to optimize graph structure -- MCTS')
    parser.add_argument('--patience', type=int, default=100, help='Patience')
    parser.add_argument('--repeat_split', type=int, default=10,
                        help='the number of the dataset split')
    parser.add_argument('--sam_min_r', type=float, default=0.0002,
                        help='when the time of a node sampled is less than sample_min_rate, then drop it -- in sftn file')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--total_sample', type=int, default=400000,
                        help='the total sample number -- SFTN')
    parser.add_argument('--trdep', type=int, default=2,
                        help='Set the depth of sampling tree when training')
    parser.add_argument('--tsdep', type=int, default=2,
                        help='Set the depth of inference tree when testing')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='Set weight decay (L2 loss on parameters).')
    return parser.parse_args()
