import os
import sys
import argparse

def add_attr(attr, dataset, hidden, weight_decay, lr, dropout, batch_size=50):
    attr[dataset] = {}
    attr[dataset]['hidden'] = hidden
    attr[dataset]['weight_decay'] = weight_decay
    attr[dataset]['lr'] = lr
    attr[dataset]['dropout'] = dropout
    attr[dataset]['batch_size'] = batch_size
    return attr

# fill in the best hyper-parameters of MCGL-UM
attr_MCGL = {'script' : 'train_MC_base.py', 'baseline' : 1}
add_attr(attr_MCGL, 'cora', 32, 0.001, 0.005, 0.5, 50)
add_attr(attr_MCGL, 'citeseer', 64, 0.001, 0.005, 0.3, 200)
add_attr(attr_MCGL, 'pubmed', 32, 0.001, 0.005, 0.5, 50)
add_attr(attr_MCGL, 'ms_academic', 128, 0.0001, 0.005, 0.5, 200)

# fill in the best hyper-parameters of GCN
attr_GCN_1 = {'script' : 'train_GCN.py', 'baseline' : 1}
add_attr(attr_GCN_1, 'cora', 32, 0.0005, 0.005, 0.7)
add_attr(attr_GCN_1, 'citeseer', 64, 0.001, 0.05, 0.6)
add_attr(attr_GCN_1, 'pubmed', 32, 0.0005, 0.05, 0.3)
add_attr(attr_GCN_1, 'ms_academic', 128, 0.0005, 0.01, 0.6)

# fill in the best hyper-parameters of GCN*
attr_GCN_2 = {'script' : 'train_GCN.py', 'baseline' : 2}
add_attr(attr_GCN_2, 'cora', 32, 0.0005, 0.01, 0.7)
add_attr(attr_GCN_2, 'citeseer', 64, 0.001, 0.05, 0.4)
add_attr(attr_GCN_2, 'pubmed', 32, 0.0005, 0.005, 0.5)
add_attr(attr_GCN_2, 'ms_academic', 128, 0.0005, 0.005, 0.7)

model_attr = {'GCN': attr_GCN_1, 'GCN*': attr_GCN_2, 'MCGL-UM': attr_MCGL}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='GCN',
                        help='Choose from {GCN, GCN*, MCGL-UM}')
    parser.add_argument('--acc_file', type=str, default='acc/train_best.csv',
                        help='the file to save the accuracy')
    parser.add_argument('--dataset', type=str, default='cora',
                        help='Choose from {cora, citeseer, pubmed, ms_academic}')
    parser.add_argument('--depth', type=int, default=2,
                        help='the depth of GCN model')
    parser.add_argument('--epochs', type=int, default=3000,
                        help='Number of epochs to train.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--iterations', type=int, default=10000,
                        help='Number of iterations to train.')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Disable CUDA training.')
    parser.add_argument('--noise_rate', type=float, default=1.0,
                        help='Reduce the noise rate to some point. Set it as 1.0 to keep the original noise rate.')
    parser.add_argument('--patience', type=int, default=100, help='Patience')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--trdep', type=int, default=2,
                        help='Set the depth of sampling tree when training')
    parser.add_argument('--tsdep', type=int, default=2,
                        help='Set the depth of inference tree when testing')
    args = parser.parse_args(sys.argv[1:])
    if not (args.model in model_attr.keys()):
        sys.exit("error: invalid model: {}".format(args.model))
    if not (args.dataset in model_attr[args.model].keys()):
        sys.exit("error: invalid dataset: {}".format(args.dataset))

    # ensure the correct command to run python script on your operating system
    os.system("python {} --baseline {}".format(model_attr[args.model]['script'], model_attr[args.model]['baseline'])
              + "".join([' --{} {}'.format(key, value) for (key, value) in model_attr[args.model][args.dataset].items()])
              + "".join([(' --{} {}'.format(key, value) if type(value) != bool else ' --{}'.format(key) if value else '')
                         for (key, value) in list(vars(args).items())[1:-1]]))
