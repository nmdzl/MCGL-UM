import os
import sys
import argparse

def add_attr(attr, dataset, hidden, weight_decay, lr=0.005, dropout=0.5, batch_size=50):
    attr[dataset] = {}
    attr[dataset]['h'] = hidden
    attr[dataset]['w'] = weight_decay
    attr[dataset]['b'] = batch_size
    attr[dataset]['l'] = lr
    attr[dataset]['d'] = dropout
    return attr

attr_MCGL = {'name' : 'train_MC_base.py', 'baseline' : 1}
add_attr(attr_MCGL, 'cora', 32, 0.001, 0.005, 0.5, 50)
add_attr(attr_MCGL, 'citeseer', 64, 0.001, 0.005, 0.3, 200)
add_attr(attr_MCGL, 'pubmed', 32, 0.001, 0.005, 0.5, 50)
add_attr(attr_MCGL, 'ms_academic', 128, 0.0001, 0.005, 0.5, 200)

attr_GCN_1 = {'name' : 'train_GCN.py', 'baseline' : 1}
add_attr(attr_GCN_1, 'cora', 32, 0.0005, 0.01, 0.7)
add_attr(attr_GCN_1, 'citeseer', 64, 0.001, 0.05, 0.4)
add_attr(attr_GCN_1, 'pubmed', 32, 0.0005, 0.005, 0.5)
add_attr(attr_GCN_1, 'ms_academic', 128, 0.0005, 0.005, 0.7)

attr_GCN_2 = {'name' : 'train_GCN.py', 'baseline' : 2}
add_attr(attr_GCN_2, 'cora', 32, 0.0005, 0.005, 0.7)
add_attr(attr_GCN_2, 'citeseer', 64, 0.001, 0.05, 0.6)
add_attr(attr_GCN_2, 'pubmed', 32, 0.0005, 0.05, 0.3)
add_attr(attr_GCN_2, 'ms_academic', 128, 0.0005, 0.01, 0.6)

model_attr = {'GCN': attr_GCN_1, 'GCN*': attr_GCN_2, 'MCGL-UM': attr_MCGL}

def search():
    for model in args.model:
        if model in model_attr.keys():
            for d in args.dataset:
                try:
                    for h in args.hidden:
                        for w in args.weight_decay:
                            for l in args.lr:
                                for dp in args.dropout:
                                    for b in args.batch_size:
                                        for n in args.noise_rate:
                                            file = open(args.acc_file, "a")
                                            file.write("\n{} | dataset={} hidden={} weight_decay={} lr={} dropout={} batch_size={} noise_rate={},".format(
                                                model, d,
                                                h if h is not None else model_attr[model][d]['h'],
                                                w if w is not None else model_attr[model][d]['w'],
                                                l if l is not None else model_attr[model][d]['l'],
                                                dp if dp is not None else model_attr[model][d]['d'],
                                                b if b is not None else model_attr[model][d]['b'],
                                                n))
                                            file.close()
                                            for s in seeds:
                                                # ensure the correct command to run python script on your operating system
                                                os.system("python {} --acc_file {} --baseline {} --dataset {} --hidden {} --weight_decay {} --lr {} --dropout {} --batch_size {} --noise_rate {} --seed {}"
                                                          .format(model_attr[model]['name'], args.acc_file,
                                                                  model_attr[model]['baseline'], d,
                                                                  h if h is not None else model_attr[model][d]['h'],
                                                                  w if w is not None else model_attr[model][d]['w'],
                                                                  l if l is not None else model_attr[model][d]['l'],
                                                                  dp if dp is not None else model_attr[model][d]['d'],
                                                                  b if b is not None else model_attr[model][d]['b'],
                                                                  n, s))
                except KeyError:
                    sys.stderr("error: invalid dataset: {}".format(d))
        else:
            sys.stderr("error: invalid model: {}".format(model))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='sample fine-tuning script')
    parser.add_argument('--acc_file', type=str, default='acc/sample_fine_tuning.csv',
                        help='the file to save the accuracy')
    parser.add_argument('--model', type=str, default=['GCN', 'GCN*', 'MCGL'],
                        help=r'the search space of model. Choose from {GCN, GCN*, MCGL}')
    parser.add_argument('--dataset', type=str, nargs='*', default=['cora', 'citeseer', 'pubmed', 'ms_academic'],
                        help='the search space of dataset. Choose from {cora, citeseer, pubmed, ms_academic}')
    parser.add_argument('--hidden', type=int, nargs='*', default=[None],
                        help='the search space of hidden units')
    parser.add_argument('--weight_decay', type=float, nargs='*', default=[None],
                        help='the search space of weight_decay')
    parser.add_argument('--lr', type=float, nargs='*', default=[None],
                        help='the search space of learning rate')
    parser.add_argument('--dropout', type=float, nargs='*', default=[None],
                        help='the search space of dropout rate')
    parser.add_argument('--batch_size', type=int, nargs='*', default=[None],
                        help='the search space of batch_size of MCGL-UM')
    parser.add_argument('--noise_rate', type=float, nargs='*', default=[1.0],
                        help='Iterate the noise rate. Set as 1.0 to keep the original noise rate.')
    parser.add_argument('--s', type=int, default=10,
                        help='Iterate the seed from 0 to s-1')
    args = parser.parse_args(sys.argv[1:])

    if args.s <= 0:
        sys.exit("error: invalid seed range: {}".format(args.s))
    seeds = range(args.s)

    file = open(args.acc_file, "w")
    file.write("seed," + ','.join([str(s) for s in seeds]))
    file.close()
    search()