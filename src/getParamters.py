import argparse


def getParamters():
    parser = argparse.ArgumentParser(description='Process some integers.')
    
    parser.add_argument('--method', '-m', metavar='METHOD', type=str, help='the name of the method you want to use', default="metis")
    parser.add_argument('--method_aux', '-a', metavar='METHOD', type=str, help='The name of the auxiliar method you want to use', default="metis")
    
    parser.add_argument('--working_dataset', '-w', metavar='DATASET', type=str, help='the dataset you want to use [dev2013, test2013, dev2014]', default="dev2013")

    parser.add_argument('--minmax_rank', '-r', metavar='N', type=int, help='The threshold for min max method', default=50)

    parser.add_argument('--use_filter', '-f', action='store_const', help='Whether to use filter or not', default=False, const=True)
    parser.add_argument('--supervised_filter', '-s', action='store_const', help='Whether to use supervised filter or not', const=True, default=False)
    parser.add_argument('--supervised_dataset', '-d', metavar='DT',  action='append', help='A list of datasets to use for training a ML model for filtering (use it multiple times if you want multiple datasets)', default=[])

    parser.add_argument('--metric', '-b', metavar='METRIC', action='append', help='Similarity metric', default=[])
    #parser.add_argument('--visual', '-v', metavar='FEAT', type=str, action='append', help='Visual feature to use', default=[], required=True)
    parser.add_argument('--visual', '-v', metavar='FEAT', type=str, action='append', help='Visual feature to use', default=[])
    parser.add_argument('--others', '-k', metavar='FEAT', type=str, action='append', help='Other features such as latitude, nbcomments', default=[])
    
    parser.add_argument('--useGT', '-u', action='store_const', help='Use it if you want to ignore the any Ground Truth data', const=True, default=False)
    parser.add_argument('--disableSelfLooping', '-l', action='store_const', help='Use this flag to prevent self loopings in the similarity graph', const=True, default=False)
    
    parser.add_argument('--outfile', '-o', metavar='FILE', type=str, help='Output filename', default='outfile.txt')
    parser.add_argument('--evalcsv', '-e', metavar='FILE', type=str, help='CSV file containing the results', default='eval.csv')
    parser.add_argument('--use_credibility', '-c', action='store_const', help='Whether to use credibility scores or not', default=False, const=True)
    parser.add_argument('--apply_new_ranking', '-n', metavar="FILE", type=str, help='Some other Run in which the rank should be used', default="")
    
    parser.add_argument('--pmin', metavar="VAL", type=float, help='Parameter for min', default="1")
    parser.add_argument('--pmean', metavar="VAL", type=float, help='Parameter for mean', default="2")
    parser.add_argument('--pmax', metavar="VAL", type=float, help='Parameter for max', default="5")
    
    parser.add_argument('--pimin', metavar="VAL", type=float, help='Parameter for max', default="0.1")
    parser.add_argument('--pimean', metavar="VAL", type=float, help='Parameter for max', default="0.1")
    parser.add_argument('--pimax', metavar="VAL", type=float, help='Parameter for max', default="0.1")
    
    parser.add_argument('--textsim', '-t', action='store_const', help='Whether to use text similarity or not', default=False, const=True)
    args = parser.parse_args()

    print "Method -", args.method
    print "Similarity Metrics -", args.metric
    print "Working Dataset -", args.working_dataset
    
    print "Use Filter -", args.use_filter
    print "Supervised Filter -", args.supervised_filter
    print "Supervised datasets -", args.supervised_dataset
    
    print "Visual Descriptions -", args.visual
    print "Other descriptions -", args.others
    print "Outfile -", args.outfile
    return args

