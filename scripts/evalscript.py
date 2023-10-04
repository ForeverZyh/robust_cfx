
import argparse
import os

'''
Given a trained model with the name ModelDatasetCFXi.pt for i in [0,9] 
and a saved CFX file with the name ModelDatasetCFXi for i in [0,9], test
whether the model is epsilon-robust for the CFX for various epsilon and 
bias_epsilon values.

Saves the output in files logs/modelname_j.txt where modelname is 
ModelDatasetCFXi and j in [0,3] corresponds to the choice of 
epsilon and bias_epsilon.
'''

def main(args):
    if args.dataset == 'german':
        config = 'assets/german_credit.json'
    elif args.dataset == 'heloc':
        config = 'assets/heloc.json'
    elif args.dataset == 'ctg':
        config = 'assets/ctg.json'
    epsilons = [0.001, 0.001, 0.01, 0.01]
    bias_epsilons = [0.01, 0.001, 0.01, 0.001]
    for i in range(10):
        for j, e, be in zip([j for j in range(len(epsilons))], epsilons, bias_epsilons):
            # call eval.py as if from the command line -- not eval.main but the whole script
            modelname = args.model + args.dataset + args.cfx + str(i)
            logname = modelname + "_" + str(j) + ".txt"
            # eval(args = [modelname, '--config', config, '--save_dir', 'trained_models/'+ args.dataset, '--cfx_save_dir',
            #              'saved_cfxs/fromchtc', '--cfx', args.cfx, '--epsilon', e, '--bias_epsilon', be])       
            exc_str = r"python eval.py " + modelname + r" --config " + config + r" --save_dir " + \
                        args.model_dir +  r" --cfx_save_dir " + args.cfx_dir + r" --cfx " + args.cfx + \
                        r" --epsilon " + str(e) + r" --bias_epsilon " + str(be) + r" --log_name " + \
                        logname + r" --cfx_filename " + modelname
            if args.cfx == 'proto':
                exc_str += " --onehot"
            print(exc_str)
            os.system(exc_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset',choices=['german','heloc','ctg'])
    parser.add_argument('model',choices=['Standard','IBP'])
    parser.add_argument('cfx',choices=['wachter','proto'])
    parser.add_argument('--cfx_dir', default="saved_cfxs/fromchtc", help="directory where cfxs are saved")
    parser.add_argument('--model_dir', default=None, help="directory where models are saved, if omitted will be trained_models/dataset")

    args = parser.parse_args()

    if args.model_dir == None:
        args.model_dir = 'trained_models/' + args.dataset
    main(args)