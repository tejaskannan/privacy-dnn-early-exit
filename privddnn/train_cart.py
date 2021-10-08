import os.path
from argparse import ArgumentParser
from datetime import datetime

from privddnn.utils.file_utils import make_dir
from privddnn.ensemble.adaboost import AdaBoostClassifier



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset-name', type=str, required=True)
    parser.add_argument('--classifier', type=str, required=True)
    args = parser.parse_args()

    model = AdaBoostClassifier(num_estimators=100, exit_size=10, clf_name=args.classifier, dataset_name=args.dataset_name)
    model.fit()

    # Make the output file name
    current_time = datetime.now()
    model_name = '{}_{}'.format(args.classifier, current_time.strftime('%d-%m-%Y-%H-%M-%S'))

    save_folder = 'saved_models'
    make_dir(save_folder)

    save_folder = os.path.join(save_folder, args.dataset_name)
    make_dir(save_folder)

    save_folder = os.path.join(save_folder, current_time.strftime('%d-%m-%Y'))
    make_dir(save_folder)

    save_path = os.path.join(save_folder, '{}.pkl.gz'.format(model_name))

    # Save the model
    model.save(path=save_path)
