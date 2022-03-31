from argparse import ArgumentParser
from privddnn.utils.file_utils import read_json_gz, save_json_gz

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--metadata-path', type=str, required=True)
    parser.add_argument('--new-dataset', type=str, required=True)
    args = parser.parse_args()

    model_params = read_json_gz(args.metadata_path)
    metadata = model_params['metadata']

    metadata['DATASET_NAME'] = args.new_dataset
    save_json_gz(model_params, args.metadata_path)
