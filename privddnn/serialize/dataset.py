import os.path
import numpy as np
from argparse import ArgumentParser
from typing import List

from privddnn.dataset.dataset import Dataset, DataFold
from privddnn.dataset.data_iterators import make_data_iterator
from privddnn.serialize.utils import array_to_fixed_point
from privddnn.utils.file_utils import make_dir


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset-name', type=str, required=True, help='The name of the dataset to serialize.')
    parser.add_argument('--precision', type=int, required=True, help='The fixed point precision.')
    parser.add_argument('--fold', type=str, required=True, choices=['train', 'val', 'test'], help='The dataset fold to serialize.')
    parser.add_argument('--data-order', type=str, required=True, choices=['nearest', 'randomized'], help='Name of the dataset order to serialize.')
    parser.add_argument('--num-inputs', type=int, required=True, help='The maximum number of inputs to include.')
    parser.add_argument('--window', type=int, help='The window size to use for the `nearest` order.')
    parser.add_argument('--is-msp', action='store_true', help='Whether to prepare the dataset for the MSP430 device.')
    args = parser.parse_args()

    assert (args.precision >= 0) and (args.precision <= 15), 'The precision must be in [0, 16).'

    dataset = Dataset(dataset_name=args.dataset_name)

    # Normalize the values
    dataset.fit_normalizer(is_global=False)
    dataset.normalize_data()

    data_iterator = make_data_iterator(name=args.data_order,
                                       dataset=dataset,
                                       clf=None,
                                       num_trials=1,
                                       fold=args.fold,
                                       window_size=args.window)

    input_list: List[np.ndarray] = []
    label_list: List[np.ndarray] = []

    for idx, (features, _, label) in enumerate(data_iterator):
        if idx >= args.num_inputs:
            break

        input_list.append(np.expand_dims(features, axis=0))
        label_list.append(label)

    inputs = np.vstack(input_list)
    labels = np.vstack(label_list).reshape(-1)

    # Quantize the input features
    quantized_inputs = array_to_fixed_point(inputs, precision=args.precision, width=16)

    # Serialize the inputs into a static C array
    num_inputs, num_features = inputs.shape

    input_str = list(map(str, quantized_inputs.reshape(-1)))
    input_var = 'static int16_t DATASET_INPUTS[] = {{ {} }};\n'.format(','.join(input_str))

    label_str = list(map(str, labels))
    label_var = 'static uint8_t DATASET_LABELS[] = {{ {} }};\n'.format(','.join(label_str))

    with open('data.h', 'w') as fout:
        fout.write('#include <stdint.h>\n')

        if args.is_msp:
            fout.write('#include <msp430.h>\n')

        fout.write('#ifndef DATA_H_\n')
        fout.write('#define DATA_H_\n')

        fout.write('#define NUM_FEATURES {}\n'.format(num_features))
        fout.write('#define NUM_INPUTS {}\n'.format(num_inputs))

        # For MSP430 implementations, place the data into FRAM
        if args.is_msp:
            fout.write('#pragma PERSISTENT(DATASET_INPUTS)\n')

        fout.write(input_var)

        # Do not save the labels for MSP430 versions. The labels are only there
        # for testing purposes.
        if not args.is_msp:
            fout.write(label_var)

        fout.write('#endif\n')
