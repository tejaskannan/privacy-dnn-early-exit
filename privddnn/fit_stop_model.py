import numpy as np
import scipy.optimize
from argparse import ArgumentParser
from functools import partial

from neural_network import restore_model, NeuralNetwork, OpName, ModelMode


def sigmoid(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(-1 * x)
    return 1.0 / (1.0 + exp_x)


def softmax(x: np.ndarray) -> np.ndarray:
    max_x = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def temp_scaling(probs: np.ndarray, temp: float) -> np.ndarray:
    logits = np.log(probs)
    return softmax(logits - temp)


def thresh_loss(x: np.ndarray, temp: float, probs: np.ndarray, labels: np.ndarray, target: float, should_print: bool) -> float:
    return loss_fn(temp=temp, thresholds=x, probs=probs, labels=labels, target=target, should_print=should_print)


def temp_loss(x: float, thresholds: np.ndarray, probs: np.ndarray, labels: np.ndarray, target: float, should_print: bool) -> float:
    return loss_fn(temp=x, thresholds=thresholds, probs=probs, labels=labels, target=target, should_print=should_print)


def loss_fn(temp: float, thresholds: np.ndarray, probs: np.ndarray, labels: np.ndarray, target: float, should_print: bool) -> float:
    # Scale the probabilites by the temperature
    scaled_probs = temp_scaling(probs=probs, temp=temp)

    preds = np.argmax(scaled_probs, axis=-1)
    stop_probs = np.max(scaled_probs, axis=-1)

    num_labels = probs.shape[-1]

    sample_thresholds = np.zeros(shape=labels.shape)
    for pred, t in enumerate(thresholds):
        mask = np.equal(preds, pred).astype(float)
        sample_thresholds += (mask * t)

    loss = 0.0
    above = np.greater(stop_probs, sample_thresholds).astype(float)

    for label in range(num_labels):
        mask = np.equal(labels, label).astype(float)

        avg_above = np.sum(mask * above) / np.sum(mask)

        if should_print:
            print('{} -> {}'.format(label, avg_above))

        loss += abs(avg_above - target)

    return loss



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help='Path to the saved model weights')
    args = parser.parse_args()

    # Restore the model
    model: NeuralNetwork = restore_model(path=args.model_path, model_mode=ModelMode.TEST)

    # Get the dataset
    val_probs = model.validate(op=OpName.PROBS)[:, 0, :]
    val_labels = model.dataset.get_val_labels()

    rand = np.random.RandomState(seed=5851)
    num_labels = val_probs.shape[-1]

    thresholds = rand.uniform(low=0.0, high=1.0, size=(num_labels, ))
    temp = rand.uniform(low=-0.7, high=0.7)
    target = 0.3
    num_iters = 2


    for _ in range(num_iters):

        thresh_loss_fn = partial(thresh_loss, temp=temp, probs=val_probs, labels=val_labels, target=target, should_print=False)
        thresh_res = scipy.optimize.minimize(fun=thresh_loss_fn,
                                             x0=thresholds,
                                             method='BFGS')

        print(thresh_res.fun)
        thresholds = thresh_res.x

        temp_loss_fn = partial(temp_loss, thresholds=thresholds, probs=val_probs, labels=val_labels, target=target, should_print=False)
        temp_res = scipy.optimize.minimize(fun=temp_loss_fn,
                                           x0=temp,
                                           method='BFGS')

        print(temp_res.fun)
        temp = temp_res.x

    x_final = result.x
    val_loss = loss_fn(x=x_final, probs=val_probs, labels=val_labels, target=target, should_print=True)

    test_probs = model.test(op=OpName.PROBS)[:, 0, :]
    test_labels = model.dataset.get_test_labels()
    test_loss = loss_fn(x=x_final, probs=test_probs, labels=test_labels, target=target, should_print=True)

    print('Val Loss: {}, Test Loss: {}'.format(val_loss, test_loss))
