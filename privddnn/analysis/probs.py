import numpy as np
import math


W = 10
PROBS = [0.79060665, 0.79871324, 0.79477251, 0.804, 0.80212355, 0.79449361, 0.79757085, 0.79254783, 0.79262213, 0.79785156]

def choose(n, k):
    return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))

epsilon = 0.0

for n in range(W):
    for label1 in range(len(PROBS)):
        for label2 in range(label1 + 1, len(PROBS)):
            p1 = PROBS[label1]
            p2 = PROBS[label2]

            prob1 = np.power(p1, n) * np.power(1.0 - p1, W - n)
            prob2 = np.power(p2, n) * np.power(1.0 - p2, W - n)

            ratio = prob2 / prob1

            epsilon = max(1.0 - ratio, epsilon)

print('Epsilon: {}'.format(epsilon))
