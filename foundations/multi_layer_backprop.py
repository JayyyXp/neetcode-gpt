from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:

        def matmul(A, x):
            return [sum(A[i][j] * x[j] for j in range(len(x)))
                    for i in range(len(A))]

        def vecplus(x, y):
            return [x[i] + y[i] for i in range(len(x))]

        def relu(x):
            return [max(0, xi) for xi in x]

        # --- Forward pass ---
        z1 = vecplus(matmul(W1, x), b1)
        a1 = relu(z1)
        z2 = vecplus(matmul(W2, a1), b2)

        n = len(z2)
        loss = sum((z2[i] - y_true[i])**2 for i in range(n)) / n

        # --- Backward pass ---

        # Output layer
        dL_dz2 = [(2/n) * (z2[i] - y_true[i]) for i in range(n)]

        dW2 = [[dL_dz2[i] * a1[j] for j in range(len(a1))]
                for i in range(len(dL_dz2))]
        db2 = dL_dz2

        # Backprop through W2 to a1
        dL_da1 = [sum(W2[i][j] * dL_dz2[i] for i in range(len(dL_dz2)))
                  for j in range(len(a1))]

        # Backprop through ReLU
        dL_dz1 = [dL_da1[i] * (1 if z1[i] > 0 else 0)
                  for i in range(len(z1))]

        # Input layer
        dW1 = [[dL_dz1[i] * x[j] for j in range(len(x))]
                for i in range(len(dL_dz1))]
        db1 = dL_dz1

        return {
            'loss': round(loss, 4),
            'dW2': [[round(v, 4) or 0.0 for v in row] for row in dW2],
            'db2': [round(v, 4) for v in db2],
            'dW1': [[round(v, 4) or 0.0 for v in row] for row in dW1],
            'db1': [round(v, 4) for v in db1],
        }