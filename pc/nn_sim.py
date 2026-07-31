
import numpy as np


class QuantizedNN:
    def __init__(self, npz_path="pc/nn_table.npz"):
        data = np.load(npz_path)
        self.W1_q = data["W1_q"]
        self.b1_q = data["b1_q"]
        self.W2_q = data["W2_q"]
        self.b2_q = data["b2_q"]

    def predict(self, x: np.ndarray):
        x_int = np.round(x).astype(np.int32)

        acc1 = x_int @ self.W1_q.astype(np.int32) + self.b1_q
        hidden = np.maximum(acc1, 0)

        acc2 = hidden.astype(np.int64) @ self.W2_q.astype(np.int64) + self.b2_q.astype(np.int64)

        return int(np.argmax(acc2)), acc2


if __name__ == "__main__":
    from sklearn.datasets import load_digits

    nn = QuantizedNN()
    digits = load_digits()

    print("Testando 10 amostras aleatórias do dataset completo:\n")
    rng = np.random.default_rng(7)
    idxs = rng.choice(len(digits.data), size=10, replace=False)
    for idx in idxs:
        x = digits.data[idx]
        true_label = digits.target[idx]
        pred, scores = nn.predict(x)
        flag = "" if pred == true_label else "  <-- ERRO"
        print(f"real={true_label}  previsto={pred}  scores={scores.tolist()}{flag}")
