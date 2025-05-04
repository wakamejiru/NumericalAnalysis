import numpy as np
import matplotlib.pyplot as plt

def G2_exact(x1, x2):
    """
    元の式を計算する関数
    """
    return np.array([
        (x2**3 - 3 * x2 * x1**2 + 2 * x1**3) / 6,
        (2 * x2**3 - 3 * x1 * x2**2 + x1**3) / 6
    ])

def G2_approx(x1, h):
    """
    近似式を計算する関数
    """
    return (h / 6) * np.array([
        3 * x1 + h,
        3 * x1 + 2 * h
    ])

if __name__ == "__main__":
    # パラメータ設定
    x1_values = np.linspace(0, 1, 100)  # x1 の範囲
    h_values = [0.1, 0.2, 0.3]  # h の値

    # プロット用のデータリスト
    exact_values = []
    approx_values = []

    # 各hの値について計算
    for h in h_values:
        exact_h_values = []
        approx_h_values = []
        for x1 in x1_values:
            x2 = x1 + h
            exact = G2_exact(x1, x2)
            approx = G2_approx(x1, h)
            exact_h_values.append(exact)
            approx_h_values.append(approx)
        exact_values.append(np.array(exact_h_values))
        approx_values.append(np.array(approx_h_values))

    # プロット
    plt.figure(figsize=(10, 6))

    for i, h in enumerate(h_values):
        plt.plot(x1_values, exact_values[i][:, 0], label=f'Exact (h={h}), Component 1', linestyle='-')
        plt.plot(x1_values, approx_values[i][:, 0], label=f'Approx (h={h}), Component 1', linestyle='--')
        plt.plot(x1_values, exact_values[i][:, 1], label=f'Exact (h={h}), Component 2', linestyle=':')
        plt.plot(x1_values, approx_values[i][:, 1], label=f'Approx (h={h}), Component 2', linestyle='-.')

    plt.xlabel('x1')
    plt.ylabel('G2 Value')
    plt.title('Comparison of Exact and Approximate G2')
    plt.legend()
    plt.grid(True)
    plt.show()