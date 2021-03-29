import matplotlib
from matplotlib import pyplot as plt
import numpy as np
# from numpy import squeeze as np.squeeze


# matplotlib.use("Agg")


def show_plot(images, mask_true, mask_pred, save_path):
    n = images.shape[0]

    plt.figure(figsize=(n, 3), dpi=400)
    # plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for i in range(n):
        plt.subplot(3, n, i + 1)
        plt.imshow(np.squeeze(images[i]))
        plt.axis(False)
        plt.tight_layout()

        plt.subplot(3, n, i + 1 + n)
        plt.imshow(np.squeeze(mask_true[i])[:, :, 1], cmap="gray")
        plt.axis(False)
        plt.tight_layout()

        plt.subplot(3, n, i + 1 + n * 2)
        plt.imshow(mask_pred[i, :, :, 1], cmap="gray")
        plt.axis(False)
        plt.tight_layout()

    plt.savefig(save_path)
