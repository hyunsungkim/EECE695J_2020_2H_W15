import matplotlib.pyplot as plt
import numpy as np

def plot_weight_dist(w_dict, b_dict):
    """
        This function plots distribution of parameters. The
        second subplot describes the histogram of the absolute
        values.
    """
    w = []
    for item in w_dict.values():
        w.extend(item.ravel())

    for item in b_dict.values():
        w.extend(item.ravel())
    
    fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(15,15))
    ax1.set_title('Weight distribution')
    ax1.hist(np.array(w).flatten(), bins=400)
#    ax1.set_xlim()
    ax2.hist(np.array(np.abs(w)).flatten(), bins=400)
