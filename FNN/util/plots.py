import os
import numpy as np
import matplotlib.pyplot as plt

def parity_plot(pred, value, inp, var_name, parity_dir_path):
    pred = pred.detach().cpu().numpy()
    value = value.detach().cpu().numpy()

    
    def fmt(x):
        try:
            return f"{float(x):g}"
        except Exception:
            return str(x)
        

    subdir = "_".join(fmt(x) for x in inp[0][:3])
    out_dir = os.path.join(parity_dir_path, subdir)
    os.makedirs(out_dir, exist_ok=True)

    for i, var in enumerate(var_name):
        p = pred[:, i]
        v = value[:, i]
        fig, ax = plt.subplots()
        ax.scatter(v, p, s=10, alpha=0.6, edgecolors="none")

        minv = np.nanmin([v.min(), p.min()])
        maxv = np.nanmax([v.max(), p.max()])
        ax.plot([minv, maxv], [minv, maxv], "k--", linewidth=1)

        ax.set_xlabel("True value")
        ax.set_ylabel("Predicted value")
        ax.set_title(f"Parity: {var}")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(minv, maxv)
        ax.set_ylim(minv, maxv)
        fig.tight_layout()

        fig.savefig(os.path.join(out_dir, f"{var}.png"), dpi=150)
        plt.close(fig)
