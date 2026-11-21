"""Heatmap of where a player was on the pitch.

Two fixes over a plain heatmap:
- Square-root scaling: if he stood still for a minute at a free kick, that spot
  would otherwise be so much hotter than anywhere else that his running all over
  the pitch fades to nothing. Square-root keeps the quieter areas visible.
- His movement path is drawn faintly on top, so you can see every place he went,
  not just where he spent the most time.
"""

import numpy as np


def pitch_heatmap(traj, length=105.0, width=68.0, flip=False, minutes=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    from stats import draw_pitch

    fig, ax = plt.subplots(figsize=(10, 10 * (width + 8) / (length + 6)))
    fig.patch.set_facecolor("#3a7d3a")
    draw_pitch(ax, length, width)
    if len(traj):
        x, y = traj.x.to_numpy(float), traj.y.to_numpy(float)
        if flip:
            x, y = length - x, width - y
        grid, _, _ = np.histogram2d(x, y, bins=[int(length), int(width)],
                                    range=[[0, length], [0, width]])
        grid = gaussian_filter(grid, 2.5)
        if grid.max() > 0:
            g = np.sqrt(grid / grid.max())        # keep quieter areas visible
            alpha = np.clip(g * 1.1, 0, 0.85)
            alpha[g < 0.06] = 0
            ax.imshow(g.T, origin="lower", extent=[0, length, 0, width], cmap="magma",
                      alpha=alpha.T, zorder=2, interpolation="bilinear")
        # Faint path, broken wherever he wasn't tracked.
        segs = traj.segment.to_numpy() if "segment" in traj else np.zeros(len(traj))
        for s in np.unique(segs):
            m = segs == s
            ax.plot(x[m], y[m], color="white", lw=0.6, alpha=0.25, zorder=3)
        ax.scatter([x.mean()], [y.mean()], s=170, color="white", ec="black", zorder=5,
                   label="Average position")
        label = "Average position"
        if minutes is not None:
            label += f"   |   based on {minutes:.1f} min of him on the pitch"
        ax.legend([ax.collections[-1]], [label], loc="upper center",
                  bbox_to_anchor=(0.5, 0.0), frameon=False, labelcolor="white", fontsize=10)
    fig.tight_layout(pad=0.3)
    return fig
