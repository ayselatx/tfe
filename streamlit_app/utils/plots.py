import matplotlib.pyplot as plt
import streamlit as st

from utils.paths import (
    IMAGE_PATHS,
    FLICKR8K_CAPTIONS,
    vision_emb_path,
    text_emb_path,
    vision_xai_dir,
    text_xai_dir,
)


def plot_recall_bars(df, return_fig=False):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(df["model"], df["recall@1"])
    ax.set_title("Recall@1 Comparison")

    if return_fig:
        return fig
    else:
        st.pyplot(fig)


def plot_radar(df, return_fig=False):
    import matplotlib.pyplot as plt
    import numpy as np

    labels = df.columns[1:]  # assuming first column is model
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)

    for _, row in df.iterrows():
        values = row[labels].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=row["model"])
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    if return_fig:
        return fig
    else:
        st.pyplot(fig)

