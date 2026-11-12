"""Fit k-means on class skill ratings, save an elbow plot, and label k=3."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

from cluster_model import predict_cluster

SKILL_COLUMNS = [
    "skill_programming",
    "skill_data_stats",
    "skill_frontend_design",
    "skill_business_pitch",
]

DATA_PATH = Path(__file__).resolve().parent / "class_dataset.csv"
PLOT_PATH = Path(__file__).resolve().parent / "elbow.png"
CHOSEN_K = 3


def main() -> None:
    students = pd.read_csv(DATA_PATH)
    skills = students[SKILL_COLUMNS]
    print(f"students: {len(students)}")

    ks = list(range(1, 7))
    inertias = []
    for k in ks:
        model = KMeans(n_clusters=k, n_init=10, random_state=0)
        model.fit(skills)
        inertias.append(model.inertia_)
        print(f"k={k} inertia={model.inertia_:.4f}")

    fig, ax = plt.subplots()
    ax.plot(ks, inertias, marker="o")
    ax.set_xlabel("k")
    ax.set_ylabel("inertia")
    ax.set_title("K-means elbow")
    ax.set_xticks(ks)
    fig.tight_layout()
    fig.savefig(PLOT_PATH)
    plt.close(fig)

    chosen = KMeans(n_clusters=CHOSEN_K, n_init=10, random_state=0)
    students["cluster"] = chosen.fit_predict(skills)
    for label, centroid in enumerate(chosen.cluster_centers_):
        count = int((students["cluster"] == label).sum())
        averages = ", ".join(f"{value:.2f}" for value in centroid)
        print(f"cluster {label}: n={count} centroid=({averages})")

    students.to_csv(DATA_PATH, index=False)

    matched = 0
    for ratings, label in zip(
        skills.itertuples(index=False, name=None), students["cluster"]
    ):
        if predict_cluster(ratings) == label:
            matched += 1
    print(f"matches: {matched}")


if __name__ == "__main__":
    main()
