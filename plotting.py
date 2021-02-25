import json
import os
from glob import glob
from typing import List, Tuple
from zipfile import ZipFile

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap

sns.set_palette("muted")
sns.set_style("whitegrid")

COMPUTE_STEPS = ["init", "fwd_pass", "comp_loss", "backprop", "opt_step", "end"]
COMMUNICATION_STEPS = ["agg"]

RENAME = {
    "CumulativeTrainTimeEpoch": "Total time",
    "backprop": "Backpropagation",
    "comp_loss": "Loss Computation",
    "agg": "Aggregation",
    "fwd_pass": "Forward Pass",
    "opt_step": "Optimizer Step",
    "init": "Batch init",
    "end": "Batch End",
}


def to_float(val) -> float:
    try:
        data = float(val)
    except ValueError as e:
        val = val.replace("tensor(", "")
        val = val.replace(", device='cuda:0')", "")
        data = float(val)
    return data


def read_values(data: List[dict]) -> List[float]:
    """Reads and converts values from a List of dicts. Each dict should contain the key `value`

    Args:
        data: Input dictionary

    Returns:
        List of converted values to float
    """
    data = [to_float(x["value"]) for x in data]
    return data


def get_cum_times(
    archive: ZipFile, file: str, num_workers: int, prefix: str = "result_global_cum_"
) -> Tuple[str, float, int]:
    """Reads all cumulative time values from the archive. Uses all files that start with `prefix`.
    Each file should contain a 1 element list of dicts

    Args:
        archive: Archive from which to read the file
        file: The file to read
        num_workers: Number of workers
        prefix: File prefix to remove

    Returns:

    """
    data = json.loads(archive.read(file))
    file = os.path.splitext(file)[0]
    name, _, _ = file.replace(prefix, "").split()

    return name, to_float(data[0]["value"]), num_workers


def get_all_cumulative_times(
    archive: ZipFile, prefix: str = "result_global_cum_"
) -> pd.DataFrame:
    """Reads and returns all cumulative times from the archive

    Args:
        archive: Archive to read from
        prefix: Prefix of files to use

    Returns:
        pd.DataFrame with columns ["name", "value", "num_workers"]
    """
    num_workers = int(
        os.path.splitext(os.path.basename(archive.filename))[0].split("-")[-1]
    )
    cum_files = [r for r in archive.namelist() if prefix in r]

    times = [get_cum_times(archive, c, num_workers, prefix=prefix) for c in cum_files]
    return pd.DataFrame(times, columns=["name", "value", "num_workers"])


def get_avg_loss(
    archive: ZipFile, key: str = "result_train_loss"
) -> List[Tuple[int, float]]:
    """Computes the average loss over all workers

    Args:
        archive: Archive to read from
        key: Common substring in file names to average from

    Returns:
        The losses with their index
    """
    files = [c for c in archive.namelist() if key in c]
    data = np.array([read_values(json.loads(archive.read(c))) for c in files])
    data = np.mean(data, axis=0)
    data = [(i, x) for i, x in enumerate(data)]
    return data


def parse_archive(file: str) -> pd.DataFrame:
    """Parses an arhive file by:
    - Reading all cumulative times
    - Reading train and validation loss

    Args:
        file: Archive file to read

    Returns:
        DataFrame containing all parsed values
    """
    backend = (
        os.path.splitext(os.path.basename(file))[0].split("_")[-1].split("-")[0].upper()
    )

    archive = ZipFile(file)
    df = get_all_cumulative_times(archive)
    workers = df.num_workers.unique()[0]

    train_loss = get_avg_loss(archive, key="result_train_loss")
    val_loss = get_avg_loss(archive, key="result_val_loss")

    train_loss = pd.Series(
        {"name": "Train Loss", "num_workers": workers, "value": train_loss}
    )
    val_loss = pd.Series(
        {"name": "Validation Loss", "num_workers": workers, "value": val_loss}
    )

    df = df.append(train_loss, ignore_index=True)
    df = df.append(val_loss, ignore_index=True)

    df["backend"] = backend
    return df.reset_index(drop=True)


def flatten_array_var(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Flattens an array contained in dataframe cells

    Args:
        df: DataFrame
        name: Name of row to keep

    Returns:
        pd.DataFrame in long form
    """
    x = df[df.name == name].explode("value")
    x["index"], x["value"] = zip(*x.value)
    return x


def plot_losses(df: pd.DataFrame, dest: str):
    """Plots the losses for each backend on a different graph

    Args:
        df: DataFrame containing data
        dest: Destination directory to save images to

    """
    backends = df.backend.unique()
    for b in backends:
        x = df[df.backend == b]
        train_losses = flatten_array_var(x, "Train Loss")
        val_losses = flatten_array_var(x, "Validation Loss")
        losses = pd.concat([train_losses, val_losses])

        g = sns.relplot(
            x="index",
            y="value",
            hue="num_workers",
            col="name",
            palette="muted",
            data=losses,
            kind="line",
            facet_kws={"sharex": False},
            alpha=0.75,
        )
        g.set_axis_labels(x_var="Step", y_var="Loss")
        g.set_titles("{col_name}")
        g._legend.set_title("Workers")
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle("Losses {}".format(b))

        name = os.path.join(dest, "losses_{}.png".format(b))

        g.savefig(name, dpi=300)


def plot_step_times(df: pd.DataFrame, steps: List[str], rename: dict, dest: str):
    """Plots the step times of the steps given in argument. Rename allows for renaming those steps

    Args:
        df: DataFrame containing data
        steps: Steps to plot
        rename: Renaming of steps
        dest: Destination directory

    """
    x = df[df.name.isin(steps)].reset_index()
    x["value"] = x["value"].astype(np.float32)
    x = x.replace(rename)
    col_order = list(rename.values())

    g = sns.relplot(
        x="num_workers",
        y="value",
        data=x,
        col="name",
        col_wrap=4,
        kind="line",
        col_order=col_order,
        height=4.5,
        hue="backend",
    )
    g.set_titles("{col_name}", fontsize=12)

    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle("Total step times", fontsize=15)
    g.set_axis_labels(x_var="Workers", y_var="Time (s)", fontsize=12)

    name = os.path.join(dest, "step_times.png")
    g.savefig(name, dpi=300)


def plot_shares(df: pd.DataFrame, steps: List[str], rename: dict, dest: str):
    """Plots a stacked barplot showing the shares for each step in the total time

    Args:
        df: DataFrame
        steps: Steps to use
        rename: Renaming of steps
        dest: Destination directory

    """
    backends = df.backend.unique()
    for b in backends:
        x = df[df.name.isin(steps) & (df.backend == b)].reset_index(drop=True)
        x["value"] = pd.to_numeric(x["value"])
        x = x.pivot_table(index="num_workers", values="value", columns=["name"])
        x = x.rename(columns=rename)

        # Compute percentages
        percentages = x.div(x.sum(axis=1), 0) * 100

        fig, ax = plt.subplots(figsize=(15, 10))
        x.plot(
            kind="bar",
            stacked=True,
            colormap=ListedColormap(sns.color_palette()),
            alpha=0.75,
            rot=0,
            ax=ax,
        )

        for n in percentages:
            for i, (cs, ab, pc) in enumerate(zip(x.cumsum(1)[n], x[n], percentages[n])):
                if pc > 3:
                    ax.text(
                        i,
                        cs - ab / 2,
                        str(np.round(pc, 1)) + "%",
                        va="center",
                        ha="center",
                    )
        ax.set_title("Step share {}".format(b), fontsize=15)
        ax.set_xlabel("Workers", fontsize=12)
        ax.set_ylabel("Time (s)", fontsize=12)
        name = os.path.join(dest, "step_shares_{}.png".format(b))

        fig.savefig(name, dpi=300)


def plot_speedups(
    df: pd.DataFrame,
    dest: str,
    key: str = "CumulativeTrainTimeEpoch",
    baseline_workers: int = 1,
    title: str = "Speedups w.r. to 1 worker",
    name="speedups",
):
    """Plots the speedups for each backend w.r. to `baseline_workers`

    Args:
        df: Dataframe to use
        key: Key of value to use
        dest: Destination dir
        baseline_workers: Baseline number of workers
        title: Plot title
        name: File name to save to

    """
    data = df[df.name == key].reset_index(drop=True)

    baseline = data[data.num_workers == baseline_workers][["backend", "value"]]
    data = data.merge(baseline, how="left", on="backend").reset_index(drop=True)
    data["speedup"] = data["value_y"] / data["value_x"]

    fig, ax = plt.subplots(figsize=(10, 8))
    hue_order = sorted(data.backend.unique())
    g = sns.barplot(
        x="num_workers",
        y="speedup",
        data=data,
        alpha=0.9,
        hue="backend",
        ax=ax,
        hue_order=hue_order,
    )

    for p in g.patches:
        g.annotate(
            format(p.get_height(), ".1f"),
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 9),
            textcoords="offset points",
        )

    g.set_xlabel("Workers")
    g.set_ylabel("Speedup")
    g.set_title(title)

    name = os.path.join(dest, "{}.png".format(name))

    fig.savefig(name, dpi=300)


@click.command()
@click.argument("src", type=str)
@click.option(
    "--custom-compute",
    type=str,
    help="Additional compute steps to use (str representation of dict)",
)
@click.option(
    "--custom-comm",
    type=str,
    help="Additional communication steps to use (str representation of dict)",
)
def main(src, custom_compute, custom_comm):
    """Generates the plots from a source directory and stores them in `<src>/graphs/`
    The source directory should contain zip files named `metrics_<backend>-<num_workers>.zip`
    """
    files = glob(os.path.join(src, "*.zip"))
    dest = os.path.join(src, "graphs")
    if not os.path.exists(dest):
        os.mkdir(dest)

    dfs = []
    for f in files:
        dfs.append(parse_archive(f))
    df = pd.concat(dfs).reset_index(drop=True)

    if custom_comm is not None:
        custom_comm = json.loads(custom_comm)
        keys = list(custom_comm.keys())
        COMMUNICATION_STEPS.extend(keys)

        RENAME.update(custom_comm)

    if custom_compute is not None:
        custom_compute = json.loads(custom_compute)
        keys = list(custom_compute.keys())
        COMPUTE_STEPS.extend(keys)

        RENAME.update(custom_compute)

    time_cats = ["CumulativeTrainTimeEpoch"] + COMPUTE_STEPS + COMMUNICATION_STEPS
    grouped = (
        df[df.name.isin(COMPUTE_STEPS)]
        .groupby(["num_workers", "backend"])["value"]
        .sum()
        .reset_index()
    )
    grouped["name"] = "CumulativeCompute"
    df = pd.concat([df, grouped])

    plot_step_times(df, time_cats, rename=RENAME, dest=dest)

    plot_shares(df, COMPUTE_STEPS + COMMUNICATION_STEPS, RENAME, dest=dest)
    plot_losses(df, dest=dest)
    plot_speedups(
        df, dest=dest, title="Total speedups w.r. to 1 worker", name="total_speedups"
    )
    plot_speedups(
        df,
        dest=dest,
        key="CumulativeCompute",
        title="Compute speedups w.r. to 1 worker",
        name="compute_speedups",
    )


if __name__ == "__main__":
    main()
