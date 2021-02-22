import json
import os
from glob import glob
from zipfile import ZipFile

import click
import pandas as pd
import seaborn as sns

# Files needed to generate the plots
RESULT_FILES = {
    "result_val_tensor_size @ {}.json": "tensor_size",
    "result_val_cuda @ {}.json": "cuda",
    "result_val_dtype @ {}.json": "dtype",
    "result_val_avg_time @ {}.json": "avg_time",
}


def read_json_file_from_zip(zip_file, filename):
    """
    Reads a json file contained in a ZipFile
    Args:
        zip_file (ZipFile): The zipfile containing the metrics
        filename (str): Filename to read from

    Returns:
        list[dict] | None: The values read from the given filename, `None` file is not in archive.
    """
    if filename not in zip_file.namelist():
        print("Could not find {} in {}".format(filename, zip_file.filename))
        return None
    with zip_file.open(filename) as f:
        result = json.load(f)
    return result


def read_values_from_json(json_list):
    """Reads the "value" attribute in a list of dicts

    Args:
        json_list (list[dict]): List of dicts, read from metrics results

    Returns:
        list[float]: The list of values, converted to float
    """
    return [float(x["value"]) for x in json_list]


def extract_values_for_worker(zip_file, worker):
    """Extracts all the values from `RESULT_FILES` for the given worker

    Args:
        zip_file (ZipFile): The zipfile containing the metrics
        worker (int): Worker ID

    Returns:
        pd.DataFrame: DataFrame containing all the measurements for a single worker
    """
    result = {}
    values_len = -1
    for k, v in RESULT_FILES.items():
        values = read_values_from_json(
            read_json_file_from_zip(zip_file, k.format(worker))
        )

        # Make sure all lengths are the same
        if values_len == -1:
            values_len = len(values)
        else:
            assert values_len == len(values)

        result[v] = values

    result["worker"] = [worker] * values_len

    result = pd.DataFrame(result)

    result = (
        result.groupby(["tensor_size", "cuda", "dtype", "worker"])["avg_time"]
        .mean()
        .reset_index()
    )
    return result


def parse_result_zip(filename):
    """Parses an archive containing metrics for Backend Benchmarking

    Args:
        filename (str): Filename of archive. Must end with the number of workers

    Returns:
        pd.DataFrame: Containing all the measurements, for all workers
    """
    zip_file = ZipFile(filename)
    num_workers = int(filename.split(".")[-2].split("-")[-1])

    dfs = []
    for i in range(num_workers):
        dfs.append(extract_values_for_worker(zip_file, i))

    df = pd.concat(dfs)
    df = df.groupby(["tensor_size", "cuda", "dtype"])["avg_time"].mean().reset_index()
    df["num_workers"] = num_workers
    return df


def get_backend_results(backend, directory, wildcard="metrics_{backend}-*.zip"):
    """Parses all archives related to a backend

    Args:
        backend (str): Backend to parse (mpi, gloo or nccl)
        directory (str): Directory containing the archives
        wildcard (str): Wildcard to search for archives (default: `"metrics_{backend}-*.zip"`)

    Returns:
        pd.DataFrame: The results for the given backend.
    """
    files = glob(os.path.join(directory, wildcard.format(backend=backend)))

    dfs = []
    for f in files:
        df = parse_result_zip(f)
        dfs.append(df)

    df = pd.concat(dfs)
    df["backend"] = backend
    df["cuda"] = df["cuda"].apply(bool)
    df["dtype"] = df["dtype"].apply(lambda x: "Float32" if x == 32 else "Float16")
    return df


def generate_backend_analysis_plot(backend, directory, by="tensor"):
    df = get_backend_results(backend, directory)

    workers = sorted(df["num_workers"].unique())
    if by == "workers":
        row_order = sorted(df["dtype"].unique())
        col_order = sorted(df["cuda"].unique())
        g = sns.FacetGrid(
            row="dtype",
            col="cuda",
            hue="num_workers",
            data=df,
            height=5,
            sharex=True,
            row_order=row_order,
            col_order=col_order,
        )
        g.fig.suptitle("{} by number of workers".format(backend.upper()), size=16)

    elif by == "tensor":
        row_order = sorted(df["dtype"].unique())
        g = sns.FacetGrid(
            row="dtype",
            col="num_workers",
            hue="cuda",
            data=df,
            height=5,
            sharex=True,
            row_order=row_order,
            col_order=workers,
        )
        g.fig.suptitle("{} by tensor type".format(backend.upper()), size=16)

    else:
        raise ValueError("Unknown graph type {}".format(by))

    g = g.map(sns.lineplot, "tensor_size", "avg_time")

    g.set_axis_labels("Tensor Size (log)", "Communication Time (log(s))")
    g.add_legend()
    for row in g.axes:
        for ax in row:
            ax.set_xscale("log")
            ax.set_yscale("log")

    g.fig.subplots_adjust(top=0.9)
    return g


def generate_backend_comparison_plot(
    backends, directory, by="tensor", num_workers=0, cuda=False
):
    dfs = []
    for backend in backends:
        df = get_backend_results(backend, directory)
        dfs.append(df)

    df = pd.concat(dfs)
    workers = sorted(df["num_workers"].unique())

    col_order = sorted(df["cuda"].unique())
    row_order = sorted(df["dtype"].unique())
    if by == "tensor":
        df = df[df["num_workers"] == num_workers]
        g = sns.FacetGrid(
            row="dtype",
            col="cuda",
            hue="backend",
            data=df,
            height=5,
            sharex=True,
            row_order=row_order,
            col_order=col_order,
        )
        g.fig.suptitle(
            "Comparison of {} for {} workers".format(
                ", ".join([b.upper() for b in backends]), num_workers
            )
        )
    elif by == "workers":
        df = df[df["num_workers"].isin(workers)]
        df = df[df["cuda"] == cuda]

        g = sns.FacetGrid(
            row="dtype",
            col="num_workers",
            hue="backend",
            data=df,
            height=5,
            sharex=True,
            row_order=row_order,
            col_order=workers,
        )
        g.fig.suptitle(
            "Comparison of {} for {} workers {}".format(
                ", ".join([b.upper() for b in backends]),
                workers,
                ", CUDA tensors" if cuda else "",
            )
        )

    g = g.map(sns.lineplot, "tensor_size", "avg_time")

    g.set_axis_labels("Tensor Size (log)", "Communication Time (log(s))")
    g.add_legend()
    for row in g.axes:
        for ax in row:
            ax.set_xscale("log")
            ax.set_yscale("log")

    g.fig.subplots_adjust(top=0.9)
    return g


@click.command()
@click.option(
    "--backends", default="mpi,gloo,nccl", type=str, help="Coma separated backend list"
)
@click.option(
    "--source-dir", "-s", type=str, help="Source directory containing results archives"
)
@click.option("--dest-dir", "-d", type=str, help="Destination directory for graphs")
@click.option(
    "--num-workers",
    default="2,4,8",
    type=str,
    help="Number of workers used for experiments (coma separated)",
)
def main(backends, source_dir, dest_dir, num_workers):
    sns.set_style("whitegrid")
    click.echo("Generating all plots for backends {}".format(backends))

    num_workers = [int(x) for x in num_workers.split(",")]
    backends = backends.split(",")

    for b in backends:
        g_tensor = generate_backend_analysis_plot(b, source_dir, by="tensor")
        g_workers = generate_backend_analysis_plot(b, source_dir, by="workers")

        g_tensor.savefig(os.path.join(dest_dir, "{}_analysis_by_tensor.png".format(b)))
        g_workers.savefig(
            os.path.join(dest_dir, "{}_analysis_by_workers.png".format(b))
        )

    for n in sorted(num_workers):
        g = generate_backend_comparison_plot(
            backends, source_dir, by="tensor", num_workers=n
        )
        g.savefig(
            os.path.join(
                dest_dir, "backends_comparison_by_tensor_{}_workers.png".format(n)
            )
        )

    for cuda in [True, False]:
        g = generate_backend_comparison_plot(
            backends, source_dir, by="workers", cuda=cuda
        )
        g.savefig(
            os.path.join(
                dest_dir,
                "backends_comparison_by_workers" + ("_CUDA" if cuda else "") + ".png",
            )
        )


if __name__ == "__main__":
    main()
