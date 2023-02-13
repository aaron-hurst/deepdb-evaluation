import os
import logging

import numpy as np
import pandas as pd

from config import LOG_FORMAT, GROUND_TRUTH_DIR, RESULTS_DIR, NAME_DELIMITER

LOGGING_LEVEL = logging.INFO
N_AGGREGATIONS = 3
N_PAIRWISE_QUERIES = 100
QUERY_SET = "chicago-taxi_trips_2020-N=100"
SAMPLES_PER_SPN = 1000


def compute_relative_error(true, predicted):
    if np.isnan(predicted):
        return 100
    elif true != 0:
        return abs((predicted - true) / true) * 100
    elif predicted == 0:
        return 0
    else:
        return 100


def compute_relative_ci(true, ci_half_width):
    if np.isnan(ci_half_width):
        return 100
    elif true != 0:
        return abs(ci_half_width / true) * 100
    elif ci_half_width == 0:
        return 0
    else:
        return 100


if __name__ == "__main__":
    logging.basicConfig(level=LOGGING_LEVEL, format=LOG_FORMAT)
    logger = logging.getLogger("main")

    # Inputs
    data_source, dataset_id, _ = QUERY_SET.split("-")
    dataset_full_id = data_source + NAME_DELIMITER + dataset_id
    ground_truth_filepath = os.path.join(
        GROUND_TRUTH_DIR, dataset_full_id, QUERY_SET + "_gt.csv"
    )
    results_filepath = os.path.join(
        RESULTS_DIR,
        "aqp",
        "deepdb",
        "results",
        dataset_full_id,
        QUERY_SET,
        f"sample_size_{SAMPLES_PER_SPN}",
        "results.csv",
    )

    # Load original columns
    df = pd.read_csv(
        results_filepath,
        usecols=[
            "query_id",
            "predicate_column",
            "aggregation_column",
            "aggregation",
            "latency",
            "predicted_value",
            "ci_low",
            "ci_high",
        ],
    )

    # Fix aggregation column values
    n_columns = df["query_id"].max() // N_PAIRWISE_QUERIES + 1
    df["predicate_column"] = df["query_id"] // N_PAIRWISE_QUERIES
    df["aggregation_column"] = (df.index // N_AGGREGATIONS) % n_columns

    # Merge with ground truth data
    df_gt = pd.read_csv(ground_truth_filepath)
    df = pd.merge(
        df,
        df_gt,
        how="left",
        on=["query_id", "predicate_column", "aggregation_column", "aggregation"],
    )

    # Compute error and bounds statistics
    df["error"] = df["predicted_value"] - df["exact_value"]
    df["relative_error"] = df.apply(
        lambda x: compute_relative_error(x.exact_value, x.predicted_value), axis=1
    )
    df["ci_half_width"] = (df["ci_high"] - df["ci_low"]) / 2
    df["ci_half_width_relative"] = df.apply(
        lambda x: compute_relative_ci(x.exact_value, x.ci_half_width),
        axis=1,
    )

    # Export results
    logger.info("Exporting results.")
    os.makedirs(os.path.dirname(results_filepath), exist_ok=True)
    # df.to_csv(results_filepath, index=False)

    # Display results
    logger.info(
        "Median relative error by column:\n%s",
        df.groupby(["predicate_column", "aggregation_column"])
        .agg({"relative_error": "median"})
        .unstack()
        .round(2),
    )
    logger.info(
        "Relative error by aggregatation:\n%s",
        df.groupby("aggregation")[["relative_error"]]
        .describe(percentiles=[0.5, 0.75, 0.95])
        .round(3),
    )
