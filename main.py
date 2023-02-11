from time import perf_counter
import logging
import os

import numpy as np
import pandas as pd

from config import (
    LOG_FORMAT,
    NAME_DELIMITER,
    DATA_DIR,
    GROUND_TRUTH_DIR,
    QUERIES_DIR,
    RESULTS_DIR,
)
from data_preparation.prepare_single_tables import prepare_all_tables
from ensemble_creation.naive import create_naive_all_split_ensemble
# from evaluation.aqp_evaluation import compute_ground_truth
from evaluation.aqp_evaluation import evaluate_aqp_queries
from schemas.aqp_datasets.schema import get_schema

LOGGING_LEVEL = logging.INFO
GENERATE_HDF_FILES = True  # force creation of new HDF files
GENERATE_ENSEMBLE = True  # forve creation of new ensembles
QUERY_SETS = [
    "ampds-basement_plugs_and_lights-N=100",
    "ampds-current-N=100",
    "ampds-furnace_and_thermostat-N=100",
    "chicago-taxi_trips_2020-N=100",
    "kaggle-aquaponics_all-N=100",
    "kaggle-light_detection-N=100",
    "kaggle-smart_building_system_413-N=100",
    "kaggle-smart_building_system_621A-N=100",
    "kaggle-smart_building_system_all-N=100",
    "kaggle-temperature_iot_on_gcp_100k-N=100",
    "kaggle-temperature_iot_on_gcp_500k-N=100",
    "kaggle-temperature_iot_on_gcp-N=100",
    "uci-gas_sensor_home_activity-N=100",
    "uci-household_power_consumption-N=100",
]

# Model parameters
MAX_ROWS_PER_HDF_FILE = 10000000
CONFIDENCE_INTERVAL_ALPHA = 0.99
BLOOM_FILTERS = False
RDC_THRESHOLD = 0.3
POST_SAMPLING_FACTOR = 10
INCREMENTAL_LEARNING_RATE = 0
RDC_SPN_SELECTION = False
PAIRWISE_RDC_PATH = None
MAX_VARIANTS = 1
MERGE_INDICATOR_EXP = True
EXPLOIT_OVERLAPPING = True


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


def run_experiment(query_set, samples_per_spn):
    logger.info(f"Analysing query set: {query_set}")
    logger.info(f"Samples per SPN: {samples_per_spn}")

    # Inputs
    data_source, dataset_id, _ = query_set.split("-")
    dataset_full_id = data_source + NAME_DELIMITER + dataset_id
    csv_path = os.path.join(DATA_DIR, "uncompressed", dataset_full_id + ".csv")
    query_filepath = os.path.join(QUERIES_DIR, dataset_full_id, query_set + ".sql")
    database_name = dataset_full_id.replace(NAME_DELIMITER, "_")

    # Outputs
    output_dir = os.path.join(RESULTS_DIR, "aqp", "deepdb")
    hdf_path = os.path.join(
        output_dir,
        "hdf",
        dataset_full_id,
        dataset_full_id + "_" + str(MAX_ROWS_PER_HDF_FILE),
    )
    hdf_filename = dataset_id + ".hdf"
    ensemble_path = os.path.join(output_dir, "spn_ensembles", dataset_full_id)
    ensemble_filename = f"ensemble_single_{dataset_full_id}_{samples_per_spn}.pkl"
    ensemble_filepath = os.path.join(ensemble_path, ensemble_filename)
    ground_truth_filepath = os.path.join(
        GROUND_TRUTH_DIR, dataset_full_id, query_set + "_gt.csv"
    )
    results_path = os.path.join(
        output_dir,
        "results",
        dataset_full_id,
        query_set,
        f"sample_size_{samples_per_spn}",
    )
    results_filepath = os.path.join(results_path, "results.csv")
    info_filepath = os.path.join(results_path, "info.txt")

    # Generate database schema
    logger.info("Generating schema.")
    schema = get_schema(data_source, dataset_id, csv_path)

    # Generate HDF files for simpler sampling
    t_generate_hdf_start = perf_counter()
    os.makedirs(hdf_path, exist_ok=True)
    if GENERATE_HDF_FILES or (hdf_filename not in os.listdir(hdf_path)):
        logger.info(f"Generate HDF files for {dataset_id}")
        prepare_all_tables(
            schema,
            hdf_path,
            csv_seperator=",",
            csv_header=0,
            max_table_data=MAX_ROWS_PER_HDF_FILE,
        )
        logger.info("HDF files successfully created")
    t_generate_hdf = perf_counter() - t_generate_hdf_start

    # Generate ensemble
    t_generate_ensemble_start = perf_counter()
    os.makedirs(ensemble_path, exist_ok=True)
    if GENERATE_ENSEMBLE or (ensemble_filename not in os.listdir(ensemble_path)):
        logger.info(f"Generate SPN ensemble.")
        create_naive_all_split_ensemble(
            schema,
            hdf_path,
            samples_per_spn,
            ensemble_path,
            dataset_full_id,
            BLOOM_FILTERS,
            RDC_THRESHOLD,
            MAX_ROWS_PER_HDF_FILE,
            POST_SAMPLING_FACTOR,
            incremental_learning_rate=INCREMENTAL_LEARNING_RATE,
        )
    t_generate_ensemble = perf_counter() - t_generate_ensemble_start

    # Read pre-trained ensemble and evaluate AQP queries
    logger.info("Evaluating queries")
    t_queries_start = perf_counter()
    n_queries, results = evaluate_aqp_queries(
        ensemble_filepath,
        query_filepath,
        results_filepath,
        schema,
        ground_truth_filepath,
        RDC_SPN_SELECTION,
        PAIRWISE_RDC_PATH,
        max_variants=MAX_VARIANTS,
        merge_indicator_exp=MERGE_INDICATOR_EXP,
        exploit_overlapping=EXPLOIT_OVERLAPPING,
        min_sample_ratio=0,
        debug=True,
        show_confidence_intervals=True,
        confidence_sample_size=samples_per_spn,
        confidence_interval_alpha=CONFIDENCE_INTERVAL_ALPHA,
    )
    t_queries = perf_counter() - t_queries_start

    # Merge with ground truth data
    df_gt = pd.read_csv(ground_truth_filepath)
    df = pd.DataFrame(results)
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
    df.to_csv(results_filepath, index=False)

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

    # Get file sizes
    s_original = schema.tables[0].table_size * len(schema.tables[0].attributes) * 4
    s_hdf = os.stat(os.path.join(hdf_path, hdf_filename)).st_size
    s_ensemble = os.stat(ensemble_filepath).st_size

    # Export parameters and statistics
    t_construction = t_generate_hdf + t_generate_ensemble
    os.makedirs(os.path.dirname(info_filepath), exist_ok=True)
    logger.info(f"Saving experiment parameters and statistics to {info_filepath}")
    if n_queries:
        mean_latency = t_queries / n_queries
    else:
        mean_latency = 0
    with open(info_filepath, "w", newline="") as f:
        f.write(f"------------- Parameters -------------\n")
        f.write(f"GENERATE_HDF_FILES           {GENERATE_HDF_FILES}\n")
        f.write(f"GENERATE_ENSEMBLE            {GENERATE_ENSEMBLE}\n")
        f.write(f"DATA_SOURCE                  {data_source}\n")
        f.write(f"DATASET_ID                   {dataset_id}\n")
        f.write(f"QUERIES_SET                  {query_set}\n")
        f.write(f"DATABASE_NAME                {database_name}\n")
        f.write(f"MAX_ROWS_PER_HDF_FILE        {MAX_ROWS_PER_HDF_FILE}\n")
        f.write(f"SAMPLES_PER_SPN              {samples_per_spn}\n")
        f.write(f"CONFIDENCE_INTERVAL_ALPHA    {CONFIDENCE_INTERVAL_ALPHA}\n")
        f.write(f"BLOOM_FILTERS                {BLOOM_FILTERS}\n")
        f.write(f"RDC_THRESHOLD                {RDC_THRESHOLD}\n")
        f.write(f"POST_SAMPLING_FACTOR         {POST_SAMPLING_FACTOR}\n")
        f.write(f"INCREMENTAL_LEARNING_RATE    {INCREMENTAL_LEARNING_RATE}\n")
        f.write(f"RDC_SPN_SELECTION            {RDC_SPN_SELECTION}\n")
        f.write(f"PAIRWISE_RDC_PATH            {PAIRWISE_RDC_PATH}\n")
        f.write(f"MAX_VARIANTS                 {MAX_VARIANTS}\n")
        f.write(f"MERGE_INDICATOR_EXP          {MERGE_INDICATOR_EXP}\n")
        f.write(f"EXPLOIT_OVERLAPPING          {EXPLOIT_OVERLAPPING}\n")

        f.write(f"\n------------- Runtime -------------\n")
        f.write(f"Generate HDF files           {t_generate_hdf:.3f} s\n")
        f.write(f"Generate SPN ensembles       {t_generate_ensemble:.3f} s\n")
        f.write(f"Total construction time      {t_construction:.3f} s\n")
        f.write(f"Run queries                  {t_queries:.3f} s\n")
        f.write(f"Queries executed             {n_queries}\n")
        f.write(f"Mean latency                 {mean_latency:.6f} s\n")

        f.write(f"\n------------- Storage -------------\n")
        f.write(f"Original data                {s_original:,d} bytes\n")
        f.write(f"HDF files                    {s_hdf:,d} bytes\n")
        f.write(f"SPN ensembles                {s_ensemble:,d} bytes\n")
        f.write(f"SPN ensembles (%)            {s_ensemble / s_original * 100:.2f} %\n")


def main():
    # Run all experiments
    for query_set in QUERY_SETS:
        for samples_per_spn in [1000]:  # , 10000, 100000, 1000000]:
            run_experiment(query_set, samples_per_spn=samples_per_spn)

    # Run a single experiment
    # run_experiment(
    #     "uci-household_power_consumption-N=100", min_pts=0.001, alpha=4000
    # )


if __name__ == "__main__":
    logging.basicConfig(level=LOGGING_LEVEL, format=LOG_FORMAT)
    logger = logging.getLogger("main")
    np.seterr(all="raise")
    main()
