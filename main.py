import json
import logging
import os
from datetime import datetime
from time import perf_counter

import numpy as np
import pandas as pd

from config import LOG_FORMAT, DATA_DIR, QUERIES_DIR, RESULTS_DIR
from data_preparation.prepare_single_tables import prepare_all_tables
from ensemble_compilation.graph_representation import SchemaGraph, Table
from ensemble_creation.naive import create_naive_all_split_ensemble
from evaluation.aqp_evaluation import evaluate_aqp_queries

LOGGING_LEVEL = logging.INFO

DATASETS = {
    "ampds-basement_plugs_and_lights": 2,
    "ampds-current": 2,
    "ampds-furnace_and_thermostat": 2,
    "chicago-taxi_trips_2020": 4,
    "kaggle-aquaponics": 4,
    "kaggle-light_detection": 2,
    "kaggle-smart_building_system": 2,
    "kaggle-temperature_iot_on_gcp": 4,
    "uci-gas_sensor_home_activity": 2,
    "uci-household_power_consumption": 2,
    "usdot-flights": 2,
    # "uci-household_power_consumption": 15,
    # "uci-household_power_consumption_synthetic": 15,
    # "uci-household_power_consumption_10m": 15,
    # "uci-household_power_consumption_100m": 15,
    # "uci-household_power_consumption_1b": 15,
    # "usdot-flights": 4,
    # "usdot-flights_synthetic": 4,
    # "usdot-flights_10m": 4,
    # "usdot-flights_100m": 4,
    # "usdot-flights_1b": 4,
}
SUFFIXES = {"_synthetic": None, "_10m": 10000000, "_100m": 100000000, "_1b": 1000000000}

SAMPLING_RANDOM_SEEDS = [15, 82, 6, 94, 67]

GENERATE_HDF_FILES = False  # force creation of new HDF files
GENERATE_ENSEMBLE = True  # force creation of new ensembles
INCLUDE_FAILED = True

HDF_MAX_ROWS = 10000000
SAMPLES_PER_SPN = 10000
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


def get_schema(dataset_id, csv_path):
    """Return SchemaGraph object for dataset based on JSON schema file."""
    n_rows = None
    dataset_id_no_suffix = dataset_id
    for suffix in SUFFIXES:
        if suffix in dataset_id:
            dataset_id_no_suffix = dataset_id_no_suffix.removesuffix(suffix)
            n_rows = SUFFIXES[suffix]
    if n_rows is None:
        n_rows = sum(1 for _ in open(csv_path)) - 1  # excludes header
    sample_rate = min(1, HDF_MAX_ROWS / n_rows)
    filepath = os.path.join(DATA_DIR, "schemas", "aqp", f"{dataset_id_no_suffix}.json")
    with open(filepath, "r") as f:
        schema = json.load(f)
    schema_graph = SchemaGraph()
    schema_graph.add_table(
        Table(
            dataset_id.replace("-", "_"),
            attributes=schema["column_names"],
            csv_file_location=csv_path,
            table_size=n_rows,
            sample_rate=sample_rate,
        )
    )
    return schema_graph, schema


def get_relative_error_pct(true, predicted):
    if np.isnan(predicted):
        return 100
    elif true != 0:
        return abs((predicted - true) / true) * 100
    elif predicted == 0:
        return 0
    else:
        return 100


def get_relative_bound_pct(true, ci_half_width):
    if np.isnan(ci_half_width):
        return 100
    elif true != 0:
        return abs(ci_half_width / true) * 100
    elif ci_half_width == 0:
        return 0
    else:
        return 100


def test_dataset(dataset_id, query_set, random_seed):
    logger.info(f"Analysing dataset: {dataset_id}")
    logger.info(f"Samples per SPN: {SAMPLES_PER_SPN}")

    # Inputs
    csv_path = os.path.join(DATA_DIR, "processed", f"{dataset_id}.csv")
    queries_filepath = os.path.join(QUERIES_DIR, f"{dataset_id}_v{query_set}.txt")

    # Outputs
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = os.path.join(RESULTS_DIR, "aqp", "deepdb")
    hdf_filepath = os.path.join(
        output_dir, "hdf", f"{dataset_id}_max_rows_{HDF_MAX_ROWS}.hdf"
    )
    ensemble_filepath = os.path.join(
        output_dir, "spn_ensembles", f"{dataset_id}_single_{SAMPLES_PER_SPN}.pkl"
    )
    ground_truth_filepath = os.path.join(
        QUERIES_DIR, "ground_truth", f"{dataset_id}_v{query_set}_gt.csv"
    )
    results_path = os.path.join(
        output_dir,
        "results",
        dataset_id,
    )
    results_name = f"queries_v{query_set}_sample_size_{SAMPLES_PER_SPN}_{timestamp}"
    results_filepath = os.path.join(results_path, results_name + ".csv")
    metadata_filepath = os.path.join(results_path, results_name + "_metadata.txt")

    # Generate database schema
    logger.info("Generating schema...")
    schema, _ = get_schema(dataset_id, csv_path)

    # Generate HDF files for simpler sampling
    t_generate_hdf_start = perf_counter()
    os.makedirs(os.path.dirname(hdf_filepath), exist_ok=True)
    if GENERATE_HDF_FILES or (
        os.path.basename(hdf_filepath) not in os.listdir(os.path.dirname(hdf_filepath))
    ):
        logger.info("Generating HDF files...")
        prepare_all_tables(
            schema,
            hdf_filepath,
            csv_seperator=",",
            csv_header=0,
            max_table_data=HDF_MAX_ROWS,
        )
        logger.info("HDF files successfully created")
    t_generate_hdf = perf_counter() - t_generate_hdf_start

    # Generate ensemble
    t_generate_ensemble_start = perf_counter()
    os.makedirs(os.path.dirname(ensemble_filepath), exist_ok=True)
    if GENERATE_ENSEMBLE or (
        os.path.basename(ensemble_filepath)
        not in os.listdir(os.path.dirname(ensemble_filepath))
    ):
        logger.info("Generating SPN ensemble...")
        create_naive_all_split_ensemble(
            schema,
            hdf_filepath,
            SAMPLES_PER_SPN,
            ensemble_filepath,
            dataset_id,
            BLOOM_FILTERS,
            RDC_THRESHOLD,
            HDF_MAX_ROWS,
            POST_SAMPLING_FACTOR,
            incremental_learning_rate=INCREMENTAL_LEARNING_RATE,
            random_seed=random_seed,
        )
    t_generate_ensemble = perf_counter() - t_generate_ensemble_start

    # Evaluate queries using pre-trained ensemble
    logger.info("Evaluating queries...")
    t_queries_start = perf_counter()
    n_queries, results = evaluate_aqp_queries(
        ensemble_filepath,
        queries_filepath,
        None,
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
        confidence_sample_size=SAMPLES_PER_SPN,
        confidence_interval_alpha=CONFIDENCE_INTERVAL_ALPHA,
    )
    t_queries = perf_counter() - t_queries_start

    # Merge with ground truth data
    df_gt = pd.read_csv(ground_truth_filepath)
    df = pd.DataFrame(results)
    if not INCLUDE_FAILED:
        df = df.dropna()
    df = pd.merge(df, df_gt, how="left", on=["query_id"])

    # Compute error and bounds statistics
    df["error"] = df["estimate"] - df["exact_result"]
    df["error_relative_pct"] = df.apply(
        lambda x: get_relative_error_pct(x.exact_result, x.estimate), axis=1
    )
    df["bounds_width"] = df["bound_high"] - df["bound_low"]
    df["bounds_width_relative_pct"] = df.apply(
        lambda x: get_relative_bound_pct(x.exact_result, x.bounds_width),
        axis=1,
    )
    df["bound_is_accurate"] = (df["bound_high"] >= df["exact_result"]) & (
        df["bound_low"] <= df["exact_result"]
    )

    # Export results
    logger.info("Exporting results.")
    os.makedirs(os.path.dirname(results_filepath), exist_ok=True)
    df.to_csv(results_filepath, index=False)

    # Get file sizes for metadata
    s_hdf = os.stat(hdf_filepath).st_size
    s_ensemble = os.stat(ensemble_filepath).st_size

    # Export parameters and statistics
    t_construction = t_generate_hdf + t_generate_ensemble
    os.makedirs(os.path.dirname(metadata_filepath), exist_ok=True)
    logger.info("Saving experiment metadata...")
    if n_queries:
        mean_latency = t_queries / n_queries
    else:
        mean_latency = 0
    with open(metadata_filepath, "w", newline="") as f:
        f.write("------------- Parameters -------------\n")
        f.write(f"DATASET_ID                   {dataset_id}\n")
        f.write(f"QUERIES_SET                  {query_set}\n")
        f.write(f"GENERATE_HDF_FILES           {GENERATE_HDF_FILES}\n")
        f.write(f"GENERATE_ENSEMBLE            {GENERATE_ENSEMBLE}\n")
        f.write(f"HDF_MAX_ROWS                 {HDF_MAX_ROWS}\n")
        f.write(f"SAMPLES_PER_SPN              {SAMPLES_PER_SPN}\n")
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

        f.write("\n------------- Runtime -------------\n")
        f.write(f"Generate HDF files           {t_generate_hdf:.3f} s\n")
        f.write(f"Generate SPN ensembles       {t_generate_ensemble:.3f} s\n")
        f.write(f"Total construction time      {t_construction:.3f} s\n")
        f.write(f"Run queries                  {t_queries:.3f} s\n")
        f.write(f"Queries executed             {n_queries}\n")
        f.write(f"Mean latency                 {mean_latency:.6f} s\n")

        f.write("\n------------- Storage -------------\n")
        f.write(f"HDF files                    {s_hdf} bytes\n")
        f.write(f"SPN ensembles                {s_ensemble} bytes\n")
    logger.info(f"Completed dataset {dataset_id} with query set {query_set}")


def main():
    for dataset_id, query_set in DATASETS.items():
        for random_seed in SAMPLING_RANDOM_SEEDS:
            test_dataset(dataset_id, query_set, random_seed)


if __name__ == "__main__":
    logging.basicConfig(level=LOGGING_LEVEL, format=LOG_FORMAT)
    logger = logging.getLogger("main")
    np.seterr(all="raise")
    main()
