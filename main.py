import logging
import os

from config import LOG_FORMAT, NAME_DELIMITER, DATA_DIR
from data_preparation.prepare_single_tables import prepare_all_tables
from ensemble_creation.naive import create_naive_all_split_ensemble
from evaluation.aqp_evaluation import compute_ground_truth
from evaluation.aqp_evaluation import evaluate_aqp_queries
from schemas.aqp_datasets.schema import get_schema

LOGGING_LEVEL = logging.INFO
HDF_FILES_GENERATED = True
ENSEMBLE_GENERATED = True
GROUND_TRUTH_COMPUTED = True
DATA_SOURCE = "uci"
DATASET_ID = "household_power_consumption"
QUERIES_SET = "uci-household_power_consumption-N=100_small"
DATABASE_NAME = "uci_household_power_consumption"
MAX_ROWS_PER_HDF_FILE = 100000000
SAMPLES_PER_SPN = 10000000
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
SHOW_CONFIDENCE_INTERVALS = True


def main():
    # Definitions
    dataset_full_id = DATA_SOURCE + NAME_DELIMITER + DATASET_ID
    csv_path = os.path.join(DATA_DIR, "uncompressed", dataset_full_id + ".csv")
    query_filepath = os.path.join(
        "aqp_evaluation", "queries", dataset_full_id, QUERIES_SET + ".sql"
    )
    hdf_path = os.path.join(
        "aqp_evaluation",
        "hdf",
        dataset_full_id,
        dataset_full_id + "_" + str(MAX_ROWS_PER_HDF_FILE),
    )
    ensemble_path = os.path.join("aqp_evaluation", "spn_ensembles", dataset_full_id)
    ensemble_filepath = os.path.join(
        ensemble_path,
        "ensemble_single_" + dataset_full_id + "_" + str(SAMPLES_PER_SPN) + ".pkl",
    )
    ground_truth_path = os.path.join("aqp_evaluation", "ground_truth", dataset_full_id)
    ground_truth_filepath = os.path.join(ground_truth_path, QUERIES_SET + ".pkl")
    results_path = os.path.join("aqp_evaluation", "results", dataset_full_id)
    results_filepath = os.path.join(
        results_path, QUERIES_SET + "_" + str(SAMPLES_PER_SPN) + ".csv"
    )

    # Generate database schema
    logger.info("Generating schema.")
    schema = get_schema(DATA_SOURCE, DATASET_ID, csv_path)

    # Generate HDF files for simpler sampling
    if not HDF_FILES_GENERATED:
        logger.info(f"Generate HDF files for {DATASET_ID} and store in path {hdf_path}")
        os.makedirs(hdf_path, exist_ok=True)
        prepare_all_tables(
            schema,
            hdf_path,
            csv_seperator=",",
            csv_header=0,
            max_table_data=MAX_ROWS_PER_HDF_FILE,
        )
        logger.info("HDF files successfully created")

    # Generate ensemble
    if not ENSEMBLE_GENERATED:
        logger.info(f"Generate ensemble and store in path {ensemble_path}")
        os.makedirs(ensemble_path, exist_ok=True)
        create_naive_all_split_ensemble(
            schema,
            hdf_path,
            SAMPLES_PER_SPN,
            ensemble_path,
            dataset_full_id,
            BLOOM_FILTERS,
            RDC_THRESHOLD,
            MAX_ROWS_PER_HDF_FILE,
            POST_SAMPLING_FACTOR,
            incremental_learning_rate=INCREMENTAL_LEARNING_RATE,
        )

    # Compute ground truth for AQP queries
    if not GROUND_TRUTH_COMPUTED:
        logger.info("Computing ground truth")
        os.makedirs(ground_truth_path, exist_ok=True)
        compute_ground_truth(
            ground_truth_filepath,
            DATABASE_NAME,
            db_type="ss",
            db_server_id="D43139",
            db_user="aqp",
            db_password="aqpevaluation",
            db_host="localhost",
            db_port=1434,
            query_filename=query_filepath,
        )
        logger.info("Ground truth completed")

    # Read pre-trained ensemble and evaluate AQP queries
    evaluate_aqp_queries(
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
        show_confidence_intervals=SHOW_CONFIDENCE_INTERVALS,
        confidence_sample_size=SAMPLES_PER_SPN,
        confidence_interval_alpha=CONFIDENCE_INTERVAL_ALPHA,
    )

    # Read pre-trained ensemble and evaluate the confidence intervals
    # TODO ... or maybe not relevant?


if __name__ == "__main__":
    logging.basicConfig(level=LOGGING_LEVEL, format=LOG_FORMAT)
    logger = logging.getLogger("main")
    main()
