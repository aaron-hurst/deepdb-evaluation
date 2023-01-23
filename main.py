from time import perf_counter
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
    results_path = os.path.join(
        "aqp_evaluation",
        "results",
        dataset_full_id,
        QUERIES_SET + "_" + str(SAMPLES_PER_SPN),
    )
    results_filepath = os.path.join(results_path, "results.csv")
    info_filepath = os.path.join(results_path, "info.txt")

    # Generate database schema
    logger.info("Generating schema.")
    schema = get_schema(DATA_SOURCE, DATASET_ID, csv_path)

    # Generate HDF files for simpler sampling
    t_generate_hdf_start = perf_counter()
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
    t_generate_hdf = perf_counter() - t_generate_hdf_start

    # Generate ensemble
    t_generate_ensemble_start = perf_counter()
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
    t_generate_ensemble = perf_counter() - t_generate_ensemble_start

    # Compute ground truth for AQP queries
    t_ground_truth_start = perf_counter()
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
    t_ground_truth = perf_counter() - t_ground_truth_start

    # Read pre-trained ensemble and evaluate AQP queries
    t_queries_start = perf_counter()
    n_queries = evaluate_aqp_queries(
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
    t_queries = perf_counter() - t_queries_start

    # Read pre-trained ensemble and evaluate the confidence intervals
    # TODO ... or maybe not relevant?

    # Get file sizes
    s_original = schema.tables[0].table_size * len(schema.tables[0].attributes) * 4
    s_hdf = os.stat(os.path.join(hdf_path, DATASET_ID + ".hdf")).st_size
    s_ensemble = os.stat(ensemble_filepath).st_size

    # Export parameters and statistics
    t_construction = t_generate_hdf + t_generate_ensemble
    os.makedirs(os.path.dirname(info_filepath), exist_ok=True)
    logger.info(f"Saving experiment parameters and statistics to {info_filepath}")
    with open(info_filepath, "w", newline="") as f:
        f.write(f"------------- Parameters -------------\n")
        f.write(f"HDF_FILES_GENERATED          {HDF_FILES_GENERATED}\n")
        f.write(f"ENSEMBLE_GENERATED           {ENSEMBLE_GENERATED}\n")
        f.write(f"GROUND_TRUTH_COMPUTED        {GROUND_TRUTH_COMPUTED}\n")
        f.write(f"DATA_SOURCE                  {DATA_SOURCE}\n")
        f.write(f"DATASET_ID                   {DATASET_ID}\n")
        f.write(f"QUERIES_SET                  {QUERIES_SET}\n")
        f.write(f"DATABASE_NAME                {DATABASE_NAME}\n")
        f.write(f"MAX_ROWS_PER_HDF_FILE        {MAX_ROWS_PER_HDF_FILE}\n")
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
        f.write(f"SHOW_CONFIDENCE_INTERVALS    {SHOW_CONFIDENCE_INTERVALS}\n")

        f.write(f"\n------------- Runtime -------------\n")
        f.write(f"Generate HDF files           {t_generate_hdf:.6f} s\n")
        f.write(f"Generate SPN ensembles       {t_generate_ensemble:.6f} s\n")
        f.write(f"Total construction time      {t_construction:.6f} s\n")
        f.write(f"Compute ground truth         {t_ground_truth:.6f} s\n")
        f.write(f"Run queries                  {t_queries:.6f} s\n")
        f.write(f"Queries executed             {n_queries}\n")
        f.write(f"Mean latency                 {t_queries / n_queries:.6f} s\n")

        f.write(f"\n------------- Storage -------------\n")
        f.write(f"Original data                {s_original:,d}\n")
        f.write(f"HDF files                    {s_hdf:,d}\n")
        f.write(f"SPN ensembles                {s_ensemble:,d}\n")
        f.write(f"SPN ensembles (%)            {s_ensemble / s_original * 100:.2f} %\n")


if __name__ == "__main__":
    logging.basicConfig(level=LOGGING_LEVEL, format=LOG_FORMAT)
    logger = logging.getLogger("main")
    main()
