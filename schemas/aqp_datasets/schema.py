import json
import os

from ensemble_compilation.graph_representation import SchemaGraph, Table

from config import DATA_DIR

SUFFIXES = {"_10m": 10000000, "_100m": 100000000, "_1b": 1000000000}


def get_schema(dataset_id, csv_path):
    """Return SchemaGraph object for dataset based on JSON schema file."""
    n_rows = None
    dataset_id_no_suffix = dataset_id
    for suffix in SUFFIXES:
        if suffix in dataset_id:
            dataset_id_no_suffix = dataset_id_no_suffix.removesuffix(suffix)
            n_rows = SUFFIXES[suffix]
    if n_rows is None:
        dataset_id_no_suffix = dataset_id
        n_rows = sum(1 for _ in open(csv_path)) - 1  # excludes header
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
        )
    )
    return schema_graph, schema
