import json
import os

from ensemble_compilation.graph_representation import SchemaGraph, Table

from config import DATA_DIR

SUFFIXES = ["_10m", "_100m", "_1b"]


def get_schema(dataset_id, csv_path):
    """Return SchemaGraph object for dataset based on JSON schema file."""
    dataset_id_no_suffix = dataset_id
    for suffix in SUFFIXES:
        dataset_id_no_suffix = dataset_id_no_suffix.removesuffix(suffix)
    filepath = os.path.join(DATA_DIR, "schemas", "aqp", f"{dataset_id_no_suffix}.json")
    with open(filepath, "r") as f:
        schema = json.load(f)
    schema_graph = SchemaGraph()
    schema_graph.add_table(
        Table(
            dataset_id.replace("-", "_"),
            attributes=schema["column_names"],
            csv_file_location=csv_path,
            table_size=schema["n_rows"],
        )
    )
    return schema_graph, schema
