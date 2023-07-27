import json

from ensemble_compilation.graph_representation import SchemaGraph, Table


def get_schema(dataset_id, csv_path):
    # TODO special code for handling scaled up datasets (i.e., different n_rows)
    with open("schemas.json", "r") as fp:
        schemas = json.load(fp)
    schema = SchemaGraph()
    schema.add_table(
        Table(
            dataset_id.replace("-", "_"),
            attributes=schemas[dataset_id]["column_names"],
            csv_file_location=csv_path,
            table_size=schemas[dataset_id]["n_rows"],
        )
    )
    schema_raw = schemas[dataset_id]
    return schema, schema_raw
