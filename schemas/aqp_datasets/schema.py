import json

from ensemble_compilation.graph_representation import SchemaGraph, Table


def get_schema(data_source, dataset_id, csv_path):
    with open("schemas.json", "r") as fp:
        schemas = json.load(fp)
    schema = SchemaGraph()
    schema.add_table(
        Table(
            data_source + "_" + dataset_id,
            attributes=schemas[data_source][dataset_id]["column_names"],
            csv_file_location=csv_path,
            table_size=schemas[data_source][dataset_id]["n_rows"],
        )
    )
    return schema
