from ensemble_compilation.graph_representation import SchemaGraph, Table


def get_schema(data_source, dataset_id, csv_path):
    if (data_source == "uci") and (dataset_id == "household_power_consumption"):
        return get_household_power_consumption_schema(csv_path)
    else:
        raise ValueError("Invalid dataset.")


def get_household_power_consumption_schema(csv_path):
    schema = SchemaGraph()
    schema.add_table(
        Table(
            "household_power_consumption",
            attributes=[
                "global_active_power",
                "global_reactive_power",
                "voltage",
                "global_intensity",
                "sub_metering_1",
                "sub_metering_2",
                "sub_metering_3",
            ],
            csv_file_location=csv_path,
            table_size=2049280,
        )
    )
    return schema
