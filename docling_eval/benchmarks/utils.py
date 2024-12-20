

def write_datasets_info(name: str, output_dir:Path, num_train_rows:int, num_test_rows:int):

    columns =  [
        {"name": BenchMarkColumns.DOCLING_VERSION, "type": "string"},
        {"name": BenchMarkColumns.STATUS, "type": "string"},
        {"name": BenchMarkColumns.DOC_ID, "type": "string"},
        {"name": BenchMarkColumns.GROUNDTRUTH, "type": "string"},
        {"name": BenchMarkColumns.PREDICTION, "type": "string"},
        {"name": BenchMarkColumns.ORIGINAL, "type": "string"},
        {"name": BenchMarkColumns.MIMETYPE, "type": "string"},
        {"name": BenchMarkColumns.PICTURES, "type": { "list": { "item": "Image"}}},
        {"name": BenchMarkColumns.PAGE_IMAGES, "type": { "list": { "item": "Image"}}},
    ]
    
    dataset_infos = {
        "train": {
            "description": f"Training split of {name}",
            "schema": {
                "columns": columns
            },
            "num_rows": num_train_rows
        },
        "test": {
            "description": f"Test split of {name}",
            "schema": {
                "columns": columns
            },
            "num_rows": num_test_rows
        }
    }

    with open(output_dir/f"dataset_infos.json", "w") as fw:
        fw.write(json.dumps(dataset_infos))    
