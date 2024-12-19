#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import glob
import json
import os
from pathlib import Path

import numpy as np

from typing import Dict, Tuple

import torch
import pytest
#import cv2
from PIL import Image, ImageDraw
from huggingface_hub import snapshot_download

from docling_eval.docling.utils import (
    map_to_records
)

from docling_core.types.doc.document import (
    DoclingDocument,
    ProvenanceItem,
    TableCell,
    TableData,
    ImageRef,
    PageItem,
    TableItem,
)

from docling_ibm_models.tableformer.utils.app_profiler import AggProfiler
import docling_ibm_models.tableformer.data_management.tf_predictor as tf_predictor
from docling_ibm_models.tableformer.data_management.tf_predictor import \
    TFPredictor

from docling_eval.docling.models.tableformer.tf_constants import tf_config

def init_tf_model() -> dict:
    r"""
    Initialize the testing environment
    """
    config = tf_config
    
    # Download models from HF
    download_path = snapshot_download(repo_id="ds4sd/docling-models", revision="v2.1.0")
    save_dir = os.path.join(download_path, "model_artifacts/tableformer/fast")

    config["model"]["save_dir"] = save_dir
    return config

def get_iocr_page(parsed_page: Dict, table_bbox:Tuple[float, float, float, float]):
    
    height = parsed_page["sanitized"]["dimension"]["height"]
    width = parsed_page["sanitized"]["dimension"]["width"]
    
    records = map_to_records(parsed_page["sanitized"]["cells"])

    cnt = 0
    
    tokens = []
    text_lines = []
    for i,rec in enumerate(records):
        tokens.append({
            "bbox": {
                "l": rec["x0"],
                "t": height-rec["y1"],
                "r": rec["x1"],
                "b": height-rec["y0"]
            },
            "text": rec["text"],
            "id": i
        })
        
        text_lines.append({
            "bbox": [rec["x0"], height-rec["y1"], rec["x1"], height-rec["y0"]],
            "text": rec["text"]
        })

        """
        if table_bbox[0]<=tokens[-1]["bbox"]["l"] and \
           table_bbox[2]>=tokens[-1]["bbox"]["r"] and \
           table_bbox[1]<=tokens[-1]["bbox"]["b"] and \
           table_bbox[3]>=tokens[-1]["bbox"]["t"]:
            cnt += 1
            print(f"text-cell [{cnt}]: ", tokens[-1]["text"], "\t", tokens[-1]["bbox"])
        """
        
    iocr_page = {
        "tokens": tokens,
        "height": height,
        "width": width
    }
    
    return iocr_page

def to_np(pil_image: Image.Image):
    # Convert to NumPy array
    np_image = np.array(pil_image)

    # Handle different formats
    if np_image.ndim == 3:  # RGB or RGBA image
        if np_image.shape[2] == 4:  # RGBA image
            # Discard alpha channel and convert to BGR
            np_image = np_image[:, :, :3]  # Keep only RGB channels

        # Convert RGB to BGR by reversing the last axis
        np_image = np_image[:, :, ::-1]

        return np_image
    else:
        raise ValueError("Unsupported image format")

def tf_predict(config,
               page_image:Image.Image,
               parsed_page: dict,
               table_bbox:Tuple[float, float, float, float],
               viz:bool = True,
               device:str = "cpu",
               num_threads:int = 2,
               image_scale:float = 1.0):
    r"""
    Test the TFPredictor
    """

    table_bboxes = [[table_bbox[0], table_bbox[1], table_bbox[2], table_bbox[3]]]
    
    iocr_page = get_iocr_page(parsed_page, table_bbox=table_bbox)
    
    iocr_page["image"] = to_np(page_image)
    iocr_page["table_bboxes"] = table_bboxes

    # Loop over the iocr_pages
    predictor = TFPredictor(config, device=device, num_threads=num_threads)
    
    tf_output = predictor.multi_table_predict(
        iocr_page,
        table_bboxes = table_bboxes,
        do_matching=True,
        correct_overlapping_cells=False,
        sort_row_col_indexes=True
    )
    #print("tf-output: ", json.dumps(tf_output, indent=2))

    table_out = tf_output[0]

    do_cell_matching = True
    
    table_cells = []
    for element in table_out["tf_responses"]:
        
        tc = TableCell.model_validate(element)
        if do_cell_matching and tc.bbox is not None:
            tc.bbox = tc.bbox.scaled(1 / image_scale)
        table_cells.append(tc)

    # Retrieving cols/rows, after post processing:
    num_rows = table_out["predict_details"]["num_rows"]
    num_cols = table_out["predict_details"]["num_cols"]
    otsl_seq = table_out["predict_details"]["prediction"][
        "rs_seq"
    ]

    table_data = TableData(num_rows=num_rows,
                                 num_cols=num_cols,
                                 table_cells=table_cells)

    return table_data


