import argparse
import copy
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import pypdfium2 as pdfium
from tqdm import tqdm  # type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

from bs4 import BeautifulSoup  # type: ignore
from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.document import (
    DoclingDocument,
    ImageRef,
    PageItem,
    PictureItem,
    ProvenanceItem,
    TableCell,
    TableData,
    TableItem,
)
from docling_core.types.doc.labels import DocItemLabel
from docling_parse.pdf_parsers import pdf_parser_v2
from PIL import Image as PILImage

from docling_eval.benchmarks.constants import BenchMarkColumns
from docling_eval.benchmarks.utils import write_datasets_info
from docling_eval.docling.conversion import create_converter
from docling_eval.docling.models.tableformer.tf_model_prediction import (
    init_tf_model,
    tf_predict,
)
from docling_eval.docling.utils import (
    crop_bounding_box,
    docling_version,
    extract_images,
    get_binary,
    save_shard_to_disk,
)


def get_page_cells(filename: str):

    parser = pdf_parser_v2("fatal")

    try:
        key = "key"
        parser.load_document(key=key, filename=filename)

        parsed_doc = parser.parse_pdf_from_key(key=key)

        parser.unload_document(key)
        return parsed_doc

    except Exception as exc:
        logging.error(exc)

    return None


def parse_html_table(table_html):
    soup = BeautifulSoup(table_html, "html.parser")
    table = soup.find("table") or soup  # Ensure table context
    rows = table.find_all("tr")

    max_cols = 0
    for row in rows:
        cols = row.find_all(["td", "th"])
        max_cols = max(max_cols, len(cols))  # Determine maximum columns

    # Create grid to track cell positions
    grid = [[None for _ in range(max_cols * 2)] for _ in range(len(rows) * 2)]

    for row_idx, row in enumerate(rows):
        col_idx = 0  # Start from first column
        for cell in row.find_all(["td", "th"]):
            # Skip over filled grid positions (handle previous rowspan/colspan)
            while grid[row_idx][col_idx] is not None:
                col_idx += 1

            # Get text, rowspan, and colspan
            text = cell.get_text(strip=True)
            rowspan = int(cell.get("rowspan", 1))
            colspan = int(cell.get("colspan", 1))

            # Fill grid positions and yield (row, column, text)
            for r in range(rowspan):
                for c in range(colspan):
                    grid[row_idx + r][col_idx + c] = text

            # print(f"Row: {row_idx + 1}, Col: {col_idx + 1}, Text: {text}")
            yield row_idx, col_idx, rowspan, colspan, text

            col_idx += colspan  # Move to next column after colspan


def update(doc: DoclingDocument, annots: Dict, page_width: float, page_height: float):

    label = annots["category"]

    min_x = annots["coordinates"][0]["x"]
    max_x = annots["coordinates"][0]["x"]

    min_y = annots["coordinates"][0]["y"]
    max_y = annots["coordinates"][0]["y"]

    for coor in annots["coordinates"]:
        min_x = min(min_x, coor["x"])
        max_x = max(max_x, coor["x"])

        min_y = min(min_y, coor["y"])
        max_y = max(max_y, coor["y"])

    text = annots["content"]["text"]
    html = annots["content"]["html"]

    bbox = BoundingBox(
        l=min_x * page_width,
        r=max_x * page_width,
        b=min_y * page_height,
        t=max_y * page_height,
        coord_origin=CoordOrigin.TOPLEFT,
    )

    prov = ProvenanceItem(page_no=1, bbox=bbox, charspan=(0, len(text)))

    if label == "Header":
        doc.add_text(label=DocItemLabel.PAGE_HEADER, text=text, orig=text, prov=prov)

    elif label == "Footer":
        doc.add_text(label=DocItemLabel.PAGE_FOOTER, text=text, orig=text, prov=prov)

    elif label == "Paragraph":
        doc.add_text(label=DocItemLabel.TEXT, text=text, orig=text, prov=prov)

    elif label == "Index":

        rows = text.split("\n")

        num_rows = len(rows)
        num_cols = 2

        row_span = 1
        col_span = 1

        cells = []
        for row_idx, row in enumerate(rows):

            parts = row.split(" ")

            col_idx = 0
            cell = TableCell(
                row_span=row_span,
                col_span=col_span,
                start_row_offset_idx=row_idx,
                end_row_offset_idx=row_idx + row_span,
                start_col_offset_idx=col_idx,
                end_col_offset_idx=col_idx + col_span,
                text=" ".join(parts[:-1]),
            )
            cells.append(cell)

            col_idx = 1
            cell = TableCell(
                row_span=row_span,
                col_span=col_span,
                start_row_offset_idx=row_idx,
                end_row_offset_idx=row_idx + row_span,
                start_col_offset_idx=col_idx,
                end_col_offset_idx=col_idx + col_span,
                text=parts[-1],
            )
            cells.append(cell)

        table_data = TableData(num_rows=num_rows, num_cols=num_cols, table_cells=cells)
        doc.add_table(
            data=table_data, caption=None, prov=prov, label=DocItemLabel.DOCUMENT_INDEX
        )

    elif label == "List":
        # doc.add_list_item(text=text, orig=text, prov=prov)
        doc.add_text(label=DocItemLabel.TEXT, text=text, orig=text, prov=prov)

    elif label == "Caption":
        doc.add_text(label=DocItemLabel.CAPTION, text=text, orig=text, prov=prov)

    elif label == "Equation":
        doc.add_text(label=DocItemLabel.FORMULA, text=text, orig=text, prov=prov)

    elif label == "Figure":
        doc.add_picture(prov=prov)

    elif label == "Table":

        num_rows = 0
        num_cols = 0

        cells = []
        for row_idx, col_idx, rowspan, colspan, text in parse_html_table(
            table_html=html
        ):
            cell = TableCell(
                row_span=rowspan,
                col_span=colspan,
                start_row_offset_idx=row_idx,
                end_row_offset_idx=row_idx + rowspan,
                start_col_offset_idx=col_idx,
                end_col_offset_idx=col_idx + colspan,
                text=text,
            )
            cells.append(cell)

            num_rows = max(row_idx + rowspan, num_rows)
            num_cols = max(col_idx + colspan, num_cols)

        table_data = TableData(num_rows=num_rows, num_cols=num_cols, table_cells=cells)
        doc.add_table(data=table_data, caption=None, prov=prov)

    elif label == "Chart":
        doc.add_picture(prov=prov)

    elif label == "Footnote":
        doc.add_text(label=DocItemLabel.FOOTNOTE, text=text, orig=text, prov=prov)

    elif label == "Heading1":
        doc.add_heading(text=text, orig=text, level=1, prov=prov)

    else:
        return


def create_dpbench_e2e_dataset(
    dpbench_dir: Path, output_dir: Path, image_scale: float = 1.0
):

    # Create Converter
    doc_converter = create_converter(
        artifacts_path=output_dir / "artifacts", page_image_scale=image_scale
    )

    # load the groundtruth
    with open(dpbench_dir / f"dataset/reference.json", "r") as fr:
        gt = json.load(fr)

    records = []

    for filename, annots in tqdm(
        gt.items(),
        desc="Processing files for DP-Bench with end-to-end",
        total=len(gt),
        ncols=128,
    ):

        pdf_path = dpbench_dir / f"dataset/pdfs/{filename}"
        # logging.info(f"\n\n===============================\n\nfile: {pdf_path}\n\n")

        conv_results = doc_converter.convert(source=pdf_path, raises_on_error=True)

        pred_doc, pictures, page_images = extract_images(
            conv_results.document,
            pictures_column=BenchMarkColumns.PICTURES.value,  # pictures_column,
            page_images_column=BenchMarkColumns.PAGE_IMAGES.value,  # page_images_column,
        )

        true_doc = DoclingDocument(name=f"ground-truth {filename}")
        true_doc.pages = pred_doc.pages

        page_width = pred_doc.pages[1].size.width
        page_height = pred_doc.pages[1].size.height

        # logging.info(f"w={page_width}, h={page_height}")

        for elem in annots["elements"]:
            update(true_doc, elem, page_width=page_width, page_height=page_height)

        record = {
            BenchMarkColumns.DOCLING_VERSION: docling_version(),
            BenchMarkColumns.STATUS: str(conv_results.status),
            BenchMarkColumns.DOC_ID: str(filename),
            BenchMarkColumns.GROUNDTRUTH: json.dumps(true_doc.export_to_dict()),
            BenchMarkColumns.PREDICTION: json.dumps(pred_doc.export_to_dict()),
            BenchMarkColumns.ORIGINAL: get_binary(pdf_path),
            BenchMarkColumns.MIMETYPE: "application/pdf",
            BenchMarkColumns.PAGE_IMAGES: page_images,
            BenchMarkColumns.PICTURES: pictures,
        }
        records.append(record)

    test_dir = output_dir / "test"
    os.makedirs(test_dir, exist_ok=True)

    save_shard_to_disk(items=records, dataset_path=test_dir)

    write_datasets_info(
        name="DPBench: end-to-end",
        output_dir=output_dir,
        num_train_rows=0,
        num_test_rows=len(records),
    )


def create_dpbench_layout_dataset(
    dpbench_dir: Path, output_dir: Path, image_scale: float = 1.0
):
    create_dpbench_e2e_dataset(
        dpbench_dir=dpbench_dir, output_dir=output_dir, image_scale=image_scale
    )


def create_dpbench_tableformer_dataset(
    dpbench_dir: Path, output_dir: Path, image_scale: float = 1.0
):

    tf_config = init_tf_model()

    # load the groundtruth
    with open(dpbench_dir / f"dataset/reference.json", "r") as fr:
        gt = json.load(fr)

    records = []

    for filename, annots in tqdm(
        gt.items(),
        desc="Processing files for DP-Bench with TableFormer",
        total=len(gt),
        ncols=128,
    ):

        pdf_path = dpbench_dir / f"dataset/pdfs/{filename}"
        # logging.info(f"\n\n===============================\n\nfile: {pdf_path}\n\n")

        parsed_doc = get_page_cells(str(pdf_path))
        if parsed_doc == None:
            logging.error("could not parse pdf-file")
            continue

        true_doc = DoclingDocument(name=f"ground-truth {filename}")

        pdf = pdfium.PdfDocument(pdf_path)
        assert len(pdf) == 1, "len(pdf)==1"

        # Get page dimensions
        page = pdf.get_page(0)
        page_width, page_height = page.get_width(), page.get_height()

        # add the elements
        for elem in annots["elements"]:
            update(true_doc, elem, page_width=page_width, page_height=page_height)

        # add the pages
        page_images: List[PILImage.Image] = []

        pdf = pdfium.PdfDocument(pdf_path)
        for page_index in range(len(pdf)):
            # Get the page
            page = pdf.get_page(page_index)

            # Get page dimensions
            width, height = page.get_width(), page.get_height()

            # Render the page to an image
            image_scale = 1.0  # Adjust scale if needed
            page_image = page.render(scale=image_scale).to_pil()

            page_images.append(page_image)

            # Close the page to free resources
            page.close()

            image_ref = ImageRef(
                mimetype="image/png",
                dpi=round(72 * image_scale),
                size=Size(
                    width=float(page_image.width), height=float(page_image.height)
                ),
                uri=Path(f"{BenchMarkColumns.PAGE_IMAGES}/{page_index}"),
            )
            page_item = PageItem(
                page_no=page_index + 1,
                size=Size(width=float(width), height=float(height)),
                image=image_ref,
            )

            true_doc.pages[page_index + 1] = page_item

        # add the pictures
        pictures: List[PILImage.Image] = []
        for item, level in true_doc.iterate_items():
            if isinstance(item, PictureItem):
                for prov in item.prov:
                    page_image = page_images[prov.page_no - 1]
                    # page_image.show()

                    picture_image = crop_bounding_box(
                        page_image=page_image,
                        page=true_doc.pages[prov.page_no],
                        bbox=prov.bbox,
                    )
                    # picture_image.show()

                    image_ref = ImageRef(
                        mimetype="image/png",
                        dpi=round(72 * image_scale),
                        size=Size(
                            width=float(picture_image.width),
                            height=float(picture_image.height),
                        ),
                        uri=Path(f"{BenchMarkColumns.PICTURES}/{len(pictures)}"),
                    )
                    item.image = image_ref

                    picture_json = item.model_dump(
                        mode="json", by_alias=True, exclude_none=True
                    )
                    # print(json.dumps(picture_json, indent=2))
                    pictures.append(picture_image)

        # deep copy of the true-document
        pred_doc = copy.deepcopy(true_doc)

        # replace the groundtruth tables with predictions from TableFormer
        for item, level in pred_doc.iterate_items():
            if isinstance(item, TableItem):
                for prov in item.prov:

                    # md = item.export_to_markdown()
                    # print("groundtruth: \n\n", md)

                    page_image = page_images[prov.page_no - 1]
                    # page_image.show()

                    table_image = crop_bounding_box(
                        page_image=page_image,
                        page=pred_doc.pages[prov.page_no],
                        bbox=prov.bbox,
                    )
                    table_json = item.model_dump(
                        mode="json", by_alias=True, exclude_none=True
                    )
                    # print(json.dumps(table_json, indent=2))
                    # table_image.show()

                    table_data = tf_predict(
                        config=tf_config,
                        page_image=page_image,
                        parsed_page=parsed_doc["pages"][prov.page_no - 1],
                        table_bbox=(prov.bbox.l, prov.bbox.b, prov.bbox.r, prov.bbox.t),
                    )

                    item.data = table_data

                    # md = item.export_to_markdown()
                    # print("prediction from table-former: \n\n", md)

                    # input("continue")

        record = {
            BenchMarkColumns.DOCLING_VERSION: docling_version(),
            BenchMarkColumns.STATUS: "SUCCESS",
            BenchMarkColumns.DOC_ID: str(filename),
            BenchMarkColumns.GROUNDTRUTH: json.dumps(true_doc.export_to_dict()),
            BenchMarkColumns.PREDICTION: json.dumps(pred_doc.export_to_dict()),
            BenchMarkColumns.ORIGINAL: get_binary(pdf_path),
            BenchMarkColumns.MIMETYPE: "application/pdf",
            BenchMarkColumns.PAGE_IMAGES: page_images,
            BenchMarkColumns.PICTURES: pictures,
        }
        records.append(record)

    test_dir = output_dir / "test"
    os.makedirs(test_dir, exist_ok=True)

    save_shard_to_disk(items=records, dataset_path=test_dir)

    write_datasets_info(
        name="DPBench: tableformer",
        output_dir=output_dir,
        num_train_rows=0,
        num_test_rows=len(records),
    )


def parse_arguments():
    """Parse arguments for DP-Bench parsing."""

    parser = argparse.ArgumentParser(
        description="Process DP-Bench benchmark from directory into HF dataset."
    )
    parser.add_argument(
        "-i",
        "--dpbench-directory",
        help="input directory with documents",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        help="output directory with shards",
        required=False,
        default="./benchmarks/dpbench",
    )
    parser.add_argument(
        "-s",
        "--image-scale",
        help="image-scale of the pages",
        required=False,
        default=1.0,
    )
    parser.add_argument(
        "-m",
        "--mode",
        help="mode of dataset",
        required=False,
        choices=["end-2-end", "table", "formula", "all"],
    )
    args = parser.parse_args()

    return (
        Path(args.dpbench_directory),
        Path(args.output_directory),
        float(args.image_scale),
        args.mode,
    )


def main():

    dpbench_dir, output_dir, image_scale, mode = parse_arguments()

    # Create the directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    odir_e2e = Path(output_dir) / "end_to_end"
    odir_tab = Path(output_dir) / "tables"
    odir_eqn = Path(output_dir) / "formulas"

    os.makedirs(odir_e2e, exist_ok=True)
    os.makedirs(odir_tab, exist_ok=True)
    # os.makedirs(odir_eqn, exist_ok=True)

    for _ in ["test", "train"]:
        os.makedirs(odir_e2e / _, exist_ok=True)
        os.makedirs(odir_tab / _, exist_ok=True)

    if mode == "end-2-end":
        create_dpbench_e2e_dataset(
            dpbench_dir=dpbench_dir, output_dir=odir_e2e, image_scale=image_scale
        )

    elif mode == "table":
        create_dpbench_tableformer_dataset(
            dpbench_dir=dpbench_dir, output_dir=odir_tab, image_scale=image_scale
        )

    elif mode == "all":
        create_dpbench_e2e_dataset(
            dpbench_dir=dpbench_dir, output_dir=odir_e2e, image_scale=image_scale
        )

        create_dpbench_tableformer_dataset(
            dpbench_dir=dpbench_dir, output_dir=odir_tab, image_scale=image_scale
        )


if __name__ == "__main__":
    main()
