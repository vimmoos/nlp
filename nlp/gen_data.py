from datasets import load_dataset
from pathlib import Path
import argparse
import json
import gzip
from multiprocess import Pool
from functools import partial
from typing import Tuple, Dict, Any, List

DEFAULTS = {
    "version": "e7f1b32",
    "base_url": "https://raw.github.com/sigmorphon/2023InflectionST/",
    "base_path": "datasets",
}
INFO = """
The version indicate the commit from the git repo:
https://github.com/sigmorphon/2023InflectionST
"""


ROMANCE = ["ita", "spa", "fra"]
LANGS = ["tur", *ROMANCE]

MODE_NAMIG = {
    "train": ".trn",
    "test": ".tst",
    "val": ".dev",
}


def read_lang(lang: str, version: str, base_url: str) -> Tuple[object, str]:
    """Reads language data from a given URL structure.

    Args:
        lang: The language code.
        version: Version identifier (e.g., branch name or commit hash).
        base_url: The base URL where language files are located.

    Returns:
        Dict[str, Any]:
            * A Hugging Face Dataset object containing the loaded data.
            * The version string.
    """

    base_file_name = base_url + version + "/part1/data/" + lang

    # Construct filenames for 'train', 'test', and 'val' splits
    data_file = {k: base_file_name + v for k, v in MODE_NAMIG.items()}

    dataset = load_dataset("text", data_files=data_file)
    return dataset, version


def extract_process(dataset: Dict[str, Any], version: str) -> Dict[str, Any]:
    """Extracts and restructures data for the model.

    Args:
        dataset: A dictionary containing data for different modes (e.g., 'train', 'test').
        version: Version identifier (e.g., branch name or commit hash).

    Returns:
        Dict[str, Any]: A processed dictionary with the following structure:
            {
                'mode': {
                    'data': [...],  # List of processed data examples
                    'version': str,
                    'info': str
                },
                ... (for each mode)
            }
    """

    return {
        mode: {
            "data": [
                {
                    k: v
                    for k, v in zip(
                        ["lemma", "features", "inflected"],
                        line["text"].split("\t"),
                    )
                }
                for line in data
            ],
            "version": version,
            "info": INFO,
        }
        for mode, data in dataset.items()
    }


def dump(data: Dict[str, Any], lang: str, base_path: Path):
    """Serializes and compresses data into JSON files.

    Args:
        data: A dictionary containing the data to be dumped. It's expected
            to have different modes ('train', 'test',  'val') as keys.
        lang: The language code.
        base_path: The base directory where the files will be saved.
    """

    base_path.mkdir(
        exist_ok=True, parents=True
    )  # Create the directory if needed

    for mode, val in data.items():
        filepath = (
            base_path / f"{lang}_{mode}.json.gz"
        )  # Construct output filepath

        with gzip.open(filepath, "wt") as f:  # Open file with gzip compression
            json.dump(val, f)


def create_multi_lang(
    llang: List[str], version: str, base_url: str, base_path: str
):
    datas = [
        extract_process(*read_lang(lang, version, base_url)) for lang in llang
    ]
    # merge all the data
    result = datas[0].copy()

    for d in datas[1:]:
        for key in d.keys():
            result[key]["data"].extend(d[key]["data"])

    base_path = Path(base_path)
    dump(result, "multi_romance", base_path)


def pre_proc_data(lang: str, version: str, base_url: str, base_path: str):
    """Coordinates the data preprocessing pipeline for a single language.

    Args:
        lang: The language code.
        version: Version identifier (e.g., branch name or commit hash).
        base_url: The base URL where raw data files are located.
        base_path: The base directory where processed data will be saved.
    """

    base_path = Path(base_path)  # Convert string paths to Path objects

    raw_data, raw_version = read_lang(lang, version, base_url)  # Read raw data
    processed_data = extract_process(raw_data, raw_version)  # Process the data
    dump(processed_data, lang, base_path)  # Save processed data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        default=DEFAULTS["version"],
        help="branch or commit of the github repo",
    )

    parser.add_argument(
        "-u",
        "--base_url",
        type=str,
        default=DEFAULTS["base_url"],
        help="the github repo from which the data will be taken",
    )

    parser.add_argument(
        "-p",
        "--base_path",
        type=str,
        default=DEFAULTS["base_path"],
        help="Base path where to dump the processed data",
    )

    parser.add_argument(
        "-l",
        "--lang",
        required=True,
        choices=LANGS + ["all", "multi_lang"],
        help="The language of the data, if all then all language will be processed, if multi_lang a dataset containing all the available languages will be created",
    )
    args = parser.parse_args()
    base_args = vars(args)

    if args.lang == "multi_lang":
        del base_args["lang"]
        create_multi_lang(**base_args, llang=ROMANCE)
        exit(0)

    if args.lang != "all":
        pre_proc_data(**base_args)
        exit(0)

    del base_args["lang"]

    base_path = Path(args.base_path)
    base_path.mkdir(exist_ok=True, parents=True)

    with Pool() as p:
        p.map(partial(pre_proc_data, **base_args), LANGS)

    create_multi_lang(**base_args, llang=ROMANCE)
