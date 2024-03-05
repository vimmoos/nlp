from datasets import load_dataset
from pathlib import Path
import argparse
import json
import gzip
from multiprocess import Pool
from functools import partial

INFO = """
The version indicate the commit from the git repo:
https://github.com/sigmorphon/2023InflectionST
"""
LANGS = ["dan", "deu", "eng", "ita", "tur"]


def read_lang(
    lang,
    version="e7f1b32",
    base_url="https://github.com/sigmorphon/2023InflectionST/tree/",
):
    base_file_name = base_url + version + "/part1/data/" + lang
    data_file = {
        "train": base_file_name + ".trn",
        "test": base_file_name + ".tst",
        "val": base_file_name + ".dev",
    }
    dataset = load_dataset("text", data_files=data_file)
    return dataset, version


def extract_process(dataset, version):
    return {
        mode: {
            "data": [
                {
                    k: v
                    for k, v in zip(["lemma", "features", "inflected"], vs.split("\t"))
                }
                for vs in json.loads(http_data["text"][0])["payload"]["blob"][
                    "rawLines"
                ]
            ],
            "version": version,
            "info": INFO,
        }
        for mode, http_data in dataset.items()
    }


def dump(data, lang, base_path=Path("datasets")):
    base_path.mkdir(exist_ok=True, parents=True)
    for mode, val in data.items():
        with gzip.open(base_path / f"{lang}_{mode}.json.gz", "wt") as f:
            json.dump(val, f)


def pre_proc_data(
    lang,
    version="e7f1b32",
    base_url="https://github.com/sigmorphon/2023InflectionST/tree/",
    base_path=Path("datasets"),
):
    if isinstance(base_path, str):
        base_path = Path(base_path)
    raw = read_lang(lang, version, base_url)
    processed = extract_process(*raw)
    dump(processed, lang, base_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        default="e7f1b32",
        help="branch or commit of the github repo",
    )

    parser.add_argument(
        "-u",
        "--base_url",
        type=str,
        default="https://github.com/sigmorphon/2023InflectionST/tree/",
        help="the github repo from which the data will be taken",
    )

    parser.add_argument(
        "-p",
        "--base_path",
        type=str,
        default="datasets",
        help="Base path where to dump the processed data",
    )

    parser.add_argument(
        "-l",
        "--lang",
        required=True,
        choices=LANGS + ["all"],
        help="The language of the data, if all then all language will be processed",
    )
    args = parser.parse_args()
    base_args = vars(args)

    if args.lang != "all":
        pre_proc_data(**base_args)
        exit(0)

    del base_args["lang"]

    base_path = Path(args.base_path)
    base_path.mkdir(exist_ok=True, parents=True)

    with Pool() as p:
        p.map(partial(pre_proc_data, **base_args), LANGS)
