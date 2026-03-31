import argparse
from preprocess import preprocess
from align import align_annotations
from fcoref_run import run_fcoref
from stanza_run import run_stanza
from make_salient_entities import make_salient_entities
from evaluation import run_evaluation


def main(step="all"):

    if step in ["all", "preprocess"]:
        preprocess()

    if step in ["all", "align"]:
        align_annotations()

    if step in ["all", "fcoref"]:
        run_fcoref()

    if step in ["all", "stanza"]:
        run_stanza()

    if step in ["all", "salient"]:
        make_salient_entities()

    if step in ["all", "evaluate"]:
        run_evaluation()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, default="all")
    args = parser.parse_args()

    main(step=args.step)