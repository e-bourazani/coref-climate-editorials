from preprocess import preprocess
from align import align_annotations
from fcoref_run import run_fcoref
from stanza_run import run_stanza
from make_salient_entities import make_salient_entities
from evaluation import run_evaluation


def main():

    preprocess()
    align_annotations()
    run_fcoref()
    run_stanza()
    make_salient_entities()
    run_evaluation()


if __name__ == "__main__":
    main()