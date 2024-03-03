import argparse
import sacrebleu

def calculate_bleu(ref_file, pred_file):
    ref = []
    with open(ref_file, 'r') as file:
        ref = file.readlines()
        ref = [item.strip() for item in ref]

    hi_pred = []
    with open(pred_file, 'r') as file:
        hi_pred = file.readlines()
        hi_pred = [item.strip() for item in hi_pred] 

    bleu_score = sacrebleu.corpus_bleu(hi_pred, [ref])
    return bleu_score.score

def main(args):
    bleu_score = calculate_bleu(args.ref_file, args.pred_file)
    print("BLEU Score:", bleu_score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate BLEU score.")
    parser.add_argument("--ref_file", type=str, required=True, help="Path to the reference file")
    parser.add_argument("--pred_file", type=str, required=True, help="Path to the prediction file")
    args = parser.parse_args()
    main(args)
