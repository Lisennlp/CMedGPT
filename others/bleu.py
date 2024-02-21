from nltk.translate.bleu_score import sentence_bleu
from typing import List


def compute_blue(reference:List[list], candidate: list):
    score1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
    score2 = sentence_bleu(reference, candidate, weights=(0, 1, 0, 0))
    score3 = sentence_bleu(reference, candidate, weights=(0, 0, 1, 0))
    score4 = sentence_bleu(reference, candidate, weights=(0, 0, 0, 1))
    reference.clear()
    print(f'Bleu1-4: {[score1, score2, score3, score4]}')
    return score1, score2, score3, score4
    