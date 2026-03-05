import re
import string


def _normalize(text: str) -> str:
    lowered = text.lower()
    no_punc = lowered.translate(str.maketrans("", "", string.punctuation))
    no_articles = re.sub(r"\b(a|an|the)\b", " ", no_punc)
    return " ".join(no_articles.split())


def exact_match(prediction: str, gold_answers: list[str]) -> float:
    if not gold_answers:
        return 0.0
    pred = _normalize(prediction)
    return 1.0 if any(pred == _normalize(gold) for gold in gold_answers) else 0.0


def f1_score(prediction: str, gold_answers: list[str]) -> float:
    if not gold_answers:
        return 0.0

    pred_tokens = _normalize(prediction).split()
    if not pred_tokens:
        return 0.0

    def _single_f1(gold: str) -> float:
        gold_tokens = _normalize(gold).split()
        if not gold_tokens:
            return 0.0

        common = 0
        gold_counts: dict[str, int] = {}
        for token in gold_tokens:
            gold_counts[token] = gold_counts.get(token, 0) + 1

        for token in pred_tokens:
            if gold_counts.get(token, 0) > 0:
                common += 1
                gold_counts[token] -= 1

        if common == 0:
            return 0.0

        precision = common / len(pred_tokens)
        recall = common / len(gold_tokens)
        return 2 * precision * recall / (precision + recall)

    return max(_single_f1(gold) for gold in gold_answers)


def recall_at_k(relevant_doc_id: str, ranked_doc_ids: list[str]) -> float:
    return 1.0 if relevant_doc_id in ranked_doc_ids else 0.0


def reciprocal_rank(relevant_doc_id: str, ranked_doc_ids: list[str]) -> float:
    for index, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id == relevant_doc_id:
            return 1.0 / index
    return 0.0
