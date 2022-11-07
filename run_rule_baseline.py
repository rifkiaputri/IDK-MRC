import random
import pandas as pd
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from simpletransformers.ner import NERModel, NERArgs
from unans_bleu_score import is_identical, unans_corpus_bleu
from text_utils import html_decode, get_sentence_tokens, get_sentences_tokens


smoothie = SmoothingFunction().method4
ner_labels = [
    'B-CRD', 'B-DAT', 'B-EVT', 'B-FAC', 'B-GPE', 'B-LAN', 'B-LAW', 'B-LOC', 'B-MON',
    'B-NOR', 'B-ORD', 'B-ORG', 'B-PER', 'B-PRC', 'B-PRD', 'B-QTY', 'B-REG', 'B-TIM',
    'B-WOA', 'I-CRD', 'I-DAT', 'I-EVT', 'I-FAC', 'I-GPE', 'I-LAN', 'I-LAW', 'I-LOC',
    'I-MON', 'I-NOR', 'I-ORD', 'I-ORG', 'I-PER', 'I-PRC', 'I-PRD', 'I-QTY', 'I-REG',
    'I-TIM', 'I-WOA', 'O'
]
ner_args = NERArgs()
ner_args.silent = True
ner_args.use_multiprocessing = True
ner_args.use_multiprocessing_for_evaluation = True
ner_model = NERModel(
    "xlmroberta", "cahya/xlm-roberta-base-indonesian-NER",
    labels=ner_labels,
    args=ner_args,
    use_cuda=True,
    cuda_device=0
)
q_tags = [
    'Dimana', 'Mana', 'Siapa', 'Berapa', 'Mengapa', 'Kenapa', 'Kapan', 'Apa'
]


def get_sentences_ner(sents, orig_sents):
    results, _ = ner_model.predict(sents, split_on_space=False)
    all_ner = []
    last_tok_type = 'O'
    for idx, result in enumerate(results):
        sent_ner = []
        for item in result:
            for token, tok_type in item.items():
                if tok_type[0] == 'B' or (tok_type[0] == 'I' and last_tok_type == 'O'):
                    sent_ner.append((token, tok_type[2:]))
                elif tok_type[0] == 'I' and last_tok_type != 'O':
                    new_tok = sent_ner[-1][0] + ' ' + token
                    if new_tok not in orig_sents[idx]:
                        new_tok = sent_ner[-1][0] + token
                    sent_ner[-1] = (new_tok, tok_type[2:])
                last_tok_type = tok_type[0]
        all_ner.append(sent_ner)
    
    return all_ner


def extract_question_tag(question):
    for q_tag in q_tags:
        if q_tag in question:
            return q_tag
    for q_tag in q_tags:
        if q_tag.lower() in question:
            return q_tag.lower()
    return None


def get_random_question_tag(orig_tag):
    if orig_tag is None:
        return random.choice(q_tags), None
    while True:
        mod_tag = random.choice(q_tags)
        if mod_tag.lower() != orig_tag.lower():
            break
    if orig_tag.islower():
        mod_tag = mod_tag.lower()
    return mod_tag, orig_tag


def get_entity_replacement(q_ent, c_ents, ans):
    orig_ent_tok, orig_ent_type = q_ent
    repl_candidates = []
    for ent in c_ents:
        if orig_ent_type in ['GPE', 'LOC']:
            if ent[1] in ['GPE', 'LOC'] and orig_ent_tok.lower() != ent[0].lower():
                repl_candidates.append(ent[0])
        else:
            if ent[1] == orig_ent_type and orig_ent_tok.lower() != ent[0].lower():
                repl_candidates.append(ent[0])

    if len(repl_candidates) == 0:
        repl_candidates = [ent[0] for ent in c_ents if orig_ent_tok.lower() != ent[0].lower()]

    if len(repl_candidates) == 0:
        repl, orig = get_random_question_tag(extract_question_tag(ans))
    else:
        repl = random.choice(repl_candidates)
        orig = orig_ent_tok
        if repl.lower() == orig.lower():
            repl, orig = get_random_question_tag(extract_question_tag(ans))

    return repl, orig


def predict_rule_based(ans_questions, contexts):
    print('Extract NER in ans questions...')
    ans_questions_tokens = [get_sentence_tokens(ans, lower=False) for ans in ans_questions]
    ans_questions_ner = get_sentences_ner(ans_questions_tokens, ans_questions)
    
    print('Extract NER in context...')
    contexts_tokens, contexts_sents, contexts_sent_idxs = [], [], []
    for idx, context in enumerate(contexts):
        sents_tokens, sents = get_sentences_tokens(context)
        for sent_tokens, sent in zip(sents_tokens, sents):
            contexts_tokens.append(sent_tokens)
            contexts_sents.append(sent)
            contexts_sent_idxs.append(idx)
    contexts_ner = get_sentences_ner(contexts_tokens, contexts_sents)

    contexts_sents_ner = []
    for c_idx, ner in zip(contexts_sent_idxs, contexts_ner):
        try:
            contexts_sents_ner[c_idx] += ner
        except IndexError:
            contexts_sents_ner.append(ner)

    assert len(ans_questions_ner) == len(contexts_sents_ner)

    unans_questions = []
    for ans, context, ner_ans, ner_context in tqdm(zip(ans_questions, contexts, ans_questions_ner, contexts_sents_ner), total=len(ans_questions), desc='Generate unans'):
        if len(ner_ans) < 1:
            repl, orig = get_random_question_tag(extract_question_tag(ans))
        else:
            if len(ner_context) == 0:
                repl, orig = get_random_question_tag(extract_question_tag(ans))
            else:    
                repl, orig = get_entity_replacement(ner_ans[0], ner_context, ans)
        if orig is None:
            unans = repl + ' ' + ans
        else:
            unans = ans.replace(orig, repl, 1)
        unans_questions.append(unans)

    return unans_questions


def eval_model(val_path, output_dir):
    eval_df = pd.read_json(val_path)
    eval_df.drop_duplicates(subset=['context', 'ans'], keep=False)

    target_ans = [html_decode(t) for t in eval_df['ans'].tolist()]
    target_ans_tok = [get_sentence_tokens(sent) for sent in target_ans]
    target_truth = [html_decode(t) for t in eval_df['unans'].tolist()]
    target_truth_tok = [[get_sentence_tokens(sent)] for sent in target_truth]
    contexts = [html_decode(t) for t in eval_df['context'].tolist()]

    target_pred = predict_rule_based(target_ans, contexts)
    target_pred_tok = [get_sentence_tokens(sent) for sent in target_pred]

    result_dict = {
        'context': contexts,
        'ans question': target_ans,
        'pred': target_pred,
        'truth': target_truth
    }
    result_df = pd.DataFrame(result_dict)
    result_df.to_csv(output_dir + '/predict_result_rule_based.csv')
    
    same_num, total = 0, 0
    for pred, ans in zip(target_pred_tok, target_ans_tok):
        if is_identical(pred, ans):
            same_num += 1
        total += 1
    diff_ratio = (total - same_num) / total
    print('Diff Ratio:', diff_ratio)

    ubleu3 = unans_corpus_bleu(target_truth_tok, target_pred_tok, target_ans_tok, smoothing_function=smoothie, weights=(0.33, 0.33, 0.33, 0))
    print('UBLEU-3:', ubleu3)

    ubleu4 = unans_corpus_bleu(target_truth_tok, target_pred_tok, target_ans_tok, smoothing_function=smoothie, weights=(0.25, 0.25, 0.25, 0.25))
    print('UBLEU-4:', ubleu4)

    bleu3 = corpus_bleu(target_truth_tok, target_pred_tok, smoothing_function=smoothie, weights=(0.33, 0.33, 0.33, 0))
    print('BLEU-3:', bleu3)

    bleu4 = corpus_bleu(target_truth_tok, target_pred_tok, smoothing_function=smoothie, weights=(0.25, 0.25, 0.25, 0.25))
    print('BLEU-4:', bleu4)

    print("    ".join([str(ubleu3*100), str(ubleu4*100), str(bleu3*100), str(bleu4*100), str(diff_ratio)]))


if __name__ == '__main__':
    eval_model(
        'dataset/qg/dev_id_aligned.json',
        'outputs'
    )
