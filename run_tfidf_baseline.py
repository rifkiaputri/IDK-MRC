import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from sklearn.feature_extraction.text import TfidfVectorizer
from unans_bleu_score import is_identical, unans_corpus_bleu
from sklearn.metrics.pairwise import cosine_similarity
from text_utils import html_decode, get_sentence_tokens

smoothie = SmoothingFunction().method4


def create_tfidf_features(corpus, max_features=5000, max_df=0.95, min_df=2):
    """
    Creates a tf-idf matrix for the `corpus` using sklearn.
    """
    tfidf_vectorizor = TfidfVectorizer(decode_error='replace', strip_accents='unicode', analyzer='word',
                                       stop_words='english', ngram_range=(1, 1), max_features=max_features,
                                       norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True,
                                       max_df=max_df, min_df=min_df)
    X = tfidf_vectorizor.fit_transform(corpus)
    print('tf-idf matrix successfully created.')
    return X, tfidf_vectorizor


def calculate_similarity(X, vectorizor, query, top_k=5):
    """
    Vectorizes the `query` via `vectorizor` and calculates the cosine similarity of
    the `query` and `X` (all the documents) and returns the `top_k` similar documents.
    """
    # Vectorize the query to the same length as documents
    query_vec = vectorizor.transform(query)
    # Compute the cosine similarity between query_vec and all the documents
    cosine_similarities = cosine_similarity(X,query_vec).flatten()
    # Sort the similar documents from the most similar to less similar and return the indices
    most_similar_doc_indices = np.argsort(cosine_similarities, axis=0)[:-top_k-1:-1]
    return (most_similar_doc_indices, cosine_similarities)


def predict_tf_idf(ans_questions):
    """
    https://sci2lab.github.io/ml_tutorial/tfidf/#create_tfidf_features
    """
    print('Creating answerable tf-idf matrix...')
    X, v = create_tfidf_features(ans_questions)
    
    unans_questions = []
    for ans in tqdm(ans_questions, desc='Searching unans'):
        ans_tokens = get_sentence_tokens(ans)
        sim_indices, _ = calculate_similarity(X, v, [ans])
        best_unans = ans
        for idx in sim_indices:
            unans = ans_questions[idx]
            unans_tokens = get_sentence_tokens(unans)
            if not is_identical(unans_tokens, ans_tokens):
                best_unans = unans
            if best_unans != ans:
                break
        unans_questions.append(best_unans)
    return unans_questions


def eval_model(val_path, output_dir):
    eval_df = pd.read_json(val_path)
    eval_df.drop_duplicates(subset=['context', 'ans'], keep=False)

    target_ans = [html_decode(t) for t in eval_df['ans'].tolist()]
    target_ans_tok = [get_sentence_tokens(sent) for sent in target_ans]
    
    target_truth = [html_decode(t) for t in eval_df['unans'].tolist()]
    target_truth_tok = [[get_sentence_tokens(sent)] for sent in target_truth]

    target_pred = predict_tf_idf(target_ans)
    target_pred_tok = [get_sentence_tokens(sent) for sent in target_pred]

    result_dict = {
        'context': [html_decode(t) for t in eval_df['context'].tolist()],
        'ans question': target_ans,
        'pred': target_pred,
        'truth': target_truth
    }
    result_df = pd.DataFrame(result_dict)
    result_df.to_csv(output_dir + '/predict_result_tf-idf.csv')
    
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

    print("\t".join([str(ubleu3*100), str(ubleu4*100), str(bleu3*100), str(bleu4*100), str(diff_ratio)]))


if __name__ == '__main__':
    eval_model(
        'dataset/qg/dev_id_aligned.json',
        'outputs'
    )
