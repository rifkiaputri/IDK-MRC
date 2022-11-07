import argparse
import os
import re
import logging
import pandas as pd
import wandb
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from tqdm import tqdm
from simpletransformers.t5 import T5Model, T5Args
from simpletransformers.question_answering import QuestionAnsweringModel
from unans_bleu_score import is_identical, unans_corpus_bleu
from qa.squad_2_0_eval import compute_f1
from text_utils import html_decode

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


def get_model_args(input_args):
    model_args = T5Args()
    model_args.num_train_epochs = input_args.num_train_epochs
    model_args.no_save = False
    model_args.no_cache = True
    model_args.evaluate_during_training = False
    model_args.early_stopping_metric_minimize = False
    model_args.save_steps = -1
    model_args.save_eval_checkpoints = False
    model_args.save_model_every_epoch = True
    model_args.reprocess_input_data = True
    model_args.overwrite_output_dir = True
    model_args.best_model_dir = input_args.out_dir + 'best_model/'
    model_args.cache_dir = input_args.cache_dir
    model_args.output_dir = input_args.out_dir
    model_args.wandb_project = input_args.wb_train_name
    model_args.train_batch_size = input_args.train_batch_size
    model_args.num_beams = input_args.num_beams
    model_args.top_k = input_args.top_k
    model_args.top_p = input_args.top_p
    model_args.num_return_sequences = input_args.num_return_seq
    model_args.max_seq_length = input_args.max_seq_length
    model_args.max_length = input_args.max_out_length
    model_args.do_sample = input_args.do_sample
    model_args.use_multiprocessing = False
    model_args.use_multiprocessing_for_evaluation = False
    model_args.thread_count = 1
    model_args.process_count = 1
    model_args.preprocess_inputs = True
    model_args.fp16 = False
    model_args.manual_seed = input_args.seed
    model_args.n_gpu = input_args.n_gpu
    return  model_args


def train_model(train_path, pretrain_name, model_args):
    train_df = pd.read_json(train_path)
    train_df.drop_duplicates(subset=['input_text'], keep=False)
    model = T5Model('mt5', pretrain_name, args=model_args)
    model.train_model(train_df)


def natural_keys(text):
        '''
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        float regex comes from https://stackoverflow.com/a/12643073/190597
        '''
        def atof(text):
            try:
                retval = float(text)
            except ValueError:
                retval = text
            return retval

        return [atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text)]


def eval_model(val_path, val_path_for_writing, wb_eval_name, wb_eval_run_name, no_bleu, model_args, smoothie, qa_models):
    from run_rule_baseline import get_sentence_tokens

    eval_df = pd.read_json(val_path)
    eval_df.drop_duplicates(subset=['input_text'], keep=False)
    eval_df_for_writing = pd.read_json(val_path_for_writing)
    eval_df_for_writing.drop_duplicates(subset=['context', 'ans'], keep=False)

    model_versions = [name for name in os.listdir(model_args.output_dir) if os.path.isdir(os.path.join(model_args.output_dir, name)) if 'epoch-2' in name]
    model_versions.sort(key=natural_keys)

    if not no_bleu:
        wandb.init(project=wb_eval_name, reinit=True)
        wandb.run.name = wb_eval_run_name
        wandb.run.save()

    print('\n**** Begin evaluation & decoding ****')
    for m_ver in model_versions:
        m_ver_item = m_ver.split('-')
        print()
        print('model ver:', m_ver)
        print('step:', m_ver_item[1])
        print('epoch:', int(m_ver_item[3]))

        target_truth = [html_decode(t) for t in eval_df['target_text'].tolist()]
        target_truth_tok = [[get_sentence_tokens(sent)] for sent in target_truth]
        contexts_clean = [html_decode(t) for t in eval_df_for_writing['context'].tolist()]
        ans_clean = [html_decode(t) for t in eval_df_for_writing['ans'].tolist()]
        
        model = T5Model('mt5', model_args.output_dir + m_ver, args=model_args)
        target_pred_candidates_unfiltered = model.predict([html_decode(t) for t in eval_df['input_text'].tolist()])
        target_pred_candidates = []
        for q_id, preds in enumerate(target_pred_candidates_unfiltered):
            preds_filt = [html_decode(pred) if pred != '' else ans_clean[q_id] for pred in preds]
            target_pred_candidates.append(preds_filt)

        if model_args.num_return_sequences > 1:
            len_qa_models = len(qa_models)
            target_pred, target_pred_tok, to_predict = [], [], []
            if len_qa_models > 0:
                majority_num = int(len_qa_models/2) + 1
                c_id = 0
                for preds, ans, context in tqdm(
                        zip(target_pred_candidates, ans_clean, contexts_clean),
                        total=len(target_pred_candidates),
                        desc='Build QA input'
                    ):
                    to_predict.append({
                        'context': context,
                        'qas': [{
                            'question': pred,
                            'id': str(c_id) + '_' + str(id)
                        } for id, pred in enumerate(preds)]
                    })
                    c_id += 1 

                qa_models_preds = {}
                for qa_model in qa_models:
                    qa_answers, probas = qa_model.predict(to_predict, n_best_size=5)
                    for answer, proba in zip(qa_answers, probas):
                        ans_candidates = answer['answer']
                        prob_candidates = proba['probability']
                        answer_text = None
                        curr_prob = 0
                        for ans_cand, prob_cand in zip(ans_candidates, prob_candidates):
                            if prob_cand > curr_prob:
                                answer_text = ans_cand
                                curr_prob = prob_cand
                        if answer_text is None:
                            answer_text = ans_candidates[0]
                            curr_prob = prob_candidates[0]
                        if curr_prob < 0.6:
                            answer_text = ''
                        if answer['id'] not in qa_models_preds:
                            qa_models_preds[answer['id']] = [answer_text]
                        else:
                            qa_models_preds[answer['id']].append(answer_text)

            c_id = 0
            for preds, ans, context in tqdm(
                    zip(target_pred_candidates, ans_clean, contexts_clean),
                    total=len(target_pred_candidates),
                    desc='Calculate similarity'
                ):

                curr_score = 0
                best_pred, best_pred_tok = '', ''
                ans_tok = get_sentence_tokens(ans)
                ans_status, identical_status = [], []

                for pred_id, pred in enumerate(preds):
                    pred_tok = get_sentence_tokens(pred)
                    if is_identical(pred_tok, ans_tok):
                        ans_status.append(True)
                        identical_status.append(True)
                        continue

                    if len_qa_models > 0:
                        qa_id = str(c_id) + '_' + str(pred_id)
                        assert len(qa_models_preds[qa_id]) == len_qa_models
                        if qa_models_preds[qa_id].count('') < majority_num:
                            answers = [ans for ans in qa_models_preds[qa_id] if ans != '']
                            same_answers = []
                            for i, a_i in enumerate(answers):
                                for j, a_j in enumerate(answers):
                                    if i == j:
                                        continue
                                    if compute_f1(a_i, a_j) < 0.6:
                                        same_answers.append(False)
                                    else:
                                        same_answers.append(True)

                            if all(same_answers):
                                ans_status.append(True)
                                identical_status.append(False)
                                continue

                    ans_status.append(False)
                    identical_status.append(False)
                    try:
                        sent_bleu = sentence_bleu([ans_tok], pred_tok, smoothing_function=smoothie)
                    except ValueError:
                        print('value pred error:', pred)
                    
                    if sent_bleu > curr_score:
                        curr_score = sent_bleu
                        best_pred = pred
                        best_pred_tok = pred_tok

                if best_pred == '':
                    assert len(ans_status) == len(preds)
                    preds_unans_only = [pred for pred_id, pred in enumerate(preds) if not ans_status[pred_id]]
                    if len(preds_unans_only) > 0:
                        best_pred = preds_unans_only[0]
                    else:
                        for is_iden, pred in zip(identical_status, preds):
                            if not is_iden:
                                best_pred = pred
                                break
                        if best_pred == '':
                            best_pred = preds[0]
                    best_pred_tok = get_sentence_tokens(best_pred)

                target_pred.append(best_pred)
                target_pred_tok.append(best_pred_tok)
                c_id += 1
        else:
            target_pred = target_pred_candidates
            target_pred_tok = [get_sentence_tokens(sent) for sent in target_pred]

        result_dict = {
            'context': contexts_clean,
            'ans question': ans_clean,
            'pred': target_pred,
            'truth': target_truth
        }

        result_df = pd.DataFrame(result_dict)
        sampling_mode = 'top-k-top-p' if model_args.top_p is not None else 'beam'
        sampling_prob = str(model_args.top_p) if model_args.top_p is not None else 'nb-' + str(model_args.num_beams)
        # result_df.to_csv(
        #     model_args.output_dir + m_ver + '/predict_result_tydi_train_' + sampling_mode + '-sample-' + str(model_args.num_return_sequences) + 
        #     '_pv2_d3_qa_xlmr_' + sampling_prob + '.csv'
        # )

        if not no_bleu:
            same_num, total = 0, 0
            target_ans_tok = [get_sentence_tokens(sent) for sent in ans_clean]
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

            wandb.log({
                'ubleu3': ubleu3,
                'ubleu4': ubleu4,
                'bleu3': bleu3,
                'bleu4': bleu4,
                'diff_ratio': diff_ratio,
                'epoch': int(m_ver_item[3]),
            })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, required=True, help='path to training data')
    parser.add_argument('--val_path', type=str, required=True, help='path to evaluation data')
    parser.add_argument('--val_path_for_writing', type=str, required=True, help='path to evaluation data for result writing purpose')
    parser.add_argument('--pretrain_name', type=str, required=True, help='pretrain name')
    parser.add_argument('--qa_model', type=str, required=True, help='path to QA model')
    parser.add_argument('--n_gpu', type=int, required=False, default=1, help='num of GPU')
    parser.add_argument('--wb_train_name', type=str, required=True, help='wandb project name for training logging')
    parser.add_argument('--wb_eval_name', type=str, required=True, help='wandb project name for eval logging')
    parser.add_argument('--wb_eval_run_name', type=str, required=True, help='wandb run name for eval logging')
    parser.add_argument('--out_dir', type=str, required=True, help='path to output directory')
    parser.add_argument('--cache_dir', type=str, required=True, help='path to cache directory')
    parser.add_argument('--seed', type=int, required=True, help='random seed')
    parser.add_argument('--num_train_epochs', type=int, required=True, help='number of training epochs')
    parser.add_argument('--train_batch_size', type=int, required=True, help='train batch size')
    parser.add_argument('--max_seq_length', type=int, required=True, help='maximum sequence length')
    parser.add_argument('--num_beams', type=int, default=1, help='num beams for decoding')
    parser.add_argument('--top_k', type=int, default=None, help='top-k for decoding')
    parser.add_argument('--top_p', type=float, default=None, help='top-p for decoding')
    parser.add_argument('--num_return_seq', type=int, default=5, help='number of return sequences for decoding')
    parser.add_argument('--max_out_length', type=int, default=50, help='maximum output length for decoding')
    parser.add_argument('--do_sample', action='store_true', help='whether to use sampling or not')
    parser.add_argument('--eval_only', action='store_true', help='whether to only run eval only or not')
    parser.add_argument('--no_bleu', action='store_true', help='whether to calculate bleu or not')
    args = parser.parse_args()
    model_args = get_model_args(args)

    # Train the model
    if not args.eval_only:
        train_model(args.train_path, args.pretrain_name, model_args)

    # Evaluate the model & calculate BLEU score
    smoothie = SmoothingFunction().method4
    qa_models = []
    seed_device_pairs = [('0', 1), ('0', 1), ('0', 2), ('0', 2), ('0', 3), ('0', 3)]  # Note: please define based on the GPU availability
    for seed, device_num in tqdm(seed_device_pairs, total=len(seed_device_pairs), desc='Initialize QA models'):
        qa_models.append(QuestionAnsweringModel('xlmroberta', args.qa_model + seed + '/best_model/', cuda_device=device_num))
    eval_model(args.val_path, args.val_path_for_writing, args.wb_eval_name, args.wb_eval_run_name, args.no_bleu, model_args, smoothie, qa_models)
