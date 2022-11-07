import argparse
import json
import os
import torch
import wandb
from transformers.utils import logging
from simpletransformers.question_answering import QuestionAnsweringModel
from squad_2_0_eval import evaluate, compute_f1

torch.multiprocessing.set_sharing_strategy('file_system')
logger = logging.get_logger(__name__)


def get_model(args, train_args, pretrained=None):
    if 'indobert' in args.pretrain_name:
        train_args['config'] = {
            'num_labels': 2,
            'id2label': {
                "0": "LABEL_0",
                "1": "LABEL_1"
            },
            'label2id': {
                "LABEL_0": 0,
                "LABEL_1": 1
            }
        }

    if pretrained is None:
        pretrained = args.pretrain_name

    cuda_device = -1 if 'n_gpu' in train_args and train_args['n_gpu'] > 1 else args.device_num

    return QuestionAnsweringModel(args.model_name, pretrained, args=train_args, cuda_device=cuda_device)


def eval_qa(model, val_path, eval_output_dir, filename='eval_result'):
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    result, text = model.eval_model(val_path, output_dir=eval_output_dir)
    exact_match, f1, result_squad = evaluate(val_path, eval_output_dir + 'predictions_test.json')
    result['exact'] = exact_match
    result['f1'] = f1

    # Save result
    with open(eval_output_dir + filename + '.json', 'w') as f:
        json.dump(result, f)

    with open(eval_output_dir + filename + '_text.json', 'w') as f:
        json.dump(text, f)

    with open(eval_output_dir + filename + '_squad_2.0.json', 'w') as f:
        json.dump(result_squad, f)

    return result, text, result_squad


def hasans_f1_metric(true_answers, predicted_answers):
    f1_scores = []
    for true_ans, pred_ans in zip(true_answers, predicted_answers):
        if len(true_ans) > 0:
            f1 = compute_f1(true_ans, pred_ans)
            f1_scores.append(f1)
    return 100.0 * sum(f1_scores) / len(f1_scores)


def avg_f1_metric(true_answers, predicted_answers):
    ans_f1_scores, unans_f1_scores = [], []
    for true_ans, pred_ans in zip(true_answers, predicted_answers):
        f1 = compute_f1(true_ans, pred_ans)
        if len(true_ans) > 0:
            ans_f1_scores.append(f1)
        else:
            unans_f1_scores.append(f1)
    ans_f1 = 100.0 * sum(ans_f1_scores) / len(ans_f1_scores)
    unans_f1 = 100.0 * sum(unans_f1_scores) / len(unans_f1_scores)
    return (ans_f1 + unans_f1) / 2


def train_qa(args, train_args, train_path, eval_path):
    wandb.init(project=args.wb_name)
    wandb.run.name = args.out_name
    wandb.run.save()
    model = get_model(args, train_args)
    model.train_model(train_path, eval_data=eval_path, ans_f1=hasans_f1_metric)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, help='path to training data')
    parser.add_argument('--val_path', type=str, help='path to validation data')
    parser.add_argument('--test_path', type=str, help='path to testing data')
    parser.add_argument('--model_name', type=str, required=True, choices=['bert', 'xlmroberta'], help='model name')
    parser.add_argument('--pretrain_name', type=str, required=True, help='pretrain name')
    parser.add_argument('--epoch', type=int, required=True, help='num of epoch')
    parser.add_argument('--device_num', type=int, required=False, default=0, help='cuda device num')
    parser.add_argument('--n_gpu', type=int, required=False, default=1, help='num of GPU')
    parser.add_argument('--wb_name', type=str, required=True, help='wandb project name')
    parser.add_argument('--wb_name_test', type=str, required=True, help='wandb project name for only_test')
    parser.add_argument('--out_path', type=str, required=True, help='path to output directory')
    parser.add_argument('--out_name', type=str, required=True, help='output directory name')
    parser.add_argument('--lang', type=str, required=True, help='language str id')
    parser.add_argument('--seed', type=int, required=True, help='random seed')
    parser.add_argument('--uncased', action='store_true', help='whether to use uncased data or not')
    parser.add_argument('--only_test', action='store_true', help='whether to run testing only or not')
    

    args = parser.parse_args()
    print("Running model for seed:", str(args.seed))
    print("Input file:", args.train_path)
    output_dir = args.out_path + '/' + args.wb_name + '/' + args.model_name + '/' + args.out_name + '/' + str(args.seed) + '/'
    train_args = {
        'learning_rate': 2e-5,
        'num_train_epochs': args.epoch,
        'max_seq_length': 512,
        'max_query_length': 128,
        'doc_stride': 128,
        'overwrite_output_dir': True,
        'reprocess_input_data': True,
        'train_batch_size': 16,
        'fp16': True,
        'output_dir': output_dir,
        'wandb_project': args.wb_name,
        'n_best_size': 20,
        'manual_seed': args.seed,
        'encoding': 'utf-8',
        'save_eval_checkpoints': False,
        'save_model_every_epoch': False,
        'save_steps': -1,
        'do_lower_case': args.uncased,
        'adam_epsilon': 1e-8,
        'n_gpu': args.n_gpu,
        'best_model_dir': output_dir + 'best_model/',
        'evaluate_during_training': True,
        'evaluate_during_training_steps': 2000,
        'early_stopping_metric': 'ans_f1',
        'early_stopping_metric_minimize': False,
        'no_cache': True
    }

    # Train the model
    if not args.only_test:
        train_qa(args, train_args, args.train_path, args.val_path)
    else:
        wandb.init(project=args.wb_name_test)
        wandb.run.name = args.out_name
        wandb.run.save()

    # Evaluate the model
    model = get_model(args, train_args, pretrained=output_dir + 'best_model/')
    _, _, result_squad = eval_qa(model, args.test_path, output_dir + 'test/')
    wandb.log({
        'HasAns_exact': result_squad['HasAns_exact'],
        'HasAns_f1': result_squad['HasAns_f1'],
        'NoAns_exact': result_squad['NoAns_exact'] if 'NoAns_exact' in result_squad else 0,
        'NoAns_f1': result_squad['NoAns_f1'] if 'NoAns_f1' in result_squad else 0,
        'exact': result_squad['exact'],
        'f1': result_squad['f1']
    })