import re
import argparse
import json
import numpy as np
import torch.nn as nn
import language_evaluation
from multiprocessing import Pool
from tqdm import tqdm

import sys
sys.path.append(".")
from utils.gpt_eval import *


class evaluation_suit():
    def __init__(self):
        self.language_eval = language_evaluation.CocoEvaluator(coco_types=["BLEU", "ROUGE_L", "CIDEr", "METEOR", "SPICE"])
        self.GPT = []
        self.accuracy = {"answer": [], "GT": []}
        self.language = {"answer": [], "GT": []}

    def eval_acc(self):
        scores = []
        for i in tqdm(range(len(self.accuracy["answer"]))):
            answer = self.accuracy["answer"][i]
            GT = self.accuracy["GT"][i]
            if answer == GT:
                scores.append(1.0)
            else:
                scores.append(0.0)

        scores = sum(scores) / len(scores)
        return scores

    def eval_chatGPT(self, data):
        with Pool(16) as p:  # Change the number based on your CPU cores
            scores_all = p.map(gpt_forward, data)
        

        scores = [x for x in scores_all if x != -1]
        delted = len(scores_all) - len(scores)
        print(f"Deleted {delted} invalid samples")
        scores = list(map(float, scores))
        
        scores = sum(scores) / len(scores)
        return scores

    def eval_language(self):
        """
        return the dict evaluation results
        """
        answer = self.language["answer"]
        GT = self.language["GT"]
        chunk_size = 500
        n_total = len(answer)

        # Fast path: single chunk
        if n_total <= chunk_size:
            results = self.language_eval.run_evaluation(answer, GT)
            return {f"val/{k}": float(v) for k, v in results.items()}

        # Split and combine afterwards (weighted by chunk size)
        results_accumulator = {}
        total_items = 0

        for i in range(0, n_total, chunk_size):
            answer_split = answer[i:i + chunk_size]
            GT_split = GT[i:i + chunk_size]
            chunk_len = len(answer_split)
            total_items += chunk_len

            results_gen = self.language_eval.run_evaluation(answer_split, GT_split)

            # Accumulate weighted sums
            for k, v in results_gen.items():
                results_accumulator[k] = results_accumulator.get(k, 0.0) + float(v) * chunk_len

        # Weighted mean over all items
        results_gen_dict = {f"val/{k}": v_sum / total_items for k, v_sum in results_accumulator.items()}
        return results_gen_dict


    def forward(self, answer, GT):
        answer = answer.replace('A: ', '')
        GT = GT.replace('A: ', '')
        answer = answer.replace(' <|im_end|>', '')
        GT = GT.replace(' <|im_end|>', '')
        self.accuracy["answer"].append(answer)
        self.accuracy["GT"].append(GT)
        self.GPT.append((answer, GT))
        self.language["GT"].append(GT)
        self.language["answer"].append(answer)
        # self.match["GPT"].append((answer, GT))

            
    def evaluation(self):
        print("evaluation start!")
        scores = {}
        try:
            print("accuracy evaluation")
            scores["accuracy"] = self.eval_acc()
        except:
            print("Error in accuracy evaluation")
            scores["accuracy"] = 0.0
        try:
            print("chatGPT evaluation")
            scores["chatgpt"] = self.eval_chatGPT(self.GPT)
        except:
            print("Error in chatGPT evaluation")
            scores["chatgpt"] = 0.0
        try:
            print("language evaluation")
            scores["language"] = self.eval_language()
        except:
            print("Error in language evaluation")
            scores["language"] = {}

        return scores

if __name__ == '__main__':
    # get args
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--root_path1', type=str, default="outputs/simlingo/predictions/language_preds_cot_rank_0.json", help='path to prediction file')
    args = parser.parse_args()
    
    with open(args.root_path1, 'r') as f :
        pred_file = json.load(f)
        
    gt = [preds[1].replace('A: ','') for preds in pred_file]
    pred = [preds[0].replace('A: ','') for preds in pred_file]

    print("Number of predictions: ", len(pred))
    print("Number of ground truths: ", len(gt))
    
    evaluation = evaluation_suit()
    
    for i in range(len(pred)):
        evaluation.forward(pred[i], gt[i])

    output = evaluation.evaluation()

    print("accuracy score: ", output["accuracy"])
    print("chatgpt score: ", output["chatgpt"])
    print("language score: ", output["language"])
    
    # save the evaluation results
    save_path = args.root_path1.replace(".json", "_metrics_gpt-4o-2024-08-06.json")
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=4)