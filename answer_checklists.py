"""
Script for using an LLM to answer eval checklists.
"""
import asyncio
import json
from collections import defaultdict
from typing import Optional
import fire
from tqdm import tqdm
import logging
from const import ANSWER_FLAG, CHECKLIST_DIR, GEN_BATCH_SIZE
from inference import call_api, gcfs, get_api_client

def main(
    data_file: str,
    provider: str,
    batch_size: int = GEN_BATCH_SIZE,
    num_evals: int = 1,
    num_generations: int = 1,
    answer_flag: str = ANSWER_FLAG,
    question_key: str = "checklist_questions",
    answer_key: str = "checklist_answers_generated",
    template_file: str = "checklist_evaluator_template.md",
    save_file: Optional[str] = None,
):
    logger = logging.getLogger("checklists")
    # open data w/ instructions, generations and unanswered checklists
    logger.info("Answering checklists for HEHE data.")
    if not data_file.endswith(".json"):
        raise ValueError("Only json files supported.")
    data_path = CHECKLIST_DIR + data_file
    with gcfs().open(data_path, "r") as f:
        data = json.load(f)
    # open checklist eval template
    with open("prompt_templates/" + template_file, "r") as f:
        template = f.read()
    # use correct response and answer keys if evaluating refinements
    if num_refinements > 0:
        response_key = f"r_{num_refinements}"
        answer_key = f"a_{num_refinements}"
    else:
        response_key = "generation"
    # unpack and format data into evaluator inputs
    inputs = []
    for idx, row in data.items():
        generations = row[response_key]
        if num_evals == 1:
            generations = [generations]
        for g_idx, generation in enumerate(generations):
            inputs.extend(
                [
                    (
                        g_idx,
                        template.format(
                            **{"message": row["instruction"], "generation": generation, "question": question}
                        ),
                        idx,
                    )
                    for question in row[question_key]
                ]
            )
    # get api client and model name for provider
    # TODO: currently model is hard coded for each provider
    client, model = get_api_client(provider)
    # iterative over input batches and generate answers
    score_dict = defaultdict(lambda: defaultdict(list))
    for i in tqdm(range(0, len(inputs), batch_size)):
        batch_idxs = range(i, min(i + batch_size, len(inputs)))
        g_idx_batch = [inputs[idx][0] for idx in batch_idxs]
        input_batch = [inputs[idx][1] for idx in batch_idxs]
        idx_batch = [inputs[idx][2] for idx in batch_idxs]
        if num_evals > 1:
            # majority vote evals
            responses = asyncio.run(call_api(provider, client, model, input_batch, n))
            for idx, response_batch in enumerate(responses):
                answer_locs = [response.rfind(ANSWER_FLAG) + len(ANSWER_FLAG) for response in response_batch]
                score_batch = [
                    1 if "YES" in response[answer_locs[j] :] else 0 for j, response in enumerate(response_batch)
                ]
                score_dict[idx_batch[idx]][g_idx_batch[idx]].append(max(set(score_batch), key=score_batch.count))
        else:
            # single eval
            responses = asyncio.run(call_api(provider, client, model, input_batch))
            answer_locs = [response.rfind(ANSWER_FLAG) + len(ANSWER_FLAG) for response in responses]
            for idx, response in enumerate(responses):
                score = 1 if "YES" in response[answer_locs[idx] :] else 0
                score_dict[idx_batch[idx]][g_idx_batch[idx]].append(score)
    # add a column to data with the generated evals
    for idx, values in score_dict.items():
        data[idx][answer_key] = []
        for g_idx, scores in values.items():
            data[idx][answer_key].append(scores)
    if save_file:
        save_path = CHECKLIST_DIR + save_file
    else:
        save_path = data_path
    with gcfs().open(save_path, "w") as f:
        json.dump(data, f)
        
if __name__ == "__main__":
    fire.Fire(main)
