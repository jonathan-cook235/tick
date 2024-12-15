"""
Script to generate instruction-specific checklists with an LLM.
"""
import asyncio
import json
from typing import Optional
import fire
from tqdm import tqdm
from const import ANSWER_FLAG, GEN_BATCH_SIZE, PROMPT_DIR
import logging
from inference import call_api, gcfs, get_api_client

def format_checklist(qs_raw: str) -> list:
    """
    Util for formatting generated checklists into lists.
    """
    qs = qs_raw.split("\n")
    # strip unwanted formatting
    qs = list(filter(None, [q.strip("123456789.)-\n ") for q in qs]))
    # drop any parts of response caught by the newline split that aren't questions
    qs = [q for q in qs if q.endswith("?")]
    return qs
    
def main(
    provider: str,
    subsets: Optional[str] = "non-technical,finance,stem,code",
    batch_size: int = GEN_BATCH_SIZE,
    answer_flag: str = ANSWER_FLAG,
    template_path: str = "prompt_templates/checklist_generator_template.md",
):
    logger = logging.getLogger("checklists")
    subsets = subsets.split(",")
    for subset in subsets:
        logger.info(f"Generating checklists for HEHE6 {subset}.")
        data_path = PROMPT_DIR + f"{subset}-en.jsonl"
        data = []
        with gcfs().open(data_path, "r") as f:
            for line in f:
                data.append(json.loads(line))
        # open checklist generation template
        with open(template_path, "r") as f:
            template = f.read()
        # unpack and format data into evaluator inputs
        instructions = [template.format(**{"message": row["prompt"]}) for row in data]
        # get api client and model name for provider
        # TODO: currently model is hard coded for each provider
        client, model = get_api_client(provider)
        questions = []
        for i in tqdm(range(0, len(instructions), batch_size)):
            batch_idxs = range(i, min(i + batch_size, len(instructions)))
            input_batch = [instructions[idx] for idx in batch_idxs]
            loop = asyncio.get_event_loop()
            responses = loop.run_until_complete(call_api(provider, client, model, input_batch))
            answer_locs = [r.rfind(ANSWER_FLAG) + len(ANSWER_FLAG) for r in responses]
            question_lists = [format_checklist(r[answer_locs[j] :]) for j, r in enumerate(responses)]
            questions.extend(question_lists)
        with gcfs().open(data_path, "w") as f:
            for idx, row in enumerate(data):
                row["checklist_questions"] = questions[idx]
                json.dump(row, f)
                f.write("\n")
                
if __name__ == "__main__":
    fire.Fire(main)
