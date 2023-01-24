import argparse
from collections import defaultdict
import logging
import os
import random

import jsonlines
import numpy as np
from scipy.stats import sem
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed

CAUSAL_LM_CLASS_PROMPT = "Title: {}\n\nText: {}\n\nTL;DR: {}\n\nQuestion: Is the above an excellent summary of the given text? An excellent summary is coherent, accurate, concise, and detailed. Answer with Yes or No.\n\nAnswer:"
POSITIVE = " Yes"
NEGATIVE = " No"

os.environ["CXX"] = "g++"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reward model prediction")

    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="A jsonl file containing the test data.",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="A jsonl file containing the output filepath.",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )

    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="A seed for reproducible training."
    )

    parser.add_argument(
        "--num_samples", type=int, default=5, help="Number of summary samples.",
    )

    args = parser.parse_args()

    # Sanity checks
    extension = args.input_file.split(".")[-1]
    assert extension == "jsonl", "`input_file` should be a jsonl file."

    return args


LETTERS = ["A", "B", "C", "D", "E"]


def main() -> None:
    args = parse_args()
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    def load_dataset(input_file):  # type: ignore
        examples = []
        with jsonlines.open(input_file, "r") as reader:
            for row in reader:
                idx = row["id"]
                title = row["title"]
                post = row["post"]
                for index, i in enumerate(LETTERS):
                    generated_summary = row[f"generated_refinement_{i}"]
                    examples.append(
                        (
                            idx,
                            index,
                            CAUSAL_LM_CLASS_PROMPT.format(
                                title, post, generated_summary
                            ),
                        )
                    )
                human_summary = row["ideal_human_summary"]
                examples.append(
                    (
                        idx,
                        args.num_samples,
                        CAUSAL_LM_CLASS_PROMPT.format(title, post, human_summary),
                    )
                )
        return examples

    test_dataset = load_dataset(args.input_file)
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    # NOTE WITH OPT YOU SHOULD NOT USE THE FAST TOKENIZER!!
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        device_map="auto",
    )
    print("Loaded!")

    class RewardModelCollator:
        def __init__(  # type: ignore
            self, tokenizer, padding=True, max_length=None, return_tensors="pt",
        ):
            self.tokenizer = tokenizer
            self.padding = padding
            self.max_length = max_length
            self.return_tensors = return_tensors

        def __call__(self, batch):  # type: ignore
            all_idx, all_summary_names, all_batch_sequences = [], [], []
            for input_data in batch:
                all_idx.append(input_data[0])
                all_summary_names.append(input_data[1])
                all_batch_sequences.append(input_data[2])
            return (
                all_idx,
                all_summary_names,
                self.tokenizer(
                    all_batch_sequences,
                    padding=self.padding,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors=self.return_tensors,
                ),
            )

    # DataLoaders creation:
    data_collator = RewardModelCollator(tokenizer, max_length=args.max_length,)

    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

    print("***** Running evaluation *****")
    print(f"  Num examples = {len(test_dataset)}")
    print(f"  Input Examples= \n {test_dataset[random.randint(0, len(test_dataset))]}")
    print(f"  Instantaneous batch size per device = {args.per_device_eval_batch_size}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(len(test_dataloader)), ncols=120,)

    model.eval()

    predictions = defaultdict(lambda: [0.0] * (args.num_samples + 1))  # type: ignore
    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():
            outputs = model(**batch[2], use_cache=False)
            cur_predictions = get_predictions(
                outputs.logits.float(), batch[2]["input_ids"], tokenizer
            )
        for id_summary, id_generation, prediction in zip(
            batch[0], batch[1], cur_predictions.tolist()
        ):
            predictions[id_summary][id_generation] = prediction
        progress_bar.update(1)
    evaluate_output(predictions, args.num_samples)
    export_output(predictions, args)


def evaluate_output(model_outputs, human_summary_index):  # type: ignore
    predictions = []
    for summary_id, summary_rewards in model_outputs.items():
        predictions.append(int(np.argmax(summary_rewards)))
    labels = [human_summary_index] * len(predictions)
    print(f"Accuracy score: {accuracy_score(labels, predictions)}")
    print(f"Standard error: {sem(np.array(predictions)==np.array(labels))}")


def export_output(model_outputs, args):  # type: ignore

    input_data = []
    with jsonlines.open(args.input_file, "r") as input_file:
        for elem in input_file:
            input_data.append(elem)

    with jsonlines.open(args.output_file, "w") as output_file:
        for elem in input_data:
            idx = elem["id"]
            annotations = model_outputs[idx]
            elem["selected_summary"] = LETTERS[int(np.argmax(annotations[:-1]))]
            for i, annotation in enumerate(annotations):
                if i == args.num_samples:
                    elem["human_generated_summary_score"] = annotation
                else:
                    elem[f"generated_summary_{LETTERS[i]}_score"] = annotation
            output_file.write(elem)


def get_predictions(logits, input_ids, tokenizer):  # type: ignore
    B = logits.shape[0]
    sequence_lengths = torch.ne(input_ids, tokenizer.pad_token_id).sum(-1) - 1
    probs = torch.softmax(logits, -1)
    probs = probs[torch.arange(B), sequence_lengths, :]
    positive_token, negative_token = (
        tokenizer.encode(POSITIVE, add_special_tokens=False)[0],
        tokenizer.encode(NEGATIVE, add_special_tokens=False)[0],
    )
    output = probs[torch.arange(B), positive_token] / (
        probs[torch.arange(B), positive_token] + probs[torch.arange(B), negative_token]
    )
    return output


if __name__ == "__main__":
    main()
