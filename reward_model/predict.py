import argparse
import logging
import os
import random
from typing import Optional

import jsonlines
import numpy as np
from scipy.stats import sem
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
)

POST_FORMAT = "SUBREDDIT:r/ {}\n\nTITLE: {}\n\nPOST: {}\n\nTL;DR: {}"
CAUSAL_LM_CLASS_PROMPT = "Title: {}\n\nText: {}\n\nTL;DR: {}\n\nQuestion: Is the above an excellent summary of the given text? An excellent summary is coherent, accurate, concise, and detailed. Answer with Yes or No.\n\nAnswer:"
CAUSAL_LM_COMP_PROMPT = "Title: {}\n\nText: {}\n\nSummary A: {}\n\nSummary B: {}\n\nQuestion: Which summary is the better one? An excellent summary is coherent, accurate, concise, and detailed. Answer with A or B.\n\nAnswer:"
POSITIVE = " Yes"
NEGATIVE = " No"
FIRST = " A"
SECOND = " B"

os.environ["CXX"] = "g++"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reward model prediction")

    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="A jsonl file containing the test data.",
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
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="causalLMclass",
        help="The model type to use.",
        choices=["causalLMclass", "causalLMcompl", "classification"],
    )

    parser.add_argument(
        "--prompt_input",
        action="store_true",
        help="Whether the input file is already in the format of the prompt. ",
    )

    args = parser.parse_args()

    # Sanity checks
    extension = args.test_file.split(".")[-1]
    assert extension == "jsonl", "`test_file` should be a jsonl file."

    return args


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

    def load_dataset(test_file):  # type: ignore
        examples = []
        with jsonlines.open(test_file, "r") as reader:
            for row in reader:
                if args.prompt_input:
                    if args.model_type == "causalLMclass":
                        A_input = row["prompt"]
                        B_input = reader.read()["prompt"]
                        label = 1 if row["completion"] == POSITIVE else 0
                    else:
                        A_input = row["prompt"]
                        label = 1 if row["completion"] == FIRST else 0
                else:
                    post = row["post"]
                    title = row["title"]
                    subreddit = row["subreddit"]
                    label = 1 if row["comparison_preference"] == "Summary A" else 0
                    if args.model_type == "causalLMclass":
                        A_input = CAUSAL_LM_CLASS_PROMPT.format(
                            title, post, row["generated_summary_for_comparison_A"],
                        )
                        B_input = CAUSAL_LM_CLASS_PROMPT.format(
                            title, post, row["generated_summary_for_comparison_B"],
                        )
                    elif args.model_type == "causalLMcompl":
                        A_input = CAUSAL_LM_COMP_PROMPT.format(
                            title,
                            post,
                            row["generated_summary_for_comparison_A"],
                            row["generated_summary_for_comparison_B"],
                        )
                    else:
                        A_input = POST_FORMAT.format(
                            subreddit,
                            title,
                            post,
                            row["generated_summary_for_comparison_A"],
                        )
                        B_input = POST_FORMAT.format(
                            subreddit,
                            title,
                            post,
                            row["generated_summary_for_comparison_B"],
                        )
                if args.model_type == "causalLMcompl":
                    examples.append((A_input, label))
                else:
                    examples.append((A_input, B_input, label))
        return examples

    test_dataset = load_dataset(args.test_file)

    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    # NOTE WITH OPT YOU SHOULD NOT USE THE FAST TOKENIZER!!
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    print("Loading model...")
    if args.model_type == "classification":
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            device_map="auto",
        )
    print("Loaded!")

    class RewardModelCollator:
        def __init__(  # type: ignore
            self,
            tokenizer,
            padding: bool = True,
            max_length: Optional[int] = None,
            return_tensors: str = "pt",
        ):
            self.tokenizer = tokenizer
            self.padding = padding
            self.max_length = max_length
            self.return_tensors = return_tensors

        def __call__(self, batch):  # type: ignore
            all_batch_sequences = []
            all_labels = []
            for input_data in batch:
                if len(input_data) > 2:
                    all_batch_sequences.append(input_data[0])
                    all_batch_sequences.append(input_data[1])
                else:
                    all_batch_sequences.append(input_data[0])
                all_labels.append(input_data[-1])
            return (
                self.tokenizer(
                    all_batch_sequences,
                    padding=self.padding,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors=self.return_tensors,
                ),
                all_labels,
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
    predictions = []
    labels = []
    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():
            outputs = model(**batch[0], use_cache=False)
            cur_predictions = get_predictions(
                outputs.logits.float(),
                batch[0]["input_ids"],
                tokenizer,
                args.model_type,
            )
        predictions += cur_predictions.tolist()
        labels += batch[1]
        progress_bar.update(1)
    evaluate_output(predictions, labels)
    export_output(predictions, test_dataset, args)


def evaluate_output(model_outputs, labels):  # type: ignore
    predictions = []
    for (prediction_A, prediction_B) in zip(model_outputs[0::2], model_outputs[1::2]):
        predictions.append(prediction_A > prediction_B)
    print(f"Accuracy score: {accuracy_score(labels, predictions)}")
    print(f"Standard error: {sem(np.array(predictions)==np.array(labels))}")


def export_output(predictions, data, args):  # type: ignore
    with jsonlines.open(
        os.path.join(args.output_dir, "predictions.json"), "w"
    ) as output_file:
        for i, (prediction_A, prediction_B) in enumerate(
            zip(predictions[0::2], predictions[1::2])
        ):
            output_file.write([i, data[i], prediction_A, prediction_B])


def get_predictions(logits, input_ids, tokenizer, model_type):  # type: ignore
    B = logits.shape[0]
    if model_type == "classification":
        output = logits.squeeze()
    else:
        sequence_lengths = torch.ne(input_ids, tokenizer.pad_token_id).sum(-1) - 1
        probs = torch.softmax(logits, -1)
        probs = probs[torch.arange(B), sequence_lengths, :]
        if model_type == "causalLMclass":
            positive_token, negative_token = (
                tokenizer.encode(POSITIVE, add_special_tokens=False)[0],
                tokenizer.encode(NEGATIVE, add_special_tokens=False)[0],
            )
            output = probs[torch.arange(B), positive_token] / (
                probs[torch.arange(B), positive_token]
                + probs[torch.arange(B), negative_token]
            )
        else:
            first_token, second_token = (
                tokenizer.encode(FIRST, add_special_tokens=False)[0],
                tokenizer.encode(SECOND, add_special_tokens=False)[0],
            )
            output = torch.zeros((2 * B), dtype=torch.float32)
            output[torch.arange(0, 2 * B, 2)] = probs[torch.arange(B), first_token]
            output[torch.arange(1, 2 * B, 2)] = probs[torch.arange(B), second_token]

    return output


if __name__ == "__main__":
    main()
