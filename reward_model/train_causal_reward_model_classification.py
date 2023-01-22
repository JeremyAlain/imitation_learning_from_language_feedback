import argparse
import logging
import math
import os

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import datasets
import jsonlines
from sklearn.metrics import accuracy_score
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_scheduler
from transformers.utils.versions import require_version

logger = get_logger(__name__)

POSITIVE = " Yes"
NEGATIVE = " No"

require_version(
    "datasets>=1.8.0",
    "accelerate>=0.12.0"
    "To fix: pip install -r examples/pytorch/text-classification/requirements.txt",
)

os.environ["CXX"] = "g++"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finetune a reward model based on a transformer to predict rewards for summaries based"
        "on human preferences (RLHF)"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Experiment name that appears on logging.",
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )

    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="A jsonl file containing the training data.",
    )

    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="A jsonl file containing the validation data.",
    )

    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="A jsonl file containing the test data.",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )

    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )

    parser.add_argument(
        "--checkpointing",
        action="store_true",
        help="Whether the various states should be saved.",
    )
    parser.add_argument(
        "--store_best",
        action="store_true",
        help="Whether the best model should be saved.",
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )

    parser.add_argument(
        "--warmup",
        type=float,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--logging_dir", type=str, default=None, help="Path for the logging."
    )

    parser.add_argument(
        "--report_to",
        default="tensorflow",
        help="Method for reporting",
        choices=["all", "tensorboard", "wandb", "comet_ml"],
    )

    parser.add_argument(
        "--log_infos",
        action="store_true",
        help="If passed, all infos are logged to console.",
    )

    parser.add_argument(
        "--not_validate_after_epoch",
        action="store_true",
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--train_size", type=int, default=0, help="Number of training elements.",
    )

    parser.add_argument(
        "--prompt_loss_weight",
        type=float,
        default=0.0,
        help="Weight used for the loss of the prompt.",
    )

    args = parser.parse_args()

    # Sanity checks
    if args.train_file is not None:
        extension = args.train_file.split(".")[-1]
        assert extension == "jsonl", "`train_file` should be a jsonl file."
    if args.validation_file is not None:
        extension = args.validation_file.split(".")[-1]
        assert extension in "jsonl", "`validation_file` should be a jsonl file."

    return args


def main() -> None:
    args = parse_args()
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator(
        log_with=args.report_to,
        logging_dir=args.logging_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    if args.log_infos:
        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )

    logger.info(accelerator.state, main_process_only=True)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    accelerator.wait_for_everyone()

    # Loading the dataset from local jsonl.
    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
    if args.test_file is not None:
        data_files["test"] = args.test_file

    def load_dataset(data_files):  # type: ignore
        # Important: Dataset should be ordered, we add elements in pairs for easier training/eval
        raw_datasets = {}
        for split, input_path in data_files.items():
            current_examples = []
            with jsonlines.open(input_path, "r") as reader:
                for row in reader:
                    current_examples.append((row, reader.read()))
            if split == "train":
                current_examples = (
                    current_examples[: args.train_size]
                    if args.train_size > 0
                    else current_examples
                )
            raw_datasets[split] = current_examples
        return raw_datasets

    raw_datasets = load_dataset(data_files=data_files)

    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    # NOTE WITH OPT YOU SHOULD NOT USE THE FAST TOKENIZER!!
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )
    train_dataset = (
        raw_datasets["train"][: args.train_size]
        if args.train_size > 0
        else raw_datasets["train"]
    )

    eval_dataset = raw_datasets["validation"]

    class RewardModelCollator:
        def __init__(  # type: ignore
            self,
            tokenizer,
            padding=True,
            max_length=None,
            pad_to_multiple_of=None,
            return_tensors="pt",
        ):
            self.tokenizer = tokenizer
            self.padding = padding
            self.max_length = max_length
            self.pad_to_multiple_of = pad_to_multiple_of
            self.return_tensors = return_tensors

        def __call__(self, batch):  # type: ignore
            all_batch_inputs = []
            for (elem0, elem1) in batch:
                if "Yes" in elem0["completion"]:
                    all_batch_inputs.append(elem0["prompt"] + elem0["completion"])
                    all_batch_inputs.append(elem1["prompt"] + elem1["completion"])
                else:
                    all_batch_inputs.append(elem1["prompt"] + elem1["completion"])
                    all_batch_inputs.append(elem0["prompt"] + elem0["completion"])

            input_elems = self.tokenizer(
                all_batch_inputs,
                padding=self.padding,
                truncation=True,
                max_length=self.max_length,
                return_tensors=self.return_tensors,
            )
            target_elems = input_elems["input_ids"].clone()

            if self.tokenizer.pad_token_id is not None:
                target_elems[target_elems == self.tokenizer.pad_token_id] = -100

            return (input_elems, target_elems)

    # DataLoaders creation:
    data_collator = RewardModelCollator(
        tokenizer,
        max_length=args.max_length,
        pad_to_multiple_of=(8 if accelerator.use_fp16 else None),
    )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(
            args.max_train_steps / num_update_steps_per_epoch
        )
        logger.info(
            "Number of epochs that was potentially set will be ignored and set to {}".format(
                args.num_train_epochs
            )
        )

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=math.ceil(
            args.warmup * args.max_train_steps * args.gradient_accumulation_steps
        ),
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    # Prepare everything with our `accelerator`.
    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # We need to initialize the trackers we use, and also store our configuration
    experiment_config = vars(args)
    # TensorBoard cannot log Enums, need the raw value
    # experiment_config["lr_scheduler_type"] = experiment_config[
    #    "lr_scheduler_type"
    # ].value
    accelerator.init_trackers(args.experiment_name, experiment_config)

    assert (
        accelerator.gradient_accumulation_steps == args.gradient_accumulation_steps
    ), "The gradient accumulation steps must be the same as the one used for the training"
    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * accelerator.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Input Examples= \n {train_dataset[0]}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        ncols=120,
    )

    total_loss, batch_loss, best_validation = 0.0, 0.0, 0.0
    steps, starting_epoch = 0, 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            # Sorts folders by date modified, most recent checkpoint is the last
            path = dirs[-1]
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]
        starting_epoch = int(training_difference.replace("epoch_", "")) + 1

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        for i, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                input_ids, labels = batch
                outputs = model(**input_ids, use_cache=False)
                loss = calculate_reward_model_loss(
                    outputs.logits, labels, args.prompt_loss_weight
                )
                del outputs
                total_loss += loss.item()
                batch_loss += loss.item()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                steps += 1

                accelerator.log({"learning_rate": lr_scheduler.get_last_lr()[0]})
                accelerator.log(
                    {"loss": batch_loss / accelerator.gradient_accumulation_steps}
                )
                accelerator.log(
                    {
                        "running_loss": total_loss
                        / (steps * args.gradient_accumulation_steps)
                    }
                )

                desc_str = f"Epoch {epoch}| Steps: {steps} | LR {lr_scheduler.get_last_lr()[0]:.2e}"
                desc_str += (
                    f" | Loss {batch_loss/accelerator.gradient_accumulation_steps:4f}"
                )

                progress_bar.set_description(desc_str)
                progress_bar.update(1)

                batch_loss = 0.0

        if not args.not_validate_after_epoch:
            accuracy = evaluate_model_on_validation_set(
                model,
                eval_dataloader,
                accelerator,
                epoch,
                steps,
                progress_bar,
                args,
                tokenizer,
            )
            if args.output_dir is not None and args.store_best:
                if accuracy > best_validation:
                    best_validation = accuracy
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(
                        args.output_dir,
                        is_main_process=accelerator.is_main_process,
                        save_function=accelerator.save,
                    )
                    if accelerator.is_main_process:
                        tokenizer.save_pretrained(args.output_dir)

                if args.checkpointing:
                    output_dir = f"epoch_{epoch }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

    accelerator.wait_for_everyone()


def evaluate_model_on_validation_set(  # type: ignore
    model,
    eval_dataloader,
    accelerator,
    epoch,
    completed_steps,
    progress_bar,
    args,
    tokenizer,
):

    model.eval()
    samples_seen = 0
    total_loss = 0.0
    total_steps = 0
    predictions = []
    for step, batch in enumerate(eval_dataloader):
        input_ids, labels = batch
        with torch.no_grad():
            outputs = model(**input_ids, use_cache=False)
        loss = calculate_reward_model_loss(
            outputs.logits, labels, args.prompt_loss_weight
        )
        total_loss += loss.item()
        # Target is always one as we have preferred summary always coming first.
        cur_predictions = get_predictions(
            batch[0]["input_ids"], outputs.logits, tokenizer
        )
        cur_predictions = accelerator.gather(cur_predictions)
        # If we are in a multiprocess environment, the last batch has duplicates
        if accelerator.num_processes > 1:
            if step == len(eval_dataloader):
                cur_predictions = cur_predictions[
                    : len(eval_dataloader.dataset) - samples_seen
                ]
            else:
                samples_seen += cur_predictions.shape[0]
        predictions += cur_predictions.detach().tolist()
        total_steps += 1

    validation_accuracy = accuracy_score([1] * len(predictions), predictions)
    validation_accuracy = validation_accuracy

    desc_str = "----------------------------"
    desc_str += f"Epoch {epoch} | Steps {completed_steps}"
    desc_str += f" | Validation Accuracy {validation_accuracy}"
    desc_str += f" | Validation Loss {total_loss/total_steps}"
    desc_str += "----------------------------"

    progress_bar.set_description(desc_str)

    accelerator.log({"validation_accuracy": validation_accuracy})
    model.train()
    return validation_accuracy


def get_predictions(input_ids, logits, tokenizer):  # type: ignore
    B = logits.shape[0]
    sequence_lengths = torch.ne(input_ids, tokenizer.pad_token_id).sum(-1) - 2
    logits = logits[torch.arange(B), sequence_lengths, :]
    summary_1_logits = logits[torch.arange(0, B, 2)]
    summary_2_logits = logits[torch.arange(1, B, 2)]
    positive_token, negative_token = (
        tokenizer.encode(POSITIVE, add_special_tokens=False)[0],
        tokenizer.encode(NEGATIVE, add_special_tokens=False)[0],
    )
    summary_1_probs = torch.softmax(summary_1_logits, -1)
    summary_2_probs = torch.softmax(summary_2_logits, -1)
    # All positive for normalization
    summary_1_probs = summary_1_probs[:, positive_token] / (
        summary_1_probs[:, positive_token] + summary_1_probs[:, negative_token]
    )
    summary_2_probs = summary_2_probs[:, positive_token] / (
        summary_2_probs[:, positive_token] + summary_2_probs[:, negative_token]
    )

    return (summary_1_probs > summary_2_probs).long()


def calculate_reward_model_loss(logits, labels, prompt_loss_weight):  # type: ignore
    B, S, V = logits.shape
    loss_fct = CrossEntropyLoss(reduction="none")
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    completion_index = torch.ne(shift_labels, -100).sum(-1) - 1
    prompt_mask = torch.ones((shift_labels.shape)) * prompt_loss_weight
    prompt_mask[torch.arange(B), completion_index] = 1.0
    prompt_mask = prompt_mask.view(-1).to(logits.device)
    loss = loss_fct(shift_logits.view(-1, V), shift_labels.view(-1))
    # Normalize taking into account the weights
    loss = loss * prompt_mask
    avg_loss = torch.sum(loss) / torch.ne(loss, 0).sum()
    return avg_loss


if __name__ == "__main__":
    main()
