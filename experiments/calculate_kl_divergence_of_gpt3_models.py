import click
from kl_gpt3.kl_gpt3 import GPT3, Batch
import numpy as np
import pandas as pd
import srsly

from utilities.language_model_utilities import set_openai_key_to_environment_variable

models_to_evaluate_with_id = {
    "finetuned_on_5000_refinements": "-",
    "finetuned_on_5000_human_summaries": "-",
    "finetuned_on_5000_initial_summaries": "-",
}


@click.command()
@click.option("--number_of_samples", required=True, type=int, default=32)
@click.option(
    "--kl_divergence_type",
    required=True,
    type=click.Choice(["forward", "backward"]),
    default="forward",
)
@click.option("--file_path", required=False, type=str)
def main(number_of_samples: int, kl_divergence_type: str, file_path: str) -> None:
    set_openai_key_to_environment_variable()
    batch_size = min(2, number_of_samples)
    baseline_davinci = GPT3(model_name="davinci", max_tokens=64, batch_size=batch_size)
    if kl_divergence_type == "forward":

        if file_path is not None:
            data = pd.read_json(file_path)

            assert len(data["texts"].tolist()) == len(data["logprobs"].tolist())
            bad_texts_list = []
            for i in range(len(data["texts"])):
                if data["texts"].tolist()[i] == "<|endoftext|>":
                    bad_texts_list.append(i)
            data = data.drop(bad_texts_list)
            data.reset_index(drop=True, inplace=True)
            baseline_davinci_samples = Batch(
                model_name="davinci",
                texts=data["texts"].tolist(),
                logprobs=data["logprobs"].tolist(),
            )
        else:
            baseline_davinci_samples = baseline_davinci.sample(
                num_samples=number_of_samples, save_logprobs=True
            )
            baseline_davinci_samples.save_to_json(
                "data/results/kl_distance_of_gpt3_models/log_probs_davinci_on_{}_davinci_samples.json".format(
                    number_of_samples
                )
            )
            bad_texts_list = []
            texts = baseline_davinci_samples.texts
            for i in range(len(texts)):
                if texts[i] == "<|endoftext|>":
                    bad_texts_list.append(i)
            baseline_davinci_samples.texts = [
                text for i, text in enumerate(texts) if i not in bad_texts_list
            ]
            baseline_davinci_samples.logprobs = np.array(
                [
                    logprob
                    for i, logprob in enumerate(
                        baseline_davinci_samples.logprobs.tolist()
                    )
                    if i not in bad_texts_list
                ]
            )

            assert len(baseline_davinci_samples.texts) == len(
                baseline_davinci_samples.logprobs.tolist()
            )

        for model_name, model_id in models_to_evaluate_with_id.items():
            print("Evaluating Model {}".format(model_name))
            model = GPT3(model_name=model_id, max_tokens=64, batch_size=batch_size)
            model_logprobs = model.get_logprobs(baseline_davinci_samples)
            kl = (baseline_davinci_samples.logprobs - model_logprobs).mean()
            print("KL Divergence of Davinci vs. {} : {}".format(model_name, kl))
            content = {
                "model_name": model_name,
                "texts": baseline_davinci_samples.texts,
                "logprobs": model_logprobs.tolist(),
            }
            srsly.write_json(
                "data/results/kl_distance_of_gpt3_models/log_probs_{}_on_davinci_{}_samples.json".format(
                    "_".join(model_name.split("_")[3:]), number_of_samples
                ),
                content,
            )

    elif kl_divergence_type == "backward":
        for model_name, model_id in models_to_evaluate_with_id.items():
            print("Evaluating Model {}".format(model_name))
            model = GPT3(model_name=model_id, max_tokens=64, batch_size=batch_size)
            model_samples = model.sample(
                num_samples=number_of_samples, save_logprobs=True
            )
            model_samples.save_to_json(
                "data/results/kl_distance_of_gpt3_models/log_probs_{}_on_{}_{}_samples.json".format(
                    model_name, number_of_samples, model_name
                )
            )

            baseline_davinci = GPT3(
                model_name="davinci", max_tokens=64, batch_size=batch_size
            )
            davinci_log_probs = baseline_davinci.get_logprobs(model_samples)
            kl = (model_samples.logprobs - davinci_log_probs).mean()
            print("KL Divergence of {} vs. Davinci : {}".format(model_name, kl))
            content = {
                "model_name": "davinci",
                "texts": model_samples.texts,
                "logprobs": davinci_log_probs.tolist(),
            }
            srsly.write_json(
                "data/results/kl_distance_of_gpt3_models/log_probs_davinci_on_{}_{}_samples.json".format(
                    number_of_samples, "_".join(model_name.split("_")[3:])
                ),
                content,
            )
    else:
        raise ValueError("Invalid KL Divergence Type")


if __name__ == "__main__":
    main()
