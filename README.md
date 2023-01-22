# Training Language Models with Language Feedback at Scale

## Disclaimer
This repository contains code related to the paper "Training Language Models with Language Feedback at Scale." 
Please note that the code is provided as a reference and requires modification to run successfully. 
The project involved multiple rounds of human data collection and cleaning, and as such, all data-generating, data-processing and
data-merging, files have  been excluded from this repository. The final data used in the experiments can be downloaded from the Hugging Face hub.
We also do provide a link to a Google Drive folder that contains all the final datasets we used for finetuning the models
(including the reward model). As such we do not provide the actual files we used to generate the refinements, but we provide 
the final refinements we finetuned on.
This repository is intended to serve as a guide for implementing similar experiments, rather than a ready-to-use solution.

## Installation
### GPT-3 access
You will only be able to run experiments using GPT-3 if you have an API Key. 
If you do have a key then create a .env file in the root folder and add the following line where "YourKey" is your API-Key.

```
OPENAI_API_KEY="YourKEY"
```

If you use an IDE such as Pycharm your environment variables won't be directly read in. You can either 
set the environment variables in your runtime arguments or you can do the following:
```
1. create a .env file in your local project structure
2. Add OPENAI_API_KEY="your_api_key" to your .env file
3. Add your .env file to your .gitignore so that they key won't be shared publicly
```

Note that the names for the finetuned models were replaced with a "-", you wouldn't be able to run them anyway, 
since they were trained with our own OpenAI API account. 

### Pre-commit hooks
We use pre-commit hooks to format our code (black & pep8), add dependencies automatically, enforce typing and other properties etc.
We mainly note this in case you fork the repo. In case you want to use it execute the following:

```
pre-commit install
```

## Dataset

## Download our SLF5K dataset from HugginfaceHub
You can download the SLF5K datast from the HugginfaceHub. We provide the following dataset splits: 
- Train Data: 5000 samples
- Development Data: 200 samples
- Validation Data: 500 samples
- Test Data: 698 samples

We used the language feedback to generate refinements and the binary comparisons to train our reward models. You can use this
data to generate your own summaries, refinements etc. by using the code in the folder `experiments`, although you will 
have to adapt the code and bring the data in the correct format.

The final refinements, human summaries, initial summaries etc. that we finetuned our models on can be downloaded from a Google Drive, which we 
explain in the next section.

## Download all finetuning datasets 
We provide all datasets we used for finetuning the models in a Google Drive folder. You can download the data [here](https://drive.google.com/drive/folders/1oeMpAmMhVgzRrA0lV74b98dh9Yq8Rc4k?usp=share_link) in the folder `summarization_finetuning_datasets`.
We provide datasets for the following experiments: 
- finetuning on refinements (100, 1K, 5K)
- finetuning on human summaries (100, 1K, 5K)
- finetuning on initial summaries (100, 1K, 5K)
- finetuning on feedback + refinements (100, 1K, 5K)
- Reward Model Classification and Comparison (100, 1K, 5K)

The standard OPT-RM was trained on a dataset that you can generate with (Todo).

### Download Results 
All the results (mostly based on human comparisons) can be found in the same Google Drive in the folder `results`. The plots 
can be generated with the script `` and can be found in the folder `plots`.

## [Optional] Download Dataset used in Stiennon et. al. 
We provide a small script to download the data from Stiennon et al. We split off our dataset from theirs, so this is only provided
in case you somehow want to extend our dataset with more samples. You don't actually need to download their data. The script can be found in 
`data/tldr_dataset_stiennon_et_al/download_data_from_stiennon_et_al.sh`.


## Citation
If you use the code on this website or the algorithm of our paper, please cite use: 
