azcopy copy "https://openaipublic.blob.core.windows.net/summarize-from-feedback/datasets/tldr_3_filtered/*" .
rm -r datasets
mv train.jsonl original_train.jsonl