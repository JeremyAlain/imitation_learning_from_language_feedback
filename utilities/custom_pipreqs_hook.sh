#!/bin/bash
pipreqs --use-local --force --ignore .mypy_cache --savepath utilities/pipreqs.txt
LC_COLLATE=en_US.UTF-8 sort utilities/non_imported_requirements.txt utilities/pipreqs.txt | uniq  > requirements.txt
rm utilities/pipreqs.txt