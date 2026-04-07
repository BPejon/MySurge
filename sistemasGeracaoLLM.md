# LLMS Geracionais de Reviews academicos

## Ai2Scholar (não conseguimos rodar) - Introducing Ai2 ScholarQA | Ai2

## AutoSurvey2 https://github.com/AutoSurveys/AutoSurvey (usa LLM-as-judge) [2406.10252] AutoSurvey: Large Language Models Can Automatically Write Surveys

## SurveyX - (exemplos no site www.surveyx.cn?) https://www.surveyx.cn/ https://github.com/IAAR-Shanghai/SurveyX [2502.14776v2] SurveyX: Academic Survey Automation via Large Language Models

## Interactive Survey (não conseguimos rodar) https://github.com/TechnicolorGUO/InteractiveSurvey [2504.08762] InteractiveSurvey: An LLM-based Personalized and Interactive Survey Paper Generation System

## SurveyGenI - https://github.com/SurveyGens/SurveyGen-I [2508.14317] SurveyGen-I: Consistent Scientific Survey Generation with Evolving Plans and Memory-Guided Writing

## ARISE - https://github.com/ziwang11112/ARISE [2511.17689] ARISE: Agentic Rubric-Guided Iterative Survey Engine for Automated Scholarly Paper Generation

## otto-SR otto-SR | Automated Systematic Reviews with AI Agents (review generatio or just paper screening for systematic reviews)

## SurveyForge - [2503.04629] SurveyForge: On the Outline Heuristics, Memory-Driven Generation, and Multi-dimensional Evaluation for Automated Survey Writing SURVEYFORGE : On the Outline Heuristics, Memory-Driven Generation, and Multi-dimensional Evaluation for Automated Survey Writing - ACL Anthology

pip install -r requirements.txt

ERROR: Could not find a version that satisfies the requirement torch==2.1.0 (from versions: 2.2.0, 2.2.1, 2.2.2, 2.3.0, 2.3.1, 2.4.0, 2.4.1, 2.5.0, 2.5.1, 2.6.0, 2.7.0, 2.7.1, 2.8.0, 2.9.0, 2.9.1, 2.10.0)
ERROR: No matching distribution found for torch==2.1.0


## LIRA LiRA: A Multi-Agent Framework for Reliable and Readable Literature Review Generation https://github.com/lira-workflow/auto-review-writing

Conda activate automated-lit-review

Erro na criação dos datasets

python scripts/download_srg.py
Traceback (most recent call last):
  File "/home/breno/Documentos/LLMModelsForGenerateReview/Lira/scripts/download_srg.py", line 5, in <module>
    import gc, os, re, ast, csv, gdown, hydra, tarfile
ModuleNotFoundError: No module named 'gdown'

Datasets

The datasets used for this project include an internal (private; only accessible via the project page which also contain instructions on how to handle it) and external (SciReviewGen) dataset. The external one can be downloaded using the following script:

python scripts/download_srg.py

This also includes an automatic download of all the metadata from Semantic Scholar for referencing, as the SciReviewGen dataset does not have this on its own. Simply adjust the seed and number of samples to customize the proportion of the data to use. Do keep in mind that this value needs to be adjusted for all configuration files.

The data entries may need to be cleaned beforehand (for example, to remove the word "Abstract" from the beginning, formatting the table to match the determined format). This can be performed by running the following script:

python scripts/clean_data.py