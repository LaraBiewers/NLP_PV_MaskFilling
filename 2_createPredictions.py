from transformers import pipeline, AutoModelForMaskedLM, AutoTokenizer
import torch
import csv
import json
import time


# 1) load model
checkpoint = "google-bert/bert-base-cased"
# model = AutoModelForMaskedLM.from_pretrained(checkpoint)
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)


# 2) load data from csv-file
rows_dataset = []
num_rows_dataset = 0
with open('dataset\\preprocessed\\preprocessedData_full.csv', 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    rows_dataset = list(reader)
    num_rows_dataset = sum(1 for _ in rows_dataset)

print(f"\n>>> num of dataset-rows: {num_rows_dataset}\n")
# print(rows_dataset)


# 3) predict masked words
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f">>> using device {device}\n")
torch.cuda.empty_cache()
unmasker = pipeline('fill-mask', trust_remote_code=True, batch_size=11, model=checkpoint, device=device, top_k=5) # device = 0 ==> use GPU [device = device]

print("\nSTART PREDICTING MASKED WORDS...")

masked_sentences = []
masked_words = []
for sample_solution_row in rows_dataset:
    masked_sentences.append(sample_solution_row['Masked Sentence'])
    masked_words.append(sample_solution_row['Masked Word'])

pre_unmask_time = time.time()
result_of_unmasker = unmasker(masked_sentences) #NOTE: call only one unmasker for whole data
post_unmask_time = time.time()
unmask_time_sec = post_unmask_time - pre_unmask_time
unmask_time_min = unmask_time_sec / 60

print("\nFINISHED PREDICTING MASKED WORDS...")
print(f">>> time needed to calculate masked words: {unmask_time_sec:.2f} Seconds == {unmask_time_min:.2f} Minutes")


# 4) Transfer data to a separate file for later calculations
print("\nSTART SAVING UNMASKER RESULTS TO JSON FILE...")

output_file = 'unmaskerResults_full.jsonl'
with open(output_file, 'w', encoding='utf-8') as f:
    for sentence_results in result_of_unmasker:
        json.dump(sentence_results, f)
        f.write('\n')

print(f"UNMASKER RESULTS SAVED TO {output_file}")
