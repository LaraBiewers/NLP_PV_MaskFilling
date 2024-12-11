
import csv
import json
import time


# 1) load Data
print("\nSTART LOADING DATA...")

load_start_time = time.time()

#csv => masked words
with open('dataset\\preprocessed\\preprocessedData_10_000.csv', 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    rows_dataset = list(reader)
    num_rows_dataset = sum(1 for _ in rows_dataset)

masked_words = []
for sample_solution_row in rows_dataset:
    masked_words.append(sample_solution_row['Masked Word'])

#json => pipeline results
def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = []
        for line in file:
            try:
                cleaned_line = line.strip() #DEL?
                json_entry = json.loads(cleaned_line)
                data.append(json_entry)
            except json.JSONDecodeError as e:
                print(f"Fehler beim Parsen der Zeile: {e}")
    return data

jsonl_file_path = 'unmaskerResults_10_000.jsonl'
results_of_unmasker = load_jsonl(jsonl_file_path)

load_end_time = time.time()
load_time = load_end_time - load_start_time

print("\nFINISHED LOADING DATA...")

print(f">>> time needed to calculate probability: {load_time:.2f} seconds\n")



# 2) evaluation 
print("\nSTART CALCULATING PROPABILITIES...")

calc_start_time = time.time()

overall_solution_score_ZERO = 0
true_positive_ZERO = 0
overall_solution_score_TOP = 0
true_positive_TOP = 0

sample_solution_row = 0

for results_per_sentence in results_of_unmasker:

    #NOTE: zero-shot
    if results_per_sentence[0]['token_str'] == masked_words[sample_solution_row]:
            overall_solution_score_ZERO += result['score']
            true_positive_ZERO += 1

    #NOTE: Top-k
    for result in results_per_sentence: 
        if result['token_str'] == masked_words[sample_solution_row]:
            overall_solution_score_TOP += result['score']
            true_positive_TOP += 1
            break

    sample_solution_row += 1


avg_score_of_model_ZERO = (overall_solution_score_ZERO / num_rows_dataset) * 100
avg_score_of_model_TOP = (overall_solution_score_TOP / num_rows_dataset) * 100

calc_end_time = time.time()
calc_time = calc_end_time - calc_start_time

print("\nFINISHED CALCULATING PROPABILITIES...")
print(f"\n>>> total masked sentences: {num_rows_dataset}")
print(f">>> correct solutions found ==> zeroShot: {true_positive_ZERO} ({true_positive_ZERO/num_rows_dataset*100:.2f}%), topK: {true_positive_TOP} ({true_positive_TOP/num_rows_dataset*100:.2f}%)")
# print(f"\n>>> average score of correct predictions ==> zeroShot: {avg_score_of_model_ZERO:.3f}%, topK: {avg_score_of_model_TOP:.3f}%") #Aussagekraft?!
print(f"\n>>> time needed to calculate probability: {calc_time:.2f} seconds\n")

