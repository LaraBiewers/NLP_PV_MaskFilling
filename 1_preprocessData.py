import random
import re
import csv
from datasets import load_dataset

# NOTE
# Is there a minimum count of words for sentences, so that "mask-filling" works "good"? 
#   ==> I did not found a minimum count, therefore i choose a minimum count of 5 Words by myself, so that the 
#       Model has some possible context.


# 1) Feed in data set
print("\nLOADING DATASET...")
#You need to go to https://recipenlg.cs.put.poznan.pl/ and manually download the dataset.
raw_dataset = load_dataset("mbien/recipe_nlg", data_dir=".\\dataset", trust_remote_code=True)
# print("\nFormat of the Dataset: ", raw_dataset, "\n")


# 2) extract sentences
def combine(direction_fragments):

    whole_direction_text = ""

    for part_of_directions in direction_fragments:
        whole_direction_text += part_of_directions + " "

    whole_direction_text = ' '.join(whole_direction_text.split()) #eliminate multiple whitespaces
    return whole_direction_text

def extract_sentences(text):

    clean_sentences = []
    instructions = re.split(r'[.!?]+ ', text)

    for instruction in instructions:
        instruction.strip()
        if len(instruction) > 0 and not re.search(r'[.!?]+', instruction[-1]):
            instruction = instruction + "."
        clean_sentences.append(instruction)

    return clean_sentences

raw_train_dataset = raw_dataset["train"]
all_sentences = []

print(f">>> amount of recipes in raw dataset: {len(raw_train_dataset)}")

print("\nSTART EXTRACTING SENTENCES...")

hurensohn = 0

#NOTE: recipe-amount: 10_000 = 7min; CIRCA 60_000 MIT 1H LAUFZEIT
# b = 0 #TEST
# test_amount_of_recipes = 50_000 #TEST
# print(f">>> working with {test_amount_of_recipes} recipes")
# while b < test_amount_of_recipes: #TEST
for recipe in raw_train_dataset: #ALL
    # recipe_directions = raw_train_dataset[b]["directions"] #TEST
    recipe_directions = recipe["directions"] #ALL
    combined_recipe_directions = combine(recipe_directions)
    recipe_sentence_list = extract_sentences(combined_recipe_directions)

    #save sentences longer than 4 words
    for sentence in recipe_sentence_list:
        if len(sentence) > 500: #NOTE: prevent to large sentences! They crash the pipeline!
            hurensohn += 1
            continue
        word_count = sentence.split()
        if len(word_count) >= 5:
            all_sentences.append([sentence])

    # b += 1 #TEST
    
print("\nFINISHED EXTRACTING SENTENCES...")
print(f">>> Amount of to long sentences: {hurensohn}")
print(f">>> Size of useable sentences: '{len(all_sentences)}' Lines.")


# 3) mask one random word in sentence
exclude_special_char_pattern = r'[\s!@#$%&*()_+={}[\]|;:<>,.?\\"]' #NOTE: don't exclude words with ' (cause of english corpus)
contains_digits_pattern = r'[0-9]'
full_word_in_braces = r'[(](\w)*[)]'
mask_token = "[MASK]"

def select_maskable_word(sentence):
    words = sentence.split()
    maskable_words = []
    for word in words:
        if len(word) > 1 and not re.search(contains_digits_pattern, word):
            if not re.search(full_word_in_braces, word) and not re.search(exclude_special_char_pattern, word[1:-1]):
                while len(word) > 0 and re.search(exclude_special_char_pattern, word[len(word)-1]):
                    word = word[:-1]
                    if len(word) == 0:
                        break
                while len(word) > 0 and re.search(exclude_special_char_pattern, word[0]):
                    word = word[1:]
                    if len(word) == 0:
                        break
                if len(word) > 0:
                    maskable_words.append(word)
    return random.choice(maskable_words) if maskable_words else None

print("\nSTART MASKING SENTENCES...")

for sentence_list in all_sentences:
    sentence = sentence_list[0]
    maskable_word = select_maskable_word(sentence)
    if maskable_word:
        maskable_word = r'\b' + maskable_word + r'\b'
        masked_sentence = re.sub(maskable_word, mask_token, sentence, count=1)
        if re.search(r'\[MASK\]', masked_sentence):
            sentence_list.append(masked_sentence)
            sentence_list.append(maskable_word[2:-2])

print("\nFINISHED MASKING SENTENCES...")

unuseable_sentences = 0
for sentence_list in all_sentences:
    if len(sentence_list) != 3:
        # print(sentence_list)
        unuseable_sentences += 1

print(f">>> Size of maskable sentences: '{len(all_sentences) - unuseable_sentences}' Lines.")
print(f">>> unuseable_sentences: '{unuseable_sentences}'")


# 4) save all_sentences which can be used
print("\nSTART WRITING TO .CSV-FILE...")
csv_file_name = "dataset\\preprocessed\\preprocessedData_full.csv"

with open(csv_file_name, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    
    header = ['Original Sentence', 'Masked Sentence', 'Masked Word']
    writer.writerow(header)
    
    for sentence_list in all_sentences:
        if len(sentence_list) == 3:
            original_sentence = sentence_list[0]
            masked_sentence = sentence_list[1]
            masked_word = sentence_list[2]
            
            row = [
                original_sentence,
                masked_sentence,
                masked_word
            ]
            writer.writerow(row)

print(f">>>list successfully saved in '{csv_file_name}'!")


#NOTE: Take a look at sentences which can't have a masked word
csv_file_name_acc = "dataset\\extracted_accidents\\preprocessedDataAccident_full.csv"

with open(csv_file_name_acc, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    
    header = ['Original Sentence with Accident']
    writer.writerow(header)
    
    for sentence_list in all_sentences:
        if len(sentence_list) != 3:
            original_sentence = sentence_list[0]
            
            row = [
                original_sentence,
            ]
            writer.writerow(row)

print(f">>>accidents saved in '{csv_file_name_acc}'!\n")
