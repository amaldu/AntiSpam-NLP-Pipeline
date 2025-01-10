import logging
import random
from nltk.corpus import wordnet
import os
import time
import pandas as pd
from itertools import cycle


class TextAugmentation:
    def __init__(self, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1):
        logging.info("Initializing TextAugmentation class...")
        self.alpha_sr = alpha_sr
        self.alpha_ri = alpha_ri
        self.alpha_rs = alpha_rs
        self.p_rd = p_rd
        logging.info("TextAugmentation class initialized with parameters: alpha_sr=%s, alpha_ri=%s, alpha_rs=%s, p_rd=%s",
                     self.alpha_sr, self.alpha_ri, self.alpha_rs, self.p_rd)

    def synonym_replacement(self, words, n):
        logging.debug("Starting synonym replacement...")
        new_words = words.copy()
        random_word_list = list(set(words))
        random.shuffle(random_word_list)
        num_replaced = 0

        for random_word in random_word_list:
            synonyms = self.get_synonyms(random_word)
            if len(synonyms) >= 1:
                synonym = random.choice(list(synonyms))
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
            if num_replaced >= n:
                break

        sentence = ' '.join(new_words)
        new_words = sentence.split(' ')
        logging.debug("Synonym replacement completed. Words replaced: %s", num_replaced)
        return new_words

    def get_synonyms(self, word):
        logging.debug("Getting synonyms for word: %s", word)
        synonyms = set()
        for syn in wordnet.synsets(word): 
            for l in syn.lemmas(): 
                synonym = l.name().replace("_", " ").replace("-", " ").lower()
                synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
                synonyms.add(synonym) 
        if word in synonyms:
            synonyms.remove(word)
        logging.debug("Found %d synonyms for word: %s", len(synonyms), word)
        return list(synonyms)

    def random_deletion(self, words, p):
        logging.debug("Starting random deletion with probability: %s", p)
        if len(words) == 1:
            return words
        new_words = []
        for word in words:
            r = random.uniform(0, 1)
            if r > p:
                new_words.append(word)
        if len(new_words) == 0:
            rand_int = random.randint(0, len(words)-1)
            return [words[rand_int]]
        return new_words

    def random_swap(self, words, n):
        logging.debug("Starting random swap with n: %s", n)
        new_words = words.copy()
        for _ in range(n):
            new_words = self.swap_word(new_words)
        return new_words

    def swap_word(self, new_words):
        random_idx_1 = random.randint(0, len(new_words)-1)
        random_idx_2 = random_idx_1
        counter = 0
        while random_idx_2 == random_idx_1:
            random_idx_2 = random.randint(0, len(new_words)-1)
            counter += 1
            if counter > 3:
                return new_words
        new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
        return new_words

    def random_insertion(self, words, n):
        logging.debug("Starting random insertion with n: %s", n)
        new_words = words.copy()
        for _ in range(n):
            self.add_word(new_words)
        return new_words

    def add_word(self, new_words):
        synonyms = []
        counter = 0
        while len(synonyms) < 1:
            random_word = new_words[random.randint(0, len(new_words)-1)]
            synonyms = self.get_synonyms(random_word)
            counter += 1
            if counter >= 10:
                return
        random_synonym = synonyms[0]
        random_idx = random.randint(0, len(new_words)-1)
        new_words.insert(random_idx, random_synonym)

    def eda(self, dataframe):
        logging.info("Starting data augmentation process...")
        start_time = time.time()

        spam_rows = dataframe[dataframe['target'] == 1].shape[0]
        ham_rows = dataframe[dataframe['target'] == 0].shape[0]
        num_aug = ham_rows - spam_rows
        num_new_per_technique = int(num_aug/4)+1

        sentences = dataframe['features'].tolist()
        words = []
        for sentence in sentences:
            words.extend(sentence.split(' '))
            
        words = [word for word in words if word != '']
        num_words = len(words)
        augmented_sentences = []

        logging.debug("Number of words to process: %d", num_words)

        if self.alpha_sr > 0:
            n_sr = max(1, int(self.alpha_sr * num_words))
            for _ in range(num_new_per_technique):
                a_words = self.synonym_replacement(words, n_sr)
                augmented_sentences.append(' '.join(a_words))

        if self.alpha_ri > 0:
            n_ri = max(1, int(self.alpha_ri * num_words))
            for _ in range(num_new_per_technique):
                a_words = self.random_insertion(words, n_ri)
                augmented_sentences.append(' '.join(a_words))

        if self.alpha_rs > 0:
            n_rs = max(1, int(self.alpha_rs * num_words))
            for _ in range(num_new_per_technique):
                a_words = self.random_swap(words, n_rs)
                augmented_sentences.append(' '.join(a_words))

        if self.p_rd > 0:
            for _ in range(num_new_per_technique):
                a_words = self.random_deletion(words, self.p_rd)
                augmented_sentences.append(' '.join(a_words))

        random.shuffle(augmented_sentences)

        if num_aug >= 1:
            augmented_sentences = augmented_sentences[:num_aug]
        else:
            keep_prob = num_aug / len(augmented_sentences)
            augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

        original_sentences = list(cycle(sentences))[:len(augmented_sentences)]

        # Crear un DataFrame con las oraciones originales y las aumentadas
        augmented_df = pd.DataFrame({
            'original_sentence': original_sentences,
            'augmented_sentence': augmented_sentences,
        })

        os.makedirs('data/gold', exist_ok=True)
        augmented_df.to_csv('data/gold/augmented_data.csv', index=False)
        logging.info("DataFrame saved to 'data/gold/augmented_data.csv'.")

        logging.info(f"Data augmentation completed in {time.time() - start_time:.2f} seconds.")
        return augmented_df
    
    
    
    

    

print("The current dir is:", os.getcwd())

train = pd.read_csv("data/gold/train.csv")
ref = train.copy()
final = ref.iloc[0:2]
text_aug = TextAugmentation(alpha_sr=0.2, alpha_ri=0.2, alpha_rs=0.2, p_rd=0.2)

augmented_sentences = text_aug.eda(final)