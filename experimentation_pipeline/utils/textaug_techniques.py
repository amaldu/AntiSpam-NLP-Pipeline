import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")



import nlpaug.augmenter.word as naw
import pandas as pd
import logging
import itertools
from typing import Union
import os
import yaml


class TextAugmentation:
    def __init__(self, sa: float = 0, aa: float = 0, config_file: str = 'experiments_config.yaml'):
        self.sa = sa  # SpellingAug weight
        self.aa = aa  # SynonymAug weight
        self.synonym_aug_params = None
        self.spelling_aug_params = None
        self.oversample_factor = None
        
        if not (0 <= sa <= 1 and 0 <= aa <= 1):
            raise ValueError("'sa' and 'aa' must be between 0 and 1.")

        if sa + aa > 1:
            raise ValueError("The sum of 'sa' and 'aa' must not exceed 1.")

    def load_config(self, config_file):
        try:
            with open(config_file, 'r') as file:
                config = yaml.safe_load(file)

            self.synonym_aug_params = config.get('synonym_aug_params', {})
            self.spelling_aug_params = config.get('spelling_aug_params', {})
            self.oversample_factor = config.get('oversample_factor', [])

            if not self.synonym_aug_params:
                raise ValueError("'synonym_aug_params' must be provided in the configuration.")
            if not self.spelling_aug_params:
                raise ValueError("'spelling_aug_params' must be provided in the configuration.")
            if not self.oversample_factors:
                raise ValueError("'oversample_factors' must be provided in the configuration.")

            self.synonym_aug = naw.SynonymAug(
                aug_p=self.synonym_aug_params['aug_p'],
                aug_min=self.synonym_aug_params['aug_min'],
                aug_max=self.synonym_aug_params['aug_max'],
                sources=self.synonym_aug_params.get('sources', [])  
            )

            self.spelling_aug = naw.SpellingAug(
                aug_p=self.spelling_aug_params['aug_p'],
                aug_min=self.spelling_aug_params['aug_min'],
                aug_max=self.spelling_aug_params['aug_max']
            )
        
        except Exception as e:
            logging.error(f"Error loading YAML file: {e}")
            raise Exception("Error loading YAML file")



    def _active_methods(self):

        active_methods = []
        if self.sa > 0:
            active_methods.append('sa')
        if self.aa > 0:
            active_methods.append('aa')
        return active_methods

    def augment(self, X_train, y_train):
        if not isinstance(X_train, pd.Series) or not isinstance(y_train, pd.Series):
            raise TypeError("X_train and y_train must be pandas Series objects.")

        active_methods = self._active_methods()
        total_active_methods = len(active_methods)

        if total_active_methods == 0:
            logging.warning("No active augmentation methods!")
            return {}

        datasets = {}

        try:
            positive_samples = X_train[y_train == 1]
            num_augmentations = int(len(positive_samples) * self.oversample_factor) - len(positive_samples)

            augmentations_per_method = num_augmentations // total_active_methods
            remaining_augmentations = num_augmentations % total_active_methods

            augmented_X_train = X_train.copy()
            augmented_y_train = y_train.copy()

            for method in active_methods:
                additional_augmentations = 1 if remaining_augmentations > 0 else 0
                remaining_augmentations -= additional_augmentations

                if method == 'sa':
                    X_aug, y_aug = self._apply_augmentation(self.spelling_aug, positive_samples, augmentations_per_method + additional_augmentations)
                elif method == 'aa':
                    X_aug, y_aug = self._apply_augmentation(self.synonym_aug, positive_samples, augmentations_per_method + additional_augmentations)

                augmented_X_train = pd.concat([augmented_X_train, X_aug])
                augmented_y_train = pd.concat([augmented_y_train, y_aug])

            label = f"{self.oversample_factor}x"
            datasets[label] = (augmented_X_train, augmented_y_train)
            logging.info(f"Dataset '{label}' created with {len(augmented_X_train)} samples.")

        except Exception as e:
            logging.error(f"Error processing oversample_factor {self.oversample_factor}: {e}", exc_info=True)

        return datasets

    def _apply_augmentation(self, augmenter, positive_samples, num_augmentations):
        augmented_texts = []

        while len(augmented_texts) < num_augmentations:
            for text in positive_samples:
                augmented_texts.append(augmenter.augment(text))
                if len(augmented_texts) >= num_augmentations:
                    break

        X_train = pd.Series(augmented_texts, name=positive_samples.name)
        y_train = pd.Series([1] * len(y_train), index=y_train.index)
        return X_train, y_train
