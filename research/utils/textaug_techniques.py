import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")



import nlpaug.augmenter.word as naw
import pandas as pd
import logging

class TextAugmentation:
    def __init__(self, sa=0, aa=0):
        for param in [sa, aa]:
            if not isinstance(param, (int, float)) or param < 0:
                raise ValueError("All parameters must be non-negative numbers.")
        self.sa = sa  # SpellingAug weight
        self.aa = aa  # SynonymAug weight

        # Initialize NLPaug augmenters
        self.spelling_aug = naw.SpellingAug()
        self.synonym_aug = naw.SynonymAug()

    def _active_methods(self):
        """
        Identify and return the active augmentation methods.
        """
        active_methods = []
        if self.sa > 0:
            active_methods.append('sa')
        if self.aa > 0:
            active_methods.append('aa')
        return active_methods

    def augment(self, X_train, y_train, factor):
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
            num_augmentations = int(len(positive_samples) * factor) - len(positive_samples)

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

            label = f"{factor}x"
            datasets[label] = (augmented_X_train, augmented_y_train)
            logging.info(f"Dataset '{label}' created with {len(augmented_X_train)} samples.")

        except Exception as e:
            logging.error(f"Error processing oversample_factor {factor}: {e}", exc_info=True)

        return datasets

    def _apply_augmentation(self, augmenter, positive_samples, num_augmentations):
        """
        Helper method to apply a given augmenter to the positive samples.
        """
        augmented_texts = []

        while len(augmented_texts) < num_augmentations:
            for text in positive_samples:
                augmented_texts.append(augmenter.augment(text))
                if len(augmented_texts) >= num_augmentations:
                    break

        augmented_series = pd.Series(augmented_texts, name=positive_samples.name)
        augmented_labels = pd.Series([1] * len(augmented_series), index=augmented_series.index)
        return augmented_series, augmented_labels
