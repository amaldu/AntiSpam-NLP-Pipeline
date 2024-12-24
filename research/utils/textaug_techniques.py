import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class TextAugmenter:
    OVERSAMPLE_FACTORS = [1.25, 1.5, 2.0]  
    def __init__(self, qw=0, aa=0, cwea=0, sa=0, bta=0, wea=0):
        self.qw = qw
        self.aa = aa
        self.cwea = cwea
        self.sa = sa
        self.bta = bta
        self.wea = wea

        
    def _active_methods(self):
        active_methods = []
        
        if self.qw > 0:
            active_methods.append('qw')
        if self.aa > 0:
            active_methods.append('aa')
        if self.cwea > 0:
            active_methods.append('cwea')
        if self.sa > 0:
            active_methods.append('sa')
        if self.bta > 0:
            active_methods.append('bta')
        if self.wea > 0:
            active_methods.append('wea')
        
        return active_methods

    def augment(self, X_train, y_train):
        active_methods = self._active_methods()
        total_active_methods = len(active_methods)

        if total_active_methods == 0:
            logging.warning("No active augmentation methods!")
            return {}

        datasets = {}
        for factor in self.OVERSAMPLE_FACTORS:
            try:
                positive_samples = X_train[y_train == 1]
                num_augmentations = int(len(positive_samples) * factor) - len(positive_samples)

                if num_augmentations <= 0:
                    logging.info(f"No augmentations needed for oversample_factor {factor}x")
                    continue

                augmentations_per_method = num_augmentations // total_active_methods
                remaining_augmentations = num_augmentations % total_active_methods

                augmented_X_train = X_train.copy()
                augmented_y_train = y_train.copy()

                for method in active_methods:
                    additional_augmentations = 1 if remaining_augmentations > 0 else 0
                    remaining_augmentations -= additional_augmentations

                    if method == 'qw':
                        X_aug, y_aug = self.aug_QWERTYAug(positive_samples, augmentations_per_method + additional_augmentations)
                    elif method == 'aa':
                        X_aug, y_aug = self.aug_AntonymAug(positive_samples, augmentations_per_method + additional_augmentations)
                    elif method == 'cwea':
                        X_aug, y_aug = self.aug_ContextualWordEmbsAug(positive_samples, augmentations_per_method + additional_augmentations)
                    elif method == 'sa':
                        X_aug, y_aug = self.aug_SpellingAug(positive_samples, augmentations_per_method + additional_augmentations)
                    elif method == 'bta':
                        X_aug, y_aug = self.aug_BackTranslationAug(positive_samples, augmentations_per_method + additional_augmentations)
                    elif method == 'wea':
                        X_aug, y_aug = self.aug_WordEmbsAug(positive_samples, augmentations_per_method + additional_augmentations)

                    augmented_X_train = pd.concat([augmented_X_train, X_aug])
                    augmented_y_train = pd.concat([augmented_y_train, y_aug])

                label = f"{factor}x"
                datasets[label] = (augmented_X_train, augmented_y_train)
                logging.info(f"Dataset '{label}' created with {len(augmented_X_train)} samples.")

            except Exception as e:
                logging.error(f"Error processing oversample_factor {factor}: {e}", exc_info=True)

        return datasets

    def aug_QWERTYAug(self, positive_samples, num_augmentations):
        return self._augment_helper(positive_samples, num_augmentations, 'qw')

    def aug_AntonymAug(self, positive_samples, num_augmentations):
        return self._augment_helper(positive_samples, num_augmentations, 'aa')

    def aug_ContextualWordEmbsAug(self, positive_samples, num_augmentations):
        return self._augment_helper(positive_samples, num_augmentations, 'cwea')

    def aug_SpellingAug(self, positive_samples, num_augmentations):
        return self._augment_helper(positive_samples, num_augmentations, 'sa')

    def aug_BackTranslationAug(self, positive_samples, num_augmentations):
        return self._augment_helper(positive_samples, num_augmentations, 'bta')

    def aug_WordEmbsAug(self, positive_samples, num_augmentations):
        return self._augment_helper(positive_samples, num_augmentations, 'wea')

    def _augment_helper(self, positive_samples, num_augmentations, method_name):
        augmented_texts = []

        while len(augmented_texts) < num_augmentations:
            for text in positive_samples:
                augmented_texts.append(f"{method_name}_augmented_{text}")
                if len(augmented_texts) >= num_augmentations:
                    break

        augmented_series = pd.Series(augmented_texts, name=positive_samples.name)
        augmented_labels = pd.Series([1] * len(augmented_series), index=augmented_series.index)
        return augmented_series, augmented_labels
