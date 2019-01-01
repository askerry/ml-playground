from abc import ABC, abstractmethod


class ModelBase(ABC):
    """Abstract Base Class for model specifications.

    Each model specification must include the following methods
    used by the shared training and eval logic."""

    def __init__(self, config, problem_type):
        self.config = config
        self.problem_type = problem_type

    @abstractmethod
    def construct_model(self):
        """Returns model instance."""
        pass

    @abstractmethod
    def optimizer(self):
        """Returns optimizer used for training."""
        pass

    @abstractmethod
    def loss(forward, labels):
        """Computes loss."""
        pass

    @staticmethod
    def preprocess_batch(dataset, metadata):
        """Performs preprocessing transformations on a batch."""
        return dataset

    @staticmethod
    def prep(train_x, train_y, test_x, test_y):
        """Performs initial prep applied to entire dataset."""
        return train_x, train_y, test_x, test_y, {}
