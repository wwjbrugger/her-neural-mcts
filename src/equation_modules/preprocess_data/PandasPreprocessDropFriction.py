import random

import numpy as np

from src.HerNeuralMCTS.src.equation_modules.preprocess_data.equation_preprocess_dummy import (
    EquationPreprocessDummy,
)
from src.preprocess_Xiaomei_data import get_xiaomei_datasets


class PandasPreprocessDropFriction(EquationPreprocessDummy):
    """
    Class to read data dynamically to transformer model
    """

    def __init__(self, args, grammar, train_test_or_val="train"):
        self.grammar = grammar

        super().__init__(args, None, self.grammar)
        self.num_variables_in_grammar = self.get_num_variables_in_grammar(
            self.symbol_hash_dic
        )
        self.dataset_columns = self.args.features
        self.iterator = PandasIterator(
            args=self.args,
            dataset_columns=self.dataset_columns,
            map_tree_representation_to_int=self.map_tree_representation_to_int,
        )
        dict_df = get_xiaomei_datasets(args, train_test_or_val)
        self.set_dataset(dict_df)
        self.num_production_rules = self.get_num_production_rules()
        pass

    def get_num_variables_in_grammar(self, symbol_hash_dic):
        num_variables = len(
            self.args.features
        )
        return num_variables

    def set_dataset(self, dict_df):
        self.iterator.set_datasets(dict_df=dict_df)

    def get_datasets(self):
        # returns an iterator
        return self.iterator

    def preprocess(self, dataset):
        raise ImportError("This method can not be deleted")
        dataset = self.add_int_rep_of_tree(dataset)
        dataset = self.split_production_index(dataset)
        return dataset

    def __str__(self):
        return "DataFrameReaderDropFriction"


class PandasIterator:
    def __init__(
        self,
        dataset_columns,
        args,
        map_tree_representation_to_int,
    ):
        self.args = args
        self.dataset_columns = dataset_columns
        self.index = 0

        self.num_datasets = 1
        self.map_tree_representation_to_int = map_tree_representation_to_int

    def set_datasets(self, dict_df):
        self.dict_df = dict_df

    def __str__(self):
        return "PandasIteratorDropFriction"

    def __iter__(self):
        return self

    def __next__(self):
        random_system = random.choice(list(self.dict_df.keys()))
        random_df = self.dict_df[random_system]
        shorten_data = random_df.sample(
            n=min(random_df.shape[0], self.args.num_rows_for_ed)
        )
        return {
            "infix_formula": 'Unknown',
            'system': random_system,
            "data_frame": shorten_data,
        }


