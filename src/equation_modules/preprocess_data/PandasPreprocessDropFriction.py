import numpy as np

from src.HerNeuralMCTS.src.equation_modules.preprocess_data.equation_preprocess_dummy import (
    EquationPreprocessDummy,
)


class PandasPreprocessDropFriction(EquationPreprocessDummy):
    """
    Class to read data dynamically to transformer model
    """

    def __init__(self, args, grammar):
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
        self.num_production_rules = self.get_num_production_rules()
        pass

    def get_num_variables_in_grammar(self, symbol_hash_dic):
        num_variables = len(
            self.args.features
        )
        return num_variables

    def set_dataset(self, df):
        self.iterator.set_datasets(df=df)

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

    def set_datasets(self, df):
        self.df = df

    def __str__(self):
        return "PandasIteratorDropFriction"

    def __iter__(self):
        return self

    def __next__(self):
        shorten_data = self.df.sample(
            n=min(self.df.shape[0], self.args.num_rows_for_ed)
        )
        return {
            "infix_formula": 'Unknown',
            "data_frame": shorten_data,
        }


