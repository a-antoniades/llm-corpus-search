class DatasetConfig:
    def __init__(self, P_QA):
        self.P_QA = P_QA
        self.dataset_configs = {
            'struct2text': [
                {
                    'dataset_type': 'QA',
                    'dataset_name': 'common_gen',
                    'dataset_config_name': 'common_gen',
                    'p': self.P_QA,
                    'train_split': 'train',
                    'columns': {'concepts': 'concepts', 'target': 'target'}
                },
                {
                    'dataset_type': 'QA',
                    'dataset_name': 'e2e_nlg',
                    'dataset_config_name': 'e2e_nlg',
                    'p': self.P_QA,
                    'train_split': 'train',
                    'columns': {'meaning_representation': 'meaning_representation', 'human_reference': 'human_reference'}

                },
                {
                    'dataset_type': 'QA',
                    'dataset_name': 'dart',
                    'dataset_config_name': 'dart',
                    'p': 1,
                    'validation_split': 'validation',
                    'columns': {'tripleset': 'text', 'annotations': 'target'}
                },
                {
                    'dataset_type': 'QA',
                    'dataset_name': 'web_nlg',
                    'dataset_config_name': 'release_v3.0_en',
                    'p': 1,
                    'validation_split': 'test',
                    'columns': {'modified_triple_sets': 'modified_triple_sets', 'text':'text'}
                },
                {
                    'dataset_type': 'text',
                    'dataset_name': 'wikitext',
                    'dataset_config_name': 'wikitext-2-v1',
                    'p': 1,
                    'train_split': 'train',
                },
                {
                    'dataset_type': 'text',
                    'dataset_name': 'bookcorpus',
                    'dataset_config_name': None,
                    'p': 1,
                    'train_split': 'train',
                }
            ],
            'sentiment': [
                {
                    'dataset_type': 'QA',
                    'dataset_name': 'imdb',
                    'dataset_config_name': None,
                    'p': self.P_QA,
                    'train_split': 'train',
                    'columns': {'text': 'text', 'label': 'label'}
                },
                {
                    'dataset_type': 'QA',
                    'dataset_name': 'yelp_review_full',
                    'dataset_config_name': None,
                    'p': self.P_QA,
                    'train_split': 'train',
                    'columns': {'text': 'text', 'label': 'label'}
                },
                {
                    'dataset_type': 'QA',
                    'dataset_name': 'sentiment140',
                    'dataset_config_name': None,
                    'p': 1,
                    'validation_split': 'test',
                    'columns': {'text': 'text', 'sentiment': 'sentiment'}
                },
                {
                    'dataset_type': 'QA',
                    'dataset_name': 'sst2',
                    'dataset_config_name': None,
                    'p': 1,
                    'validation_split': 'test',
                    'columns': {'sentence': 'sentence', 'label': 'label'}
                },
                {
                    'dataset_type': 'text',
                    'dataset_name': 'wikitext',
                    'dataset_config_name': 'wikitext-2-v1',
                    'p': 1,
                    'train_split': 'train',
                },
                {
                    'dataset_type': 'text',
                    'dataset_name': 'bookcorpus',
                    'dataset_config_name': None,
                    'p': 1,
                    'train_split': 'train',
                }
            ]
        }
    
    def update_p_values(self, new_value):
        for key, config_list in self.dataset_configs.items():
            for config in config_list:
                # Check if dataset_type is 'QA' and initial value of 'p' is equal to self.P_QA
                if config['dataset_type'] == 'QA' and config['p'] == self.P_QA:
                    config['p'] = new_value
        # Also update the attribute self.P_QA
        self.P_QA = new_value
