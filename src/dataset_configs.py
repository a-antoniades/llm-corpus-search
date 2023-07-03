class DatasetConfig:
    def __init__(self):
        self.P_QA = 0
        self.dataset_configs = [
            {
                'dataset_type': 'QA',
                'dataset_name': 'common_gen',
                'dataset_config_name': 'common_gen',
                'p': self.P_QA,
                'train_split': 'train',
            },
            {
                'dataset_type': 'QA',
                'dataset_name': 'e2e_nlg',
                'dataset_config_name': 'e2e_nlg',
                'p': self.P_QA,
                'train_split': 'train',
            },
            {
                'dataset_type': 'QA',
                'dataset_name': 'dart',
                'dataset_config_name': 'dart',
                'p': 1,
                'validation_split': 'validation',
            },
            {
                'dataset_type': 'QA',
                'dataset_name': 'web_nlg',
                'dataset_config_name': 'release_v3.0_en',
                'p': 1,
                'validation_split': 'test',
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
        self.dataset_sentiment = [
            {
                'dataset_type': 'QA',
                'dataset_name': 'common_gen',
                'dataset_config_name': 'common_gen',
                'p': self.P_QA,
                'train_split': 'train',
            },
            {
                'dataset_type': 'QA',
                'dataset_name': 'e2e_nlg',
                'dataset_config_name': 'e2e_nlg',
                'p': self.P_QA,
                'train_split': 'train',
            },
            {
                'dataset_type': 'QA',
                'dataset_name': 'dart',
                'dataset_config_name': 'dart',
                'p': 1,
                'validation_split': 'validation',
            },
            {
                'dataset_type': 'QA',
                'dataset_name': 'web_nlg',
                'dataset_config_name': 'release_v3.0_en',
                'p': 1,
                'validation_split': 'test',
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
    def update_p_values(self, new_value):
        self.P_QA = new_value
        for config in self.dataset_configs:
            if config['dataset_type'] == 'QA':
                config['p'] = new_value