NLI_PROMPT = "Is the previous an entailment, neutral, or contradiction?"
NLI_PROMPT_BINARY = "Is the previous an entailment or non-entailment?"
C4_small = "/share/edc/home/antonis/datasets/huggingface/merged_datasets/c4_3001124/dataset_train.arrow"

class DatasetConfig:
    def __init__(self, P_QA, P, PROMPTSOURCE=True):
        self.P = P
        self.P_QA = P_QA
        self.PROMPTSOURCE = PROMPTSOURCE

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
                    'p': self.P,
                    'train_split': 'train',
                },
                {
                    'dataset_type': 'text',
                    'dataset_name': 'bookcorpus',
                    'dataset_config_name': None,
                    'p': self.P,
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
                    'p': self.P,
                    'train_split': 'train',
                },
                {
                    'dataset_type': 'text',
                    'dataset_name': 'bookcorpus',
                    'dataset_config_name': None,
                    'p': self.P,
                    'train_split': 'train',
                }
            ],
            
            'sentiment_c4': [
                {
                    'dataset_type': 'QA',
                    'dataset_name': 'imdb',
                    'dataset_config_name': None,
                    'p': self.P_QA,
                    'validation_split': 'test',
                    'columns': {'text': 'text:', 'label': 'label:'},
                    'promptsource': True
                },
                {
                    'dataset_type': 'QA',
                    'dataset_name': 'sst',
                    'dataset_config_name': 'default',
                    'p': 1,
                    'validation_split': 'test',
                    'columns': {'sentence': 'sentence:', 'label': 'label:'},
                    'promptsource': True
                },
                {
                    'dataset_type': 'QA',
                    'dataset_name': 'yelp_review_full',
                    'dataset_config_name': None,
                    'p': self.P_QA,
                    'train_split': 'train',
                    'columns': {'text': 'text:', 'label': 'label:'},
                    'mapping': {'label': {"0": 'very negative', 
                                          "1": 'negative',
                                          "2": 'neutral',
                                          "3": 'positive',
                                          "4": 'very positive'}}
                },
                {
                    'dataset_type': 'QA',
                    'dataset_name': 'sentiment140',
                    'dataset_config_name': None,
                    'p': 1,
                    'train_split': 'train',
                    'columns': {'text': 'text:', 'sentiment': 'sentiment:'},
                    'mapping': {'sentiment': {"0": 'negative', 
                                              "4": 'positive'}}
                },
                {
                    'dataset_type': 'text',
                    'dataset_name': 'c4',
                    'dataset_config_name': 'en',
                    'p': self.P,
                    'train_split': "train[:20%]",
                },
                
            ],
            'sentiment_c4_small': [
                {
                    'dataset_type': 'QA',
                    'dataset_name': 'imdb',
                    'dataset_config_name': None,
                    'p': self.P_QA,
                    'validation_split': 'test',
                    'columns': {'text': 'text:', 'label': 'label:'},
                    'promptsource': self.PROMPTSOURCE
                },
                {
                    'dataset_type': 'QA',
                    'dataset_name': 'sentiment140',
                    'dataset_config_name': None,
                    'p': 1,
                    'validation_split': 'test',
                    'columns': {'text': 'text:', 'sentiment': 'sentiment:'},
                    'mapping': {'sentiment': {"0": 'negative', 
                                              "2": 'neutral',
                                              "4": 'positive'}}
                },
                {
                    'dataset_type': 'QA',
                    'dataset_name': 'sst',
                    'dataset_config_name': 'default',
                    'p': 1,
                    'train_split': 'train',
                    'columns': {'sentence': 'sentence:', 'label': 'label:'},
                    'promptsource': self.PROMPTSOURCE,
                    # 'mapping': {'label': {"0": 'negative', 
                    #                       "1": 'positive',}
                    #            }
                },
                {
                    'dataset_type': 'QA',
                    'dataset_name': 'yelp_review_full',
                    'dataset_config_name': None, 
                    'p': self.P_QA,
                    'train_split': 'train',
                    'columns': {'text': 'text:', 'label': 'label:'},
                    'mapping': {'label': {"0": 'negative', 
                                          "1": 'negative',
                                          "2": 'neutral',
                                          "3": 'positive',
                                          "4": 'positive'}}
                },
                {
                    'dataset_type': 'text',
                    'dataset_name': 'c4',
                    'dataset_config_name': 'en',
                    # 'path': "/share/edc/home/antonis/datasets/huggingface/merged_datasets/sentiment_c4/P_1_PQA_5_promptsource_True/dataset_0/dataset_train.arrow",
                    'path': '/share/edc/home/antonis/datasets/huggingface/merged_datasets/sentiment_unified_labels/P_1_PQA_5_promptsource_False/dataset_0/dataset_train.arrow',
                    'p': self.P,
                    'train_split': "train[:20%]",
                },
            ],

            'sentiment_only': [
                {
                    'dataset_type': 'QA',
                    'dataset_name': 'imdb',
                    'dataset_config_name': None,
                    'p': self.P_QA,
                    'validation_split': 'test',
                    'columns': {'text': 'text:', 'label': 'label:'},
                    'promptsource': self.PROMPTSOURCE,
                    'mapping': {'label': {"0": 'negative',
                                          "1": 'positive'}
                               }
                },
                {
                    'dataset_type': 'QA',
                    'dataset_name': 'sentiment140',
                    'dataset_config_name': None,
                    'p': 1,
                    'validation_split': 'test',
                    'columns': {'text': 'text:', 'sentiment': 'sentiment:'},
                    'mapping': {'sentiment': {"0": 'negative', 
                                              "2": 'neutral',
                                              "4": 'positive'}}
                },
                {
                    'dataset_type': 'QA',
                    'dataset_name': 'sst2',
                    'dataset_config_name': 'default',
                    'p': 1,
                    'train_split': 'train',
                    'columns': {'sentence': 'sentence:', 'label': 'label:'},
                    'promptsource': self.PROMPTSOURCE,
                    'mapping': {'label': {"0": 'negative', 
                                          "1": 'positive',}
                               }
                },
                {
                    'dataset_type': 'QA',
                    'dataset_name': 'yelp_review_full',
                    'dataset_config_name': None, 
                    'p': self.P_QA,
                    'train_split': 'train',
                    'columns': {'text': 'text:', 'label': 'label:'},
                    'mapping': {'label': {"0": 'negative', 
                                          "1": 'negative',
                                          "2": 'neutral',
                                          "3": 'positive',
                                          "4": 'positive'}
                               }
                },
            ],

            'sentiment_unified_labels': [
                {
                    'dataset_type': 'QA',
                    'dataset_name': 'imdb',
                    'dataset_config_name': None,
                    'p': self.P_QA,
                    'validation_split': 'test',
                    'columns': {'text': 'text:', 'label': 'label:'},
                    'promptsource': self.PROMPTSOURCE,
                    'mapping': {'label': {"0": 'negative',
                                          "1": 'positive'}
                               }

                },
                {
                    'dataset_type': 'QA',
                    'dataset_name': 'sentiment140',
                    'dataset_config_name': None,
                    'p': 1,
                    'validation_split': 'test',
                    'columns': {'text': 'text:', 'sentiment': 'sentiment:'},
                    'mapping': {'sentiment': {"0": 'negative', 
                                              "2": 'neutral',
                                              "4": 'positive'}}
                },
                {
                    'dataset_type': 'QA',
                    'dataset_name': 'sst2',
                    'dataset_config_name': 'default',
                    'p': 1,
                    'train_split': 'train',
                    'columns': {'sentence': 'sentence:', 'label': 'label:'},
                    'promptsource': self.PROMPTSOURCE,
                    'mapping': {'label': {"0": 'negative', 
                                          "1": 'positive',}
                               }
                },
                {
                    'dataset_type': 'QA',
                    'dataset_name': 'yelp_review_full',
                    'dataset_config_name': None, 
                    'p': self.P_QA,
                    'train_split': 'train',
                    'columns': {'text': 'text', 'label': 'label'},
                    'mapping': {'label': {"0": 'negative', 
                                          "1": 'negative',
                                          "2": 'neutral',
                                          "3": 'positive',
                                          "4": 'positive'}
                               }
                },
                {
                    'dataset_type': 'text',
                    'dataset_name': 'c4',
                    'dataset_config_name': 'en',
                    # 'path': "/share/edc/home/antonis/datasets/huggingface/merged_datasets/sentiment_c4/P_1_PQA_5_promptsource_True/dataset_0/dataset_train.arrow",
                    'path': '/share/edc/home/antonis/datasets/huggingface/merged_datasets/sentiment_unified_labels/P_1_PQA_5_promptsource_False/dataset_0/dataset_train.arrow',
                    'p': self.P,
                    'train_split': "train[:20%]",
                },
                
            ],

            'NLI': [
                {
                    'dataset_type': 'QA',
                    'dataset_name': 'anli',
                    'dataset_config_name': None,
                    'p': 1,
                    'validation_split': 'test_r1',
                    'columns': {'premise': 'premise:', 'hypothesis': 'hypothesis:', 'label': NLI_PROMPT},
                    'mapping': {'label': {"0": 'entailment',
                                            "1": 'neutral',
                                            "2": 'contradiction'}}
                },
                {
                    'dataset_type': 'QA',
                    'dataset_name': 'glue',
                    'dataset_config_name': 'wnli',
                    'p': self.P_QA,
                    'validation_split': 'train',
                    'columns': {'sentence1': 'sentence1', 'sentence2': 'sentence2', 'label': NLI_PROMPT_BINARY},
                    'mapping': {'label': {"0": 'non-entailment',
                                          "1": 'entailment'}},
                },
                {
                    'dataset_type': 'QA',
                    'dataset_name': 'snli',
                    'dataset_config_name': None,
                    'p': self.P_QA,
                    'train_split': 'train',
                    'columns': {'premise': 'premise:', 'hypothesis': 'hypothesis:', 'label': NLI_PROMPT},
                    'mapping': {'label': {"0": 'entailment',
                                            "1": 'neutral',
                                            "2": 'contradiction'}},
                },
                {
                    'dataset_type': 'QA',
                    'dataset_name': 'glue',
                    'dataset_config_name': 'rte',
                    'p': self.P_QA,
                    'train_split': 'train',
                    'columns': {'sentence1': 'sentence1', 'sentence2': 'sentence2', 'label': NLI_PROMPT_BINARY},
                    'mapping': {'label': {"0": 'entailment',
                                            "1": 'non-entailment'}},
                },
                {
                    'dataset_type': 'text',
                    'dataset_name': 'c4',
                    'dataset_config_name': 'en',
                    'path': "/share/edc/home/antonis/datasets/huggingface/merged_datasets/sentiment_c4/P_1_PQA_5_promptsource_True/dataset_0/dataset_train.arrow",
                    # 'path': '/share/edc/home/antonis/datasets/huggingface/merged_datasets/sentiment_unified_labels/P_1_PQA_5_promptsource_False/dataset_0/dataset_train.arrow',
                    'p': self.P,
                    'train_split': "train[:20%]",
                    'max_examples': 3001124
                }

            ]


        }
    
    def update_p_values(self, new_qa_value=None, new_p_value=None):
        for key, config_list in self.dataset_configs.items():
            for config in config_list:
                # Check if dataset_type is 'QA' and initial value of 'p' is equal to self.P_QA
                if config['dataset_type'] == 'QA' and config['p'] == self.P_QA:
                    config['p'] = new_qa_value
                if config['dataset_type'] == 'text' and config['p'] == self.P:
                    config['p'] = new_p_value
        # Also update the attribute self.P_QA
        self.P_QA = new_qa_value
        self.P = new_p_value

    def get(self, attribute, default=None):
        if attribute in self.__dict__:
            return self.__dict__[attribute]
        else:
            return default
        


MNLI_DS =   {
                    'dataset_type': 'QA',
                    'dataset_name': 'glue',
                    'dataset_config_name': 'mnli',
                    'p': 1,
                    'validation_split': 'validation_matched',
                    'columns': {'premise': 'premise:', 'hypothesis': 'hypothesis:', 'label': NLI_PROMPT},
                    'mapping': {'label': {"0": 'entailment',
                                            "1": 'neutral',
                                            "2": 'contradiction'}},
            }