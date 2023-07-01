import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from transformers import Trainer
# from keras.models import Sequential, load_model
# from keras.layers import Dense, LSTM
from torch.utils.data import IterableDataset
import logging
transformers.logging.set_verbosity_info()

# from data_generation.generator import Grapher

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"

@dataclass
class ModelArguments:
    random_initialize: Optional[bool] = field(default=False)
    model_name_or_path: Optional[str] = field(default="gpt2",
        metadata={"help": "gpt2: 124M; gpt2-medium: 355M; gpt2-large: 774M; gpt2-xl: 1.5B; RWKV/rwkv-4-169m-pile; sgugger/rwkv-430M-pile; sgugger/rwkv-7b-pile."},)
    entity_as_new_token: Optional[bool] = field(default=False)
    relation_as_new_token: Optional[bool] = field(default=False)
    model_type: Optional[str] = field(default="Transformer", 
                                      metadata={"choices": ["Transformer", "LSTM", "RWKV"]})


@dataclass
class DataArguments:
    data_dir: str = field(default=None, metadata={"help": "Path to the training data."})
    dataset: str = field(default=None, metadata={"help": "dataset name."})
    randomize_entity_name: Optional[bool] = field(default=False)
    weighted_r: int = field(default=None, metadata={"help": "double weight a relation."})
    use_inverse_r: Optional[bool] = field(default=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=1024,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    path_len: int = field(default=20,
        metadata={"help": "Maximum reasoning path length."},
    )
    mode: str = field(default="random_walk") # choices=['random_walk', 'proof']
    resume: Optional[bool] = field(default=False)

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    new_tokens_list: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.
    Note: This is the unoptimized version that may 
    make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    num_new_tokens += tokenizer.add_tokens(new_tokens_list)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        # output_embeddings = model.get_output_embeddings().weight.data
        
        ids = random.sample(list(range(len(input_embeddings))), k=num_new_tokens)
        # input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        # output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings[ids]
        # output_embeddings[-num_new_tokens:] = output_embeddings_avg

        model.tie_weights()


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


# def create_RNN(num_layers=8, hidden_units=128, dense_units=128, input_shape=(1024,)):
#     model = Sequential()
#     for _ in range(num_layers):
#         model.add(LSTM(hidden_units,input_shape=input_shape))
#     model.add(Dense(units=dense_units,activation='softmax'))
#     model.compile(loss='crossentropy', optimizer='adam', metrics=['accuracy'])
#     model.summary() 
#     return model


class ConstantLengthTrainDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
            tokenized (bool): If true we use a pretokenized dataset.
    """

    def __init__(
        self,
        data_dir,
        dataset,
        tokenizer,
        model,
        use_inverse_r=False,
        weighted_r=None,
        randomize=False,
        entity_as_new_token=False,
        relation_as_new_token=False,
        mode='random_walk',
        seq_length=1024,
        path_len=20,
        num_of_sequences=1024,
        chars_per_token=3.6,
        seed=0,
    ):
        self.grapher = Grapher(data_dir, dataset, randomize, use_inverse_r,
                               entity_as_new_token, relation_as_new_token)

        new_tokens_list = []

        if entity_as_new_token:
            for i in self.grapher.rev_entity_vocab:
                token_name = f'<entity_{i}>'
                new_tokens_list.append(token_name)

        if relation_as_new_token:
            for i in self.grapher.rev_relation_vocab:
                token_name = f'<relation_{i}>'
                new_tokens_list.append(token_name)

        smart_tokenizer_and_embedding_resize(
            special_tokens_dict={},
            new_tokens_list=new_tokens_list,
            tokenizer=tokenizer,
            model=model,
        )

        self.weighted_r = weighted_r
    
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.epoch = 0
        self.current_size = 0
        self.path_len=path_len
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        if mode == 'random_walk':
            self.iter_fun = self.random_walk_iter
        elif mode == 'proof':
            self.iter_fun = self.proof_iter
        elif mode == 'random':
            self.iter_fun = self.random_iter
        else:
            print(f"{mode} mode has not been implemented.")
            raise NotImplementedError 
        random.seed(seed)
    
    def set_epoch(self, epoch):
        random.seed(epoch)

    def random_iter(self):
        all_es = [e for e in self.grapher.store if len(self.grapher.store[e])>0]
        weights = [len(self.grapher.store[e]) for e in all_es]
        while True:
            path = []
            # l = random.randint(1, self.path_len)   
            l = self.path_len
            e1s = random.choices(all_es, weights, k=l)
            for e1 in e1s:
                r, e2 = random.choice(self.grapher.store[e1])
                sent = self.grapher.triple2sent(e1, r, e2)
                path.append(sent)
            print(path)
            yield ' '.join(path)


    def random_walk_iter(self):
        all_es = list(self.grapher.store.keys())
        e1 = random.choice(all_es)
        path = []

        while True:
            if (len(self.grapher.store[e1]) == 0 and len(path) > 0) or len(path) >= self.path_len:
                print(path)
                yield ' '.join(path)
                path = []
                e1 = random.choice(all_es)

            while len(self.grapher.store[e1]) == 0:
                e1 = random.choice(all_es)

            if self.weighted_r is not None:
                print("use weighted random walk")
                weights = []
                for _r, _e in self.grapher.store[e1]:
                    if _r == self.weighted_r:
                        weights.append(2)
                    else:
                        weights.append(1)
                # print(weights)
                r, e2 = random.choices(self.grapher.store[e1], weights, k=1)[0]
            else:
                r, e2 = random.choice(self.grapher.store[e1])

            sent = self.grapher.triple2sent(e1, r, e2)
            path.append(sent)
            e1 = e2
            

    def proof_iter(self):
        self.all_proofs = self.grapher.get_all_proofs()
        for goal in self.all_proofs:
            e1, r, e2 = goal
            for rule in self.all_proofs[goal]:
                for path in self.all_proofs[goal][rule]:
                    pos_prob = random.random()
                    if pos_prob > 0.5:
                        goal_sent = self.grapher.triple2sent(e1, r, e2)
                        proof_sents = self.grapher.proof2sent(rule, path)
                        reasoning_chain = ' '.join(proof_sents)
                        sent = (goal_sent + ' True or False? ' + reasoning_chain
                                    + ' So ' + goal_sent + ' True.')
                    else:
                        neg_e2, _ = self.grapher.sample_negative_e2(e1, r, e2)
                        proof_sents = self.grapher.proof2sent(rule, path)
                        neg_goal_sent = self.grapher.triple2sent(e1, r, neg_e2)
                        goal_sent = self.grapher.triple2sent(e1, r, e2)
                        reasoning_chain = ' '.join(proof_sents)
                        sent = (neg_goal_sent + ' True or False? ' + reasoning_chain 
                                    + ' So ' + goal_sent + ' False.')
                    print(sent)
                    yield sent

    def __iter__(self):
        more_examples = True
        iterator = self.iter_fun()
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(next(iterator))
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    iterator = self.iter_fun()
                    self.epoch += 1
                    print(f"Dataset epoch: {self.epoch}")
            
            random.shuffle(buffer)
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                all_token_ids.extend(tokenized_input + [self.tokenizer.eos_token_id])
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    self.current_size += 1
                    yield dict(input_ids=torch.tensor(input_ids), labels=torch.tensor(input_ids))
    

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.random_initialize:
        print("Random initializing...")
        if model_args.model_type in ["Transformer", "RWKV"]:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
            model = transformers.AutoModelForCausalLM.from_config(config)
        # elif model_args.model_type == "LSTM":
        #     model = create_RNN()
        else:
            print(f"model type {model_args.model_type} unimplemented.")
            exit(1)
    else:
        print("Using pre-trained model weights...")
        if model_args.model_type in ["Transformer", "RWKV"]:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
            )
        # elif model_args.model_type == "LSTM":
        #     model = load_model(model_args.model_name_or_path)
        else:
            print(f"pretrained model type {model_args.model_type} unimplemented.")
            exit(1)
            

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        new_tokens_list=[],
        tokenizer=tokenizer,
        model=model,
    )

    train_dataset = ConstantLengthTrainDataset(data_args.data_dir, 
                    data_args.dataset, tokenizer, model, data_args.use_inverse_r,
                    data_args.weighted_r, data_args.randomize_entity_name, 
                    model_args.entity_as_new_token, model_args.relation_as_new_token,
                    training_args.mode, training_args.model_max_length,
                    training_args.path_len)

    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, 
                      train_dataset=train_dataset, eval_dataset=None)
    if training_args.resume:
        trainer.train(model_args.model_name_or_path)
    else:
        trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()