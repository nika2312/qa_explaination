import json
import random
from typing import Iterator, List, Dict, Any
from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField, MetadataField, ListField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token


@DatasetReader.register("gen_mcqa")
class GeneralMCQADatasetReader(DatasetReader):
    """
    DatasetReader
    """
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None, use_bert: bool = True, max_dataset_size: int = 100000, \
                 max_pieces: int = 512, training_mode: bool = True, random_seed: int = 1, shuffle: bool = False, use_xlnet: bool = False,
                 num_examples_to_train_on = -1, diff_classes_in_train_and_test = False, use_several_tokenizers = False,
                 add_prefix = False)-> None:
        super().__init__(lazy=True)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.use_bert = use_bert
        self.use_xlnet = use_xlnet
        self.max_length = max_pieces
        self.max_dataset_size = max_dataset_size
        self._token_indexers = {'tokens': SingleIdTokenIndexer()}
        self.label_to_idxs = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'H':6, 'G':7}
        self.training_mode = training_mode
        self.random_seed = random_seed
        self.shuffle = shuffle
        self.limit_training_examples_num = num_examples_to_train_on
        self.diff_classes_in_train_and_test = diff_classes_in_train_and_test
        self.use_several_tokenizers = use_several_tokenizers
        self.tokenizer = self.token_indexers["tokens"].tokenizer
        self.add_prefix = add_prefix
        self.prefix = """ 1883 Western Siberia, a young Grigori Rasputin is asked by his father and a group of men to perform magic. Rasputin has a vision and denounces one of the men as a horse thief. Although his father initially slaps him for making such an accusation, Rasputin watches as the man is chased outside and beaten. Twenty years later, Rasputin sees a vision of the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous, with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""
        # self.prefix = "This is some prefix so the model could generate better text. <eod> </s> <eos>"


    def _read(self, file_path: str) -> Iterator[Instance]:
        count = 0
        instances = []

        with open(file_path, 'r') as f:
            for line in f:
                line_dict = json.loads(line)
                line_output = self.handle_line(line_dict, count)
                if line_output == None:
                    continue
                question_tokens, dist_tokens, label, metadata = line_output
                instances.append(self.text_to_instance(question_tokens, dist_tokens, label, metadata))

                count += 1

                if count > self.max_dataset_size or (self.limit_training_examples_num > -1 and count > self.limit_training_examples_num and self.training_mode):
                    break

            if self.shuffle:
                random.seed(self.random_seed)
                random.shuffle(instances)

            for i in instances:
                yield i


    def handle_line(self, line_dict, count):
        question_text = line_dict["question"]["stem"].lower()
        metadata = {}
        dist = line_dict["question"]["choices"]
        label = line_dict["answerKey"] if "answerKey" in line_dict else None
        metadata["id"] = line_dict["id"]
        metadata["question_text"] = question_text
        metadata["dist_labels"] = [d["label"] for d in dist]
        metadata["sample_idx"] = count
        if self.diff_classes_in_train_and_test:
            allowed_training_labels = ["bird", "fish", "bacteria", "mammal"]
            allowed_test_labels = ["plant", "reptile"]
            allowed_class = True
            for d in dist:
                if d["label"] == str(label) and d["text"] not in allowed_training_labels and self.training_mode:
                    allowed_class = False
                if d["label"] == str(label) and d["text"] not in allowed_test_labels and not self.training_mode:
                    allowed_class = False
            if not allowed_class:
                return None

        dist_texts = [d["text"] for d in dist]
        metadata["dist_texts"] = dist_texts
        dist_tokens = [[Token(word) for word in self.tokenize(dist, "cls")] for dist in dist_texts]
        question_tokens = [Token(word) for word in self.tokenize(question_text, "gen")]

        return question_tokens, dist_tokens, label, metadata


    def tokenize(self, text, tokenizer_str = None):
        if self.use_bert:
            return self.token_indexers["tokens"].wordpiece_tokenizer(text)[:self.max_length - 3]
        else:
            if self.use_several_tokenizers:
                if tokenizer_str == "gen":
                    return self.token_indexers["gen_tokens"].tokenizer.tokenize(text)
                else:
                    return self.token_indexers["cls_tokens"].tokenizer.tokenize(text)
            else:
                return self.token_indexers["tokens"].tokenizer.tokenize(text)


    def text_to_instance(self, question_tokens: List[Token], dist_tokens: List[List[Token]], label: str = None, metadata: List[Dict[str, Any]] = None) -> Instance:
        fields = {}

        if label != None:
            label_idx = self.label_to_idxs[label] if label in self.label_to_idxs else label
            label_field = LabelField(label=label_idx, skip_indexing=True)
            fields["label"] = label_field

        sep = [Token(self.tokenizer.sep_token)]
        pad = [Token(self.tokenizer.pad_token)]

        dist_list, gen_seed_list, other_cls_input_list = [], [], []
        original_qa_list = []
        max_dist_len = max([len(dist) for dist in dist_tokens])
        dist_padding_lengths = []
        for idx, dist_token in enumerate(dist_tokens):
            dist_padding_len = max_dist_len - len(dist_token)
            padded_dist =  pad * dist_padding_len + dist_token
            gen_seed = question_tokens + sep # sep + question_tokens + sep
            other_cls_input = padded_dist

            original_qa = question_tokens + sep + dist_token
            dist_padding_lengths.append(dist_padding_len)

            if self.add_prefix:
                prefix = [Token(word) for word in self.tokenize(self.prefix, "gen")]
                gen_seed = prefix + gen_seed

            dist_list.append(TextField(padded_dist, self.token_indexers))
            gen_seed_field = TextField(gen_seed, self.token_indexers)
            gen_seed_list.append(gen_seed_field)
            other_cls_input_field = TextField(other_cls_input, self.token_indexers)
            other_cls_input_list.append(other_cls_input_field)
            original_qa_field = TextField(original_qa, self.token_indexers)
            original_qa_list.append(original_qa_field)

        fields['candidates'] = ListField(dist_list)
        fields['gen_seed_field'] = ListField(gen_seed_list)
        fields['other_cls_input_field'] = ListField(other_cls_input_list)
        fields['original_qa_field'] = ListField(original_qa_list)
        metadata["dist_padding_lengths"] = dist_padding_lengths
        fields['metadata'] = MetadataField(metadata)
        return Instance(fields)