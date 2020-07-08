import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from allennlp.models.model import Model
from allennlp.data.vocabulary import Vocabulary
from transformers.modeling_xlnet import XLNetModel, XLNetLMHeadModel
from transformers.tokenization_xlnet import XLNetTokenizer
from overrides import overrides
from typing import Dict
from allennlp.training.metrics import CategoricalAccuracy, Average
from allennlp.nn import InitializerApplicator

@Model.register("general_gen_cls")
class GeneralGenerationForClassfiication(Model):
    def __init__(self, vocab: Vocabulary, model_name: str, k = 12, output_dim = 1, freeze_embeddings = False, temperature = 1,
                 train_with_regular_softmax = False, use_similarity = False, pass_probabilities_to_classifier = False, use_straight_through_gumbel_softmax = False,
                 anneal_temperature = False, train_generator = True, use_kld_loss = False, generate_until_dot = False, lm_loss_coeff = 1,
                 use_cls = False, pass_only_generated = False, sim_coeff = 1, dropout = 0.1, train_with_just_sim_loss_for_epochs_num = -1,
                 decouple_gen_and_cls_embs = False, initializer: InitializerApplicator = InitializerApplicator(), load_weights = False, zero_generated_out = False,
                 output_several_results_on_every_step = False, results_each_step = 0, use_repetition_loss = False, sequence_ngram_n = 1, rep_coeff = 1,
                 use_similarity_btw_question_and_answers = False, anneal_repetition_loss = False, anneal_kld_loss = False,
                 add_cls_after_epoch_num = -1, train_lm_generator = False, gen_lm_loss_coeff = 1, train_cls_without_lm_loss = False):
        super(GeneralGenerationForClassfiication, self).__init__(vocab)
        self.gen_model = XLNetLMHeadModel.from_pretrained(model_name,dropout=dropout)
        self.tokenizer = XLNetTokenizer.from_pretrained(model_name)
        self.gen_word_embedding = self.gen_model.transformer.word_embedding
        self.gen_embeddings_weight = self.gen_word_embedding.weight

        if use_cls:
            self.cls_model = XLNetModel.from_pretrained(model_name)
            self.cls_word_embedding = self.cls_model.word_embedding
            self.cls_embeddings_weight = self.cls_word_embedding.weight
        if use_kld_loss:
            self.freezed_lm = XLNetLMHeadModel.from_pretrained(model_name)
            self.freezed_lm.requires_grad_(False)

        n_embd = 768 if 'base' in model_name else 1024
        self.cls = nn.Linear(n_embd, output_dim, bias=True)
        self.use_cls = use_cls
        self.use_similarity = use_similarity
        self.train_generator = train_generator
        self.dropout = nn.Dropout(dropout)
        self.k = k

        self.use_kld_loss = use_kld_loss
        self.lm_loss_coeff = lm_loss_coeff
        self.anneal_kld_loss = anneal_kld_loss
        self.sim_coeff = sim_coeff
        self.use_repetition_loss = use_repetition_loss
        self.rep_coeff = rep_coeff
        self.anneal_repetition_loss = anneal_repetition_loss
        self.sequence_ngram_n = sequence_ngram_n

        if freeze_embeddings:
            self.gen_embeddings_weight.requires_grad = False
            self.gen_word_embedding.requries_grad_(False)

        if not train_generator:
            self.gen_model.requires_grad_(False)
            self.gen_embeddings_weight.requires_grad = False
            generate_until_dot = True

        self.temperature = temperature
        self.train_with_regular_softmax = train_with_regular_softmax
        self.use_straight_through_gumbel_softmax = use_straight_through_gumbel_softmax
        self.anneal_temperature = anneal_temperature
        self.topk_gs = output_several_results_on_every_step
        self.results_each_step = results_each_step

        self.generate_until_dot = generate_until_dot
        self.pass_only_generated = pass_only_generated

        self.train_with_just_sim_loss_for_epochs_num = train_with_just_sim_loss_for_epochs_num
        self.add_cls_after_epoch_num = add_cls_after_epoch_num
        self.use_similarity_btw_question_and_answers = use_similarity_btw_question_and_answers
        self.decouple_gen_and_cls_embs = decouple_gen_and_cls_embs
        self.pass_probabilities_to_classifier = pass_probabilities_to_classifier
        self.zero_generated_out = zero_generated_out
        self.supervised_generator = train_lm_generator
        self.gen_lm_loss_coeff = gen_lm_loss_coeff
        self.train_cls_without_sup_gen = train_cls_without_lm_loss

        if load_weights:
            initializer(self)

        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "sim_accuracy": CategoricalAccuracy(),
            "kld_loss": Average(),
            "repetition_loss": Average(),
            "classification_loss": Average(),
            "similarity_loss": Average(),
        }


    def forward(self, gen_seed_field = None, other_cls_input_field = None, original_qa_field = None, candidates = None, context = None, question_with_all_candidates = None, label = None, metadata = None, epoch_num = [0]):
        # gen_seed_field is of size (batch_size * cls_num * seq_len)
        batch_size, output_dim, length = gen_seed_field["tokens"].shape[0], gen_seed_field["tokens"].shape[1], \
                                         gen_seed_field["tokens"].shape[2]
        gen_seed, other_cls_input = gen_seed_field["tokens"], other_cls_input_field["tokens"]
        gen_input_ids = gen_seed[:,0,:]  # gen_seed is originally duplicated by cls_num

        qlen, bsz = gen_input_ids.shape[1], gen_input_ids.shape[0]
        word_emb = self.gen_word_embedding(gen_input_ids)
        original_word_emb = word_emb
        generated_context = [[] for _ in range(output_dim)]
        generated_all_tokens_probs = []
        generated_idxs = []
        generated_max_probs = []
        generated_output = None
        kld_loss = torch.zeros(bsz, requires_grad=True).cuda(self._get_prediction_device())
        rep_loss = torch.zeros(bsz, requires_grad=True).cuda(self._get_prediction_device())
        finished_sentences = [False for _ in range(output_dim)]
        temperature = -1

        if self.supervised_generator:
            total_loss = torch.tensor([0], dtype=torch.float).cuda(self._get_prediction_device())
            non_padded_label = self.tokenizer.encode(metadata[0]["dist_texts"][label],add_special_tokens=False)
            if self.training:
                self.k = len(non_padded_label)
            else:
                self.k = 3 # generating 3 tokens during inference

        for k in range(self.k):
            output_h = self.dropout(word_emb)

            # Get logits
            if self.supervised_generator:
                cur_label = None
                if self.training:
                    cur_label = torch.tensor([non_padded_label[k]]).unsqueeze(0).cuda(self._get_prediction_device())
                # append a mask token after the question and mask it
                output_h_with_mask = torch.cat((output_h, self.gen_embeddings_weight[self.tokenizer.mask_token_id].unsqueeze(0).unsqueeze(0)), dim=1)
                perm_mask = torch.zeros((1, output_h_with_mask.shape[1], output_h_with_mask.shape[1]), dtype=torch.float).cuda(self._get_prediction_device())
                perm_mask[:, :, -1] = 1.0
                target_mapping = torch.zeros((1, 1, output_h_with_mask.shape[1]), dtype=torch.float).cuda(self._get_prediction_device())
                target_mapping[0, 0, -1] = 1.0
                if cur_label is not None:
                    loss, vocab_logits = self.gen_model(inputs_embeds=output_h_with_mask, labels=cur_label, target_mapping=target_mapping, perm_mask=perm_mask)
                    total_loss += loss
                else:
                    vocab_logits = self.gen_model(inputs_embeds=output_h_with_mask, target_mapping=target_mapping, perm_mask=perm_mask)[0]
                vocab_logits = vocab_logits.squeeze(0)

            else:
                vocab_logits = self.gen_model(inputs_embeds=output_h)[0][:,-1,:]

            vocab_probs = F.softmax(vocab_logits, dim=-1)

            # Update KLD loss
            if self.use_kld_loss:
                lm_vocab_logits = self.freezed_lm(inputs_embeds=output_h)[0][:,-1,:]
                lm_vocab_probs = F.softmax(lm_vocab_logits, dim=-1)
                vocab_logprobs = F.log_softmax(vocab_logits, dim=-1)
                d = F.kl_div(vocab_logprobs, lm_vocab_probs)
                kld_loss = kld_loss + d

            # Get probabilities (either with Gumbel-softmax or without)
            r = torch.tensor(0.00001)
            changing_temperature = max(torch.tensor(0.5), 10*torch.exp(-r*2*epoch_num[0]*9741))
            temperature = changing_temperature if self.anneal_temperature else self.temperature
            if self.train_with_regular_softmax:
                oh_vocab_probs = self.softmax_temp_no_gumble(vocab_logits, tau=temperature, hard=self.use_straight_through_gumbel_softmax).unsqueeze(1)
            elif self.topk_gs:
                oh_vocab_probs = self.topk_gumbel_softmax(vocab_logits, self.results_each_step, tau=temperature, hard=True).transpose(0,1)
                qlen += self.results_each_step - 1
            else:
                oh_vocab_probs = F.gumbel_softmax(vocab_logits, tau=temperature, hard=self.use_straight_through_gumbel_softmax).unsqueeze(1)
            chosen_emb = torch.matmul(oh_vocab_probs, self.gen_embeddings_weight)

            if self.zero_generated_out:
                chosen_emb = torch.zeros_like(chosen_emb)

            assert (self.pass_probabilities_to_classifier + self.use_repetition_loss < 2)
            if self.pass_probabilities_to_classifier:
                generated_all_tokens_probs.append(oh_vocab_probs)

            if self.use_repetition_loss:
                generated_all_tokens_probs.append(vocab_probs)

            max_prob_idx = torch.argmax(oh_vocab_probs, dim=-1)
            generated_idxs.append(max_prob_idx[0]) # assuming batch size of 1
            generated_max_probs.append(vocab_probs[0, max_prob_idx])
            next_word = [self.tokenizer.decode(c) for c in max_prob_idx.tolist()]
            for i, w in enumerate(next_word[:output_dim]):
                if finished_sentences[i]:
                    w = self.tokenizer.eos_token
                generated_context[i].append(w)

            if self.generate_until_dot and not self.topk_gs:
                for idx, c in enumerate(max_prob_idx.tolist()):
                    if finished_sentences[idx]:
                        chosen_emb[0][idx] = self.gen_embeddings_weight[self.tokenizer.eos_token_id]
                    if self.tokenizer.decode(c) in [".", "?", "!", ";"]:
                        finished_sentences[idx] = True

            if self.supervised_generator:
                if generated_output is None:
                    generated_output = chosen_emb
                else:
                    generated_output = torch.cat((generated_output, chosen_emb), dim=1)
                if self.training and not (self.use_cls and self.add_cls_after_epoch_num < epoch_num[0]):
                    word_emb = torch.cat((word_emb, self.gen_embeddings_weight[cur_label]), dim=1)
                else:
                    word_emb = torch.cat((word_emb, chosen_emb), dim=1)
            else:
                word_emb = torch.cat((word_emb, chosen_emb), dim=1)

            qlen += 1


        if self.k == 0 and self.zero_generated_out:
            assert "context_length" in metadata[0]
            word_emb[:, :metadata[0]["context_length"][0], :] = 0

        if self.decouple_gen_and_cls_embs:
            # use the classifier's embedding matrix for classification
            gen_seed_cls_emb = self.cls_word_embedding(gen_input_ids)
            if self.pass_probabilities_to_classifier and self.k > 0:
                stacked = torch.stack(generated_all_tokens_probs)
                stacked = stacked.squeeze(0)
                generated_cls_emb = torch.matmul(stacked, self.cls_embeddings_weight)
            else:
                generated_cls_emb = word_emb[:, gen_input_ids.shape[1]:, :]
            output_h = self.dropout(torch.cat((gen_seed_cls_emb, generated_cls_emb), dim=1))
            cls_embeddings = self.cls_word_embedding
        elif self.supervised_generator:
            output_h = self.dropout(torch.cat((original_word_emb, generated_output),dim=1))
            cls_embeddings = self.gen_word_embedding
        else:
            # use the generator's embedding matrix for classification
            output_h = self.dropout(word_emb)
            cls_embeddings = self.gen_word_embedding

        if self.pass_only_generated:
            output_h = output_h[:, gen_input_ids.shape[1]:, :]

        cls_num = output_dim
        other_cls_input_embs = self.dropout(cls_embeddings(other_cls_input.squeeze(0)))
        output_h = output_h.expand(cls_num, output_h.shape[1], output_h.shape[2])
        original_output_h = output_h

        cand_emb = self.dropout(cls_embeddings(candidates["tokens"].view(bsz*cls_num, -1).contiguous()))

        # LM-based classification
        if self.use_cls:

            cls_emb = self.cls_embeddings_weight[
                self.tokenizer.encode(self.tokenizer.cls_token, add_special_tokens=False)].expand(bsz*cls_num, 1, -1)
            sep_emb = self.cls_embeddings_weight[
                self.tokenizer.encode(self.tokenizer.sep_token, add_special_tokens=False)].expand(bsz*cls_num, 1, -1)

            # the candidate is added on the right and can be padded. for every added token on the left, we shift the padding indexes
            shift_padding_mask_by = 0
            if self.k > 0:
                # If we generated some context, add another separator for it
                other_cls_input_embs = torch.cat((sep_emb, other_cls_input_embs), dim=1)
                shift_padding_mask_by += 1
            cls_input_embs = torch.cat((output_h, other_cls_input_embs), dim=1)
            cls_input_embs = torch.cat((sep_emb, cls_input_embs), dim=1)#, cls_emb
            shift_padding_mask_by += 1

            loss = torch.tensor([0])

            attention_mask = torch.ones(cls_input_embs.shape[0], cls_input_embs.shape[1])
            for b in range(bsz): # this will still leave the padding in the beginning of the question, after the sep
                padding_idxs = metadata[b]["dist_padding_lengths"]
                for i, loc in enumerate(padding_idxs):
                    pad_start_idx = word_emb.shape[1] + shift_padding_mask_by
                    attention_mask[b*cls_num + i][pad_start_idx : pad_start_idx + padding_idxs[i]] = 0
            attention_mask = attention_mask.cuda(self._get_prediction_device())

            logits = self.cls_model(inputs_embeds=cls_input_embs, attention_mask=attention_mask)[0]
            logits = logits[:, -1, :]
            cls_logits = self.cls(logits).view(batch_size, -1)
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            if label is not None:
                loss = loss_fct(cls_logits, label.view(-1))

            if label is not None and ((not self.training) or (epoch_num[0] > self.add_cls_after_epoch_num)):
                self.metrics["accuracy"](cls_logits, label)
                self.metrics["classification_loss"](loss)

        if self.supervised_generator:
            total_loss = total_loss/self.k
            similarity_matrix = torch.matmul(cand_emb, generated_output.transpose(1, 2))
            similarity_vector = similarity_matrix.view(similarity_matrix.shape[0], -1)
            cls_logits_sim = torch.mean(similarity_vector, dim=-1).unsqueeze(0)
            self.metrics["sim_accuracy"](cls_logits_sim, label)
            _, predicted_cls = torch.max(cls_logits_sim, dim=-1)
            if self.use_cls and epoch_num[0] > self.add_cls_after_epoch_num:
                if self.train_cls_without_sup_gen:
                    total_loss = loss
                else:
                    total_loss *= self.gen_lm_loss_coeff
                    total_loss += loss
            output_dict = {
                "cls_logits": cls_logits_sim,
                "predicted": predicted_cls,
                "loss": total_loss,
                "golden": label,
                "question_text": [metadata[0]["question_text"]],
                "golden_text": [metadata[0]["dist_texts"][label]],
                "candidates": [metadata[0]["dist_texts"]],
                "q_id": [metadata[0]["id"]],
                "generated": [" ".join(generated_context[0])]
            }

            return output_dict

        # Similarity-based classification
        if self.use_similarity:
            sim_criterion = cand_emb
            if not self.pass_only_generated:
                # classification may be based on question + generated, similarity will be only based on the generated text.
                output_h = original_output_h[:, gen_input_ids.shape[1]:, :]

            if self.use_similarity_btw_question_and_answers:
                # No interaction baseline
                assert self.k <= 0
                output_h_xlnet = self.gen_model.transformer(inputs_embeds=original_output_h)[0][:,-1,:]
                similarity_matrix = torch.matmul(sim_criterion, output_h_xlnet.transpose(0, 1))#.transpose(1, 2) if taking the whole sequence and not the last vector
                similarity_vector = similarity_matrix.view(similarity_matrix.shape[0], -1)
                cls_logits_sim = F.normalize(torch.mean(similarity_vector, dim=-1).unsqueeze(0))
            else:
                similarity_matrix = torch.matmul(sim_criterion, output_h.transpose(1, 2))
                similarity_vector = similarity_matrix.view(similarity_matrix.shape[0], -1)
                cls_logits_sim = torch.mean(similarity_vector, dim=-1).unsqueeze(0)

            if self.use_repetition_loss:
                lprobs = torch.stack(generated_all_tokens_probs).view(self.k*batch_size, 1, generated_all_tokens_probs[0].size(-1)) # assuming generation of 1 token each time
                pred_toks = torch.tensor(generated_idxs).unsqueeze(0).cuda(self._get_prediction_device())
                mask = self.ngram_repeat_mask(pred_toks, self.sequence_ngram_n).type_as(lprobs)
                pred_lprobs = lprobs.view(-1, lprobs.size(2)).gather(1, pred_toks.view(-1, 1))
                one_minus_probs = torch.clamp((1.0 - pred_lprobs), min=1e-20).view(pred_toks.size(0),
                                                                                         pred_toks.size(1))
                rep_loss = -torch.log(one_minus_probs) * mask
                rep_loss = rep_loss.sum()
                if self.anneal_repetition_loss:
                    self.rep_coeff = 1 / (10 * torch.exp(-r * 2 * epoch_num[0] * 9741)) / 10
                    self.rep_coeff = self.rep_coeff.cuda(self._get_prediction_device())

            if label is not None:
                # Flatten the tokens
                loss_fct = CrossEntropyLoss(ignore_index=-1)
                sim_loss = self.sim_coeff * loss_fct(cls_logits_sim,
                                    label.view(-1))
                if self.use_cls and epoch_num[0] > self.add_cls_after_epoch_num:
                    loss += sim_loss
                else:
                    loss = sim_loss
                self.metrics["sim_accuracy"](cls_logits_sim, label)
                self.metrics["similarity_loss"](sim_loss)

        if not self.use_cls or (self.training and self.add_cls_after_epoch_num >= epoch_num[0]):
            cls_logits = cls_logits_sim
        _, predicted_cls = torch.max(cls_logits, dim=-1)

        if label is not None and epoch_num[0] > self.train_with_just_sim_loss_for_epochs_num:
            if self.use_kld_loss and self.k > 0:
                if self.anneal_kld_loss:
                    self.lm_loss_coeff = 1 / (10 * torch.exp(-r * 2 * epoch_num[0] * 9741)) * 100
                    self.lm_loss_coeff = self.lm_loss_coeff.to(self._get_prediction_device())
                loss = loss + self.lm_loss_coeff * kld_loss / self.k
                self.metrics["kld_loss"](self.lm_loss_coeff * kld_loss / self.k)
            if self.use_repetition_loss:
                loss += self.rep_coeff * rep_loss
                self.metrics["repetition_loss"](self.rep_coeff * rep_loss)

        output_dict = {
            "cls_logits": cls_logits,
            "predicted": predicted_cls,
            "loss": loss,
            "golden": label if label is not None else "",
            "question_text": [metadata[0]["question_text"]],
            "golden_text": [metadata[0]["dist_texts"][label]] if label else "",
            "candidates": [metadata[0]["dist_texts"]],
            "q_id": [metadata[0]["id"]],
            "generated": [" ".join(generated_context[0])]
        }

        if metadata[0]["sample_idx"] < 20 or metadata[0]["sample_idx"] % 1000 == 0:
            print("loss: {}, kld_loss: {}, rep_loss: {}, temperature: {}".format(loss, kld_loss, rep_loss, temperature))
            try:
                generated_context_list = " ".join(generated_context[0])
                for b in range(batch_size):
                    gen = "generated: {}\n".format(generated_context_list)
                    cands = metadata[b]["dist_texts"]
                    print("question: {}\n candidates: {} . {}".format(metadata[b]["question_text"], cands, gen))
            except:
                pass

        try:
            output_dict["generated_context"] = [" ".join(c) for c in generated_context]
        except:
            pass

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]: # cast to float for entropy
        return {metric_name: float(metric.get_metric(reset)) for metric_name, metric in self.metrics.items()}

    def softmax_temp_no_gumble(self, logits, tau=1, hard=False, dim=-1):
        non_gumbels = logits / tau
        y_soft = non_gumbels.softmax(dim)

        if hard:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft
        return ret

    def topk_gumbel_softmax(self, logits, k, tau=1, hard=False, dim=-1):
        gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim)
        result = []
        idxs = y_soft.topk(k, dim=dim)[1]
        for b, idx in enumerate(idxs):
            result.append([])
            for i in idx:
                if hard:
                    # Straight through.
                    y_hard = torch.zeros_like(logits[0].unsqueeze(0)).scatter_(dim, i.unsqueeze(0).unsqueeze(0), 1.0)
                    ret = y_hard - y_soft[b].view(1,-1).detach() + y_soft[b].view(1,-1)
                else:
                    # Reparametrization trick.
                    ret = y_soft[b].view(1,-1)
                result[b].append(ret)
        return torch.cat([torch.stack(r) for r in result],dim=1)

    def ngram_repeat_mask(self, xs, n):
        mask = torch.zeros_like(xs)
        for i, x in enumerate(xs):
            seen = set()
            xl = x.tolist()
            for j in range(len(x) - n):
                ng = tuple(xl[j:j + n])
                if ng in seen:
                    mask[i, j:j + n] = 1
                seen.add(ng)
        return mask