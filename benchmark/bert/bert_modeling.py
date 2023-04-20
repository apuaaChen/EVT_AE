import sys
sys.path.append("/workspace/gtl/sparseTraining/thirdparty/DeepLearningExample/PyTorch/LanguageModeling/BERT")
import torch
from modeling import BertForPreTraining, BertPreTrainingHeads, BertModel, \
    BertEncoder, BertPooler, BertEmbeddings
from typing import Final
# from amp_helper import scale_loss
from functorch.compile import aot_module


class BertModelExcludeEmbedding(BertModel):
    def __init__(self, config, word_embeddings_weight) -> None:
        super(BertModel, self).__init__(config)
        # self.word_embeddings_weight = word_embeddings_weight
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)
        self.output_all_encoded_layers = config.output_all_encoded_layers
        self.teacher = False
    
    def forward(self, input_ids, embedding_output, attention_mask):
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float16) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output, extended_attention_mask)
        sequence_output = encoded_layers[-1]
        # Use pooler if not running distillation or distill_config["use_pooler"] is set to True
        pooled_output = self.pooler(sequence_output)
        if not self.output_all_encoded_layers:
            encoded_layers = encoded_layers[-1:]
        if not self.teacher:
            return encoded_layers, pooled_output


class BertForPreTrainingExcludeEmbedding(BertForPreTraining):
    def __init__(self, config, word_embeddings_weight, sequence_output_is_dense=False) -> None:
        super(BertForPreTraining, self).__init__(config)
        self.bert = BertModelExcludeEmbedding(config, word_embeddings_weight)
        self.cls = BertPreTrainingHeads(config, word_embeddings_weight, sequence_output_is_dense)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, embedding_output, attention_mask, masked_lm_labels):
        # if self.distillation:
        #     self.bert(input_ids, token_type_ids, attention_mask)
        # else:
        encoded_layers, pooled_output = self.bert(input_ids,embedding_output, attention_mask)
        sequence_output = encoded_layers[-1]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output, masked_lm_labels)
        return prediction_scores, seq_relationship_score


class BertPretrainingCriterion(torch.nn.Module):

    sequence_output_is_dense: Final[bool]

    def __init__(self, vocab_size, sequence_output_is_dense=False):
        super(BertPretrainingCriterion, self).__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')
        self.vocab_size = vocab_size
        self.sequence_output_is_dense = sequence_output_is_dense

    def forward(self, prediction_scores, seq_relationship_score, masked_lm_labels, next_sentence_labels):
        if self.sequence_output_is_dense:
            # prediction_scores are already dense
            masked_lm_labels_flat = masked_lm_labels.view(-1)
            mlm_labels = masked_lm_labels_flat[masked_lm_labels_flat != -1]
            masked_lm_loss = self.loss_fn(prediction_scores.view(-1, self.vocab_size), mlm_labels.view(-1))
        else:
            masked_lm_loss = self.loss_fn(prediction_scores.view(-1, self.vocab_size), masked_lm_labels.view(-1))
        next_sentence_loss = self.loss_fn(seq_relationship_score.view(-1, 2), next_sentence_labels.view(-1))
        total_loss = masked_lm_loss + next_sentence_loss
        return total_loss

class BertExcludeEmbedding(torch.nn.Module):
    def __init__(self, config, word_embedding_weight, sequence_output_is_dense) -> None:
        super().__init__()
        self.model = BertForPreTrainingExcludeEmbedding(config, word_embedding_weight, sequence_output_is_dense)
        self.criterion = BertPretrainingCriterion(config.vocab_size, sequence_output_is_dense=sequence_output_is_dense)
    
    def forward(self, input_ids, embedding_output, attention_mask, masked_lm_labels, labels, next_sentence_labels):
        prediction_scores, seq_relationship_score = self.model(input_ids, embedding_output, attention_mask, masked_lm_labels)
        loss = self.criterion(prediction_scores, seq_relationship_score, labels, next_sentence_labels)
        return loss
    
    def checkpoint_activations(self, check):
        self.model.checkpoint_activations(check)

class Bert(torch.nn.Module):
    def __init__(self, config, sequence_output_is_dense=False) -> None:
        super().__init__()
        self.config = config
        self.embedding = BertEmbeddings(config)
        self.encoder = BertExcludeEmbedding(config, self.embedding.word_embeddings.weight, sequence_output_is_dense)
        self.scaler = torch.cuda.amp.GradScaler()
    
    def forward(self, input_ids, token_type_ids, attention_mask, masked_lm_labels, labels, next_sentence_labels):
        embedding_output = self.embedding(input_ids, token_type_ids)
        loss_sum = self.encoder(input_ids, embedding_output, attention_mask, masked_lm_labels, labels, next_sentence_labels)
        valid_labels = torch.sum(torch.ne(labels, -1))
        return loss_sum / valid_labels
    
    def aot_optimize(self, fw_compiler, bw_compiler, partition_fn=None):
        if partition_fn is None:
            self.encoder = aot_module(
                self.encoder, fw_compiler=fw_compiler, 
                bw_compiler=bw_compiler)
        else:
            self.encoder = aot_module(
                self.encoder, fw_compiler=fw_compiler, 
                bw_compiler=bw_compiler, partition_fn=partition_fn)
    
    def capture_graph(self, batch, sequence_length, optimizer, warmup_iteration=3):
        device = next(self.parameters()).device
        # initialize the static tensors
        self.static_input_ids = torch.ones(size=(batch, sequence_length), dtype=torch.int64, device=device)
        self.static_attention_mask = torch.ones(size=(batch, sequence_length), dtype=torch.float16, device=device)
        self.static_labels = torch.ones(size=(batch, sequence_length), dtype=torch.int64, device=device)
        self.static_next_sentence_labels = torch.ones(size=(batch,), dtype=torch.int64, device=device)
        self.static_embedding_output = torch.randn(size=(batch, sequence_length, self.config.hidden_size), dtype=torch.float16, device=device, requires_grad=True)

        # warmup iterations
        s = torch.cuda.Stream(priority=-1)
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup_iteration):
                optimizer.zero_grad()
                loss_sum = self.encoder(
                    self.static_input_ids, self.static_embedding_output, 
                    self.static_attention_mask, self.static_labels, 
                    self.static_labels, self.static_next_sentence_labels)
                valid_labels = torch.sum(torch.ne(self.static_labels, -1))
                loss = loss_sum / valid_labels
                self.scaler.scale(loss).backward()
        
        torch.cuda.current_stream().wait_stream(s)

        self.static_input_ids = torch.ones(size=(batch, sequence_length), dtype=torch.int64, device=device)
        self.static_attention_mask = torch.ones(size=(batch, sequence_length), dtype=torch.float16, device=device)
        self.static_labels = torch.ones(size=(batch, sequence_length), dtype=torch.int64, device=device)
        self.static_next_sentence_labels = torch.ones(size=(batch,), dtype=torch.int64, device=device)
        self.static_embedding_output = torch.randn(size=(batch, sequence_length, self.config.hidden_size), dtype=torch.float16, device=device, requires_grad=True)

        # tracing iterations
        self.encoder_graph = torch.cuda.CUDAGraph()
        optimizer.zero_grad()
        with torch.cuda.graph(self.encoder_graph):
            s = torch.cuda.Stream(priority=-1)
            s.wait_stream(torch.cuda.current_stream())
            loss_sum = self.encoder(
                self.static_input_ids, self.static_embedding_output, 
                self.static_attention_mask, self.static_labels, 
                self.static_labels, self.static_next_sentence_labels)
            valid_labels = torch.sum(torch.ne(self.static_labels, -1))
            loss = loss_sum / valid_labels
            self.scaler.scale(loss).backward()
            torch.cuda.current_stream().wait_stream(s)
    
    def train_with_graph(self, input_ids, token_type_ids, attention_mask, masked_lm_labels, labels, next_sentence_labels):
        self.static_input_ids.copy_(input_ids)
        self.static_attention_mask.copy_(attention_mask)
        self.static_labels.copy_(labels)
        self.static_next_sentence_labels.copy_(next_sentence_labels)
        embedding_output = self.embedding(input_ids, token_type_ids)
        with torch.no_grad():
            self.static_embedding_output.copy_(embedding_output)
        self.encoder_graph.replay()
        embedding_output.backward(self.static_embedding_output.grad)

    
    def checkpoint_activations(self, check):
        self.encoder.checkpoint_activations(check)



################################################################################
# LightSeq2 baseline
################################################################################

