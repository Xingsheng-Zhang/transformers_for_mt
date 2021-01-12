import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import make_model
import numpy as np
from tqdm import tqdm, trange
class NMTModel(nn.Module):
    def __init__(self, config, pad_idx, bos_idx, eos_idx):
        super(NMTModel, self).__init__()
        self.config = config
        self.transformer = make_model(src_vocab=self.config.src_vocab_size,
                                      tgt_vocab=self.config.trg_vocab_size,
                                      num_layers=self.config.num_layers,# 6
                                      d_model=self.config.dim_model,#512
                                      d_ff=self.config.dim_ff,#2048
                                      h=self.config.num_heads,# 8
                                      dropout=self.config.dropout) # 0.1
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.alpha = self.config.alpha
        self.loss_smoothing = self.config.loss_smoothing
        self.eps = self.config.eps
    def forward(self, batch_src_ids, batch_src_mask, batch_trg_ids=None, batch_trg_mask=None, mode='train'):
        """

        :param batch_src_ids: [batch_size, max_seq_len]
        :param batch_src_mask: [batch_size, max_seq_len]
        :param batch_trg_ids: [batch_size, max_seq_len]
        :param batch_trg_mask:[batch_size, max_seq_len-1, max_seq_len-1], 去除最后一个eos
        :param train_flag:
        :param decode_flag:
        :return:
            loss if mode is train
            batch_gen_ids_list : list of batch_idx -> list[gen_seq_len] if mode is decode
        """
        trg_trui_matrix = self.subsequent_mask(self.config.max_seq_len - 1)  # [1, max_seq_len-1, max_seq_len-1]
        trg_trui_mask = torch.from_numpy(trg_trui_matrix)
        trg_trui_mask = trg_trui_mask.to(batch_trg_mask.device)
        batch_trg_mask = batch_trg_mask.unsqueeze(-2) # [batch_size, 1, max_seq_len-1 ]
        batch_trg_mask = batch_trg_mask & trg_trui_mask
        batch_src_mask = batch_src_mask.unsqueeze(-2) # [batch_size, 1, max_seq_len]
        batch_trg_input = batch_trg_ids[:, :-1] # [batch_size, max_seq_len-1], ignore [eos]
        batch_trg_label = batch_trg_ids[:, 1:]  # [batch_size, max_seq_len-1]  去掉开头bos
        if mode == 'train': # for training
            assert batch_trg_ids is not None and batch_trg_mask is not None
            transformer_decode = self.transformer(src=batch_src_ids, tgt=batch_trg_input, src_mask=batch_src_mask,
                                                  tgt_mask=batch_trg_mask)  # [batch_size, max_seq_len-1(从1到final), vocab]
            # generator_output [batch_size, max_seq_len-1, vocab]
            generator_output = self.transformer.generator(transformer_decode)
            loss = self.compute_loss(generator_output, batch_trg_label)
            return generator_output, loss
        elif mode == 'decode': # for translate
            batch_gen_ids_list = []
            # for (src_ids, src_mask) in tqdm(zip(batch_src_ids, batch_src_mask), desc='Batch Decode Iteration', total=batch_src_ids.size(0)):
            for (src_ids, src_mask) in zip(batch_src_ids, batch_src_mask):
                if src_ids.dim() == 1:
                    src_ids = src_ids.unsqueeze(0)
                if src_mask.dim() == 1:
                    src_mask = src_mask.unsqueeze(0)
                # gen_seq_ids list[gen_seq_len](not include bos, eos)
                gen_seq_ids = self._decode_sentence(src_ids, src_mask)
                batch_gen_ids_list.append(gen_seq_ids)
            return batch_gen_ids_list

    # def subsequent_mask(self, size):
    #     """Mask out subsequent positions."""
    #     attn_shape = (1, size, size)
    #     subseq_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    #     subseq_mask = (subseq_mask == 0) + 0
    #     subseq_mask_tensor = torch.from_numpy(subseq_mask)
    #     print('subseq_mask_tensor:{}'.format(subseq_mask_tensor))
    #     return subseq_mask_tensor


    def compute_loss(self, batch_output, batch_label):
        """

        :param batch_output: [batch_size, max_seq_len-1, vocab]
        :param batch_label: [batch_size, max_seq_len-1]
        :return: tensor
        """
        # print('batch_label:{}'.format(batch_label.size()))
        gold = batch_label.contiguous().view(-1)
        # pred already do log_softmax
        pred = batch_output.view(-1, batch_output.size(-1))
        if self.loss_smoothing:
            vocab_size = pred.size(-1)
            # one_hot [batch_size*max_seq_len, vocab]
            one_hot = torch.zeros_like(pred, device=pred.device).scatter(1, gold.view(-1,1), 1)
            one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / (vocab_size - 1)
            loss = - (one_hot * pred).sum(dim=1)
            non_pad_mask = gold.ne(self.pad_idx)
            loss = loss.masked_select(non_pad_mask).sum()
        else:
            # already do log_softmax
            loss = F.nll_loss(pred, gold, ignore_index=self.pad_idx, reduction='sum')
        return loss

    def _decode_sentences(self, batch_src_ids, batch_src_mask):
        """
        func of decode batch sentences
        :param batch_src_ids: [batch_size, max_seq_len]
        :param batch_src_mask: [batch_size, max_seq_len]
        :return:
        """
        batch_encode_output = self.transformer.encode(batch_src_ids, batch_src_mask)
        # TODO for batch sentences decode

    def _decode_sentence(self, src_ids, src_mask):
        """
        func of decode a sentence
        :param src_ids: [1(batch_size), max_seq_len]
        :param src_mask: [1(batch_size), max_seq_len]
        :return:best_pred_ids [gen_seq_len](not include bos and eos)
        """
        # 相当于有topk个size在做decode预测
        # encode_output: [beam_size(k), max_seq_len, hidden_size]
        # gen_seq: [beam_size(k), max_seq_len]
        # scores: [beam_size(k)]
        encode_output, gen_seq, scores = self._init_sentence_decode(src_ids, src_mask)
        best_idx = 0
        sentences_lengths = torch.zeros(self.config.beam_size, dtype=torch.long, device=src_ids.device)

        for step in range(2, self.config.max_seq_len):
            cur_trg_input = gen_seq[:, :step] # [beam_size，STEP_NUM]
            cur_trg_mask = torch.from_numpy(self.subsequent_mask(step))
            cur_trg_mask = cur_trg_mask.to(src_ids.device)
            cur_decode_output = self.transformer.decode(memory=encode_output,
                                       src_mask=src_mask,
                                       tgt=cur_trg_input,
                                       tgt_mask=cur_trg_mask)
            # cur_gen_output [beam_size, step, vocab]
            cur_gen_output = self.transformer.generator(cur_decode_output)
            # gen_seq [beam_size, max_seq_len(: cur_step)]
            gen_seq, scores = self._get_top_k_score_idx(gen_seq, cur_gen_output, scores, step)
            eos_locs = gen_seq == self.eos_idx
            eos_locs = torch.nonzero(eos_locs) # 返回出现eos的位置下标
            # sentences_length 记录已完成句子的程度，如果没完成，则长度为0
            # sentences_length [beam_size]
            for eos_loc in eos_locs:

                sent_idx = eos_loc[0]
                if sentences_lengths[sent_idx] == 0:
                    sentences_lengths[sent_idx] = eos_loc[1] # 记录生成的句子长度
            num_finished_sentences = len([s for s in sentences_lengths if s > 0])
            if num_finished_sentences == self.config.beam_size:
                div = 1 / sentences_lengths.type_as(scores)**self.alpha
                _, best_idx = (scores * div).max(0)
                best_idx = best_idx.item()
                break
        best_pred_ids = gen_seq[best_idx][1:sentences_lengths[best_idx]].tolist()
        # print('model_gen_seq:{}'.format(gen_seq.size()))
        # print('model_best_pred_ids:{}'.format(best_pred_ids))

        return best_pred_ids

    def subsequent_mask(self, size):
        """Mask out subsequent positions."""
        attn_shape = (1, size, size)
        subseq_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return (subseq_mask == 0) + 0

    def _init_batch_sentences_decode(self, batch_src_ids, batch_src_mask):
        """

        :param batch_src_ids:
        :param batch_src_mask:
        :return:
        """
        pass
        # TODO for batch sentences decode

    def _init_sentence_decode(self, src_ids, src_mask):
        """
        :param src_ids: [1(batch_size, max_seq_len)]
        :param src_mask: [1(batch_size, max_seq_len)]
        :return: encode_output: [beam_size(k), max_seq_len, hidden_size]
        :return: gen_seq: [beam_size(k), max_seq_len]
        :return: scores: [beam_size(k)]
        """
        encode_output = self.transformer.encode(src_ids, src_mask)  # [1(batch_size), max_seq_len, hidden_size]
        bos_decode_input = torch.LongTensor([[self.bos_idx]])  # [1(batch_size), 1(bos)]
        bos_decode_input = bos_decode_input.to(src_ids.device)
        bos_decode_mask_matrix = self.subsequent_mask(1)  # [1(batch_size), 1(seq_len), 1(seq_len)]
        bos_decode_mask = torch.from_numpy(bos_decode_mask_matrix)
        bos_decode_mask = bos_decode_mask.to(src_ids.device)
        bos_decode_output = self.transformer.decode(memory=encode_output,
                                                    src_mask=src_mask,
                                                    tgt=bos_decode_input,
                                                    tgt_mask=bos_decode_mask)
        # bos_gen_output [1(batch_size), 1(seq_len), vocab]
        bos_gen_output = self.transformer.generator(bos_decode_output)
        best_k_probs, best_k_idx = bos_gen_output[:, -1, :].topk(self.config.beam_size)
        # scores [beam_size(top_k)]
        # model has already compute log
        scores = best_k_probs.view(self.config.beam_size)
        # [beam_size(top_k), max_seq_len] 存储一个句子的topk的生成序列
        gen_seq = torch.full((self.config.beam_size, self.config.max_seq_len), self.pad_idx, device=src_ids.device).long()
        gen_seq[:, 0] = self.bos_idx
        gen_seq[:, 1] = best_k_idx[0]
        # encode_output [beam_size(top_k), max_seq_len, hidden_size]为每一个k序列准备一个encoder输出
        encode_output = encode_output.repeat(self.config.beam_size, 1, 1)
        return encode_output, gen_seq, scores


    def _get_top_k_score_idx(self, gen_seq, cur_gen_output, prev_scores, cur_step):
        """

        :param gen_seq: [beam_size, max_seq_len]
        :param cur_gen_output: [beam_size, step, vocab]
        :param scores:
        :param step:
        :return:
        """
        beam_size = self.config.beam_size
        # best_k2_probs [beam_size, beam_size] for each k seq find k best candidate
        best_k2_probs, best_k2_idx = cur_gen_output[:, -1, :].topk(beam_size)
        # best_k2_probs [beam_size, beam_size]
        # best_k2_idx [beam_size, beam_size]
        # compute scores
        # cur_scores [beam_size, beam_size]
        # print('size:{}'.format(torch.log(best_k2_probs).view(beam_size, -1).size()))
        # print('prev_scores.view(beam_size, 1):{}'.format(prev_scores.view(beam_size, 1).size()))
        cur_scores = best_k2_probs.view(beam_size, -1) + prev_scores.view(beam_size, 1)

        # Get the best k candidates from k^2 candidates.
        scores, best_k_idx_in_k2 = cur_scores.view(-1).topk(beam_size)

        # 找出最大的k个对应原本是哪些序列
        best_k_row = best_k_idx_in_k2 // beam_size
        best_k_col = best_k_idx_in_k2 % beam_size
        best_k_idx = best_k2_idx[best_k_row, best_k_col]

        gen_seq[:, :cur_step] = gen_seq[best_k_row, :cur_step]
        gen_seq[:, cur_step] = best_k_idx

        return gen_seq, scores