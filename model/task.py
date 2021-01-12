import torch
import numpy as np
import os
import sys
from .preprocess import Preprocess
from .NMTModel import NMTModel
from torch.optim import Adam
from tqdm import tqdm, trange
import collections.abc as container_abcs
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
import math
import torch.nn.parallel as para
import json
from .eval import compute_bleu
# from .scheduler import CosineWithRestarts
from .optim import ScheduledOptim

class NMTTASK(object):
    def __init__(self, config):
        self.config = config
        self.preprocess = Preprocess(max_seq_len=self.config.max_seq_len,
                                     src_vocab_path=self.config.src_vocab_path,
                                     trg_vocab_path=self.config.trg_vocab_path,
                                     bos_token=self.config.bos_token,
                                     eos_token=self.config.eos_token,
                                     pad_token=self.config.pad_token,
                                     unk_token=self.config.unk_token)
        self.config.src_vocab_size = len(self.preprocess.src_vocab.idx2word)
        self.config.trg_vocab_size = len(self.preprocess.trg_vocab.idx2word)
        self.data_dir = self.config.data_dir
        self._init_device() # init self.device
        self.train_dataset = None
        self.dev_dataset = None
        self.test_dataset = None
        self._load_data() # load dataset according to the mode
        """src_vocab 和trg_vocab中这几个一样的"""
        self.unk_idx = self.preprocess.src_vocab.convert_tokens_to_ids(self.config.unk_token)[0]
        self.pad_idx = self.preprocess.src_vocab.convert_tokens_to_ids(self.config.pad_token)[0]
        self.bos_idx = self.preprocess.src_vocab.convert_tokens_to_ids(self.config.bos_token)[0]
        self.eos_idx = self.preprocess.src_vocab.convert_tokens_to_ids(self.config.eos_token)[0]
        self.model = NMTModel(config=self.config,
                              pad_idx=self.pad_idx,
                              bos_idx=self.bos_idx,
                              eos_idx=self.eos_idx)
        self._decorate_model()
        if self.config.use_ScheduledOptim:
            self.optimizer = ScheduledOptim(
                Adam(self.model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-09),
                self.config.init_lr, self.config.dim_model, self.config.n_warmup_steps)
        else:
            self.optimizer = Adam(self.model.parameters(), lr=self.config.lr, betas=(0.9, 0.98), eps=1e-09)
        
        if not os.path.exists(self.config.model_dir):
            os.makedirs(self.config.model_dir, exist_ok=True)

        # self.optimizer = Adam(self.model.parameters(), lr=self.config.lr, betas=(0.9, 0.98), eps=1e-9)
        # if self.config.use_SGDR: # 学习率衰减
        #     self.train_steps = int(len(self.train_dataset) / self.config.train_batch_size)
        #     self.scheduler = CosineWithRestarts(self.optimizer, T_max=self.train_steps)

    # def local
    # @description:
    #   init the model to device (cpu or gpu if is available)
    def _init_device(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config.gpu_ids
        if not self.config.no_cuda and torch.cuda.is_available():
            # print('self.setting.gpu_ids:{}'.format(self.setting.gpu_ids))
            # os.environ["CUDA_VISIBLE_DEVICES"] = self.setting.gpu_ids
            # print('N_GPU:{}'.format(torch.cuda.device_count()))
            gpu_ids = [int(id) for id in self.config.gpu_ids.split(',')]
            self.device = torch.device("cuda")
            self.n_gpu = torch.cuda.device_count()  # the sign for para training
            if self.n_gpu > 1:
                print(
                    'device : {} ; n_gpu : {} ; gpu_ids : {} distributed training '.format(self.device, self.n_gpu,
                                                                                           gpu_ids))
            else:
                print('device : {} ; gpu_id: {} single_gpu training'.format(self.device, gpu_ids))
        else:
            self.device = torch.device('cpu')

    def _load_data(self):
        if self.config.mode == 'pre_test':
            self.load_dataset(load_train=False, load_dev=True, load_test=False)
            self.train_dataset = self.dev_dataset
            self.test_dataset = self.dev_dataset
        elif self.config.mode == 'train':
            self.load_dataset(load_train=True, load_dev=True, load_test=False)
        elif self.config.mode == 'test':
            self.load_dataset(load_train=False, load_dev=False, load_test=True)
        else:
            raise Exception('Unsupport the mode of model')
    def load_dataset(self, load_train=True, load_dev=True, load_test=True):
        if load_train:
            print(' -- Load Train dataset -- ')
            train_src_data_dir = os.path.join(self.data_dir, 'train.zh.tok')
            train_trg_data_dir = os.path.join(self.data_dir, 'train.en.tok')
            self.train_dataset = self.preprocess.load_dataset(train_src_data_dir, train_trg_data_dir)
            print(' -- Finish Train dataset -- ')
        if load_dev:
            print(' -- Load Dev dataset -- ')
            dev_src_data_dir = os.path.join(self.data_dir, 'valid.zh.tok')
            dev_trg_data_dir = os.path.join(self.data_dir, 'valid.en.tok')
            self.dev_dataset = self.preprocess.load_dataset(dev_src_data_dir, dev_trg_data_dir)
            print(' -- Finish Dev dataset -- ')
        if load_test:
            print(' -- Load Test dataset -- ')
            test_src_data_dir = os.path.join(self.data_dir, 'test.zh.tok')
            test_trg_data_dir = os.path.join(self.data_dir, 'test.en.tok')
            self.test_dataset = self.preprocess.load_dataset(test_src_data_dir, test_trg_data_dir)
            print(' -- Finish Test dataset -- ')

    def train(self, resume_base_epoch=None):
        if resume_base_epoch is None:
            if self.config.resume_latest_cpt:
                resume_base_epoch = self.get_lastest_cpt_epoch()
            else:
                resume_base_epoch = 0
        if resume_base_epoch > 0:
            self.resume_cpt_at(resume_base_epoch, resume_model=True, resume_optimizer=True)
        assert self.model is not None and self.train_dataset is not None
        train_data_loader = self.prepare_data_loader(self.train_dataset, self.config.train_batch_size)
        dev_data_loader = self.prepare_data_loader(self.dev_dataset, self.config.eval_batch_size)
        for epoch_idx in trange(resume_base_epoch, int(self.config.num_train_epochs), desc='Epoch'):
            train_loss, train_acc = self.train_epoch(train_data_loader)
            val_loss, val_acc = self.eval_epoch(dev_data_loader)

            print('[ Epoch', epoch_idx, ']')
            print('  - {header:12} loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, '.format(
                header=f"({'Training'})", loss=train_loss, accu=100*train_acc))
            print('  - {header:12} loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, '.format(
                header=f"({'Validation'})", loss=val_loss, accu=100 * val_acc))

            # save checkpoint for each epoch
            self.save_checkpoint(epoch_idx)

    def set_batch_to_device(self, batch):
        # move mini-batch data to the proper device
        if isinstance(batch, torch.Tensor):
            batch = batch.to(self.device)
            return batch

        elif isinstance(batch, dict):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(self.device)
                elif isinstance(value, dict) or isinstance(value, container_abcs.Sequence):
                    batch[key] = self.set_batch_to_device(value)

            return batch
        elif isinstance(batch, container_abcs.Sequence):
            # batch = [
            #     t.to(self.device) if isinstance(t, torch.Tensor) else t for t in batch
            # ]
            new_batch = []
            for value in batch:
                if isinstance(value, torch.Tensor):
                    new_batch.append(value.to(self.device))
                elif isinstance(value, dict) or isinstance(value, container_abcs.Sequence):
                    new_batch.append(self.set_batch_to_device(value))
                else:
                    new_batch.append(value)

            return new_batch
        else:
            raise Exception('Unsupported batch type {}'.format(type(batch)))

    def eval_epoch(self, dev_data_loader):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            total_words = 0
            total_correct_words = 0
            for batch_idx, batch in enumerate(tqdm(dev_data_loader, desc='Validation Iteration')):
                batch = self.set_batch_to_device(batch)
                batch_src_ids, batch_src_mask, batch_trg_ids, batch_trg_mask = batch
                generator_output, loss = self.model(batch_src_ids, batch_src_mask, batch_trg_ids, batch_trg_mask, mode='train')
                n_correct, n_word = self.statistic_token( generator_output, batch_label = batch_trg_ids[:, 1:])
                total_loss += loss.item()
                total_words += n_word
                total_correct_words += n_correct
        loss_per_word = total_loss / total_words
        accuracy = total_correct_words / total_words
        return loss_per_word, accuracy

    def train_epoch(self, train_data_loader):
        self.model.train()
        total_loss = 0
        total_words = 0
        total_correct_words = 0
        for batch_idx, batch in enumerate(tqdm(train_data_loader, desc='Training Iteration')):
            batch = self.set_batch_to_device(batch)
            batch_src_ids, batch_src_mask, batch_trg_ids, batch_trg_mask = batch
            self.optimizer.zero_grad()
            generator_output, loss = self.model(batch_src_ids, batch_src_mask, batch_trg_ids, batch_trg_mask,
                                                mode='train')
            loss.backward()
            loss_scaler = loss.item()
            total_loss += loss_scaler
            # self.optimizer.step()
            if self.config.use_ScheduledOptim:
                self.optimizer.step_and_update_lr()
            else:
                self.optimizer.step()
            # print('lr:{}'.format(self.optimizer.))
            # if self.config.use_SGDR:  # 学习率衰减
            #     self.scheduler.step()
            #     print('lr:{}'.format(self.optimizer.))
            n_correct, n_word = self.statistic_token(generator_output, batch_label=batch_trg_ids[:,
                                                                                   1:])  # [batch_size, max_seq_len-1]  去掉开头bos)
            total_correct_words += n_correct
            total_words += n_word

        train_loss = total_loss / total_words  # training loss per word
        train_acc = total_correct_words / total_words
        return train_loss, train_acc

    def prepare_data_loader(self, dataset, batch_size, rand_flag=True):
        # prepare data loader
        if rand_flag:
            data_sampler = RandomSampler(dataset)
        else:
            data_sampler = SequentialSampler(dataset)

        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                sampler=data_sampler)

        return dataloader

    def statistic_token(self, generator_output, batch_label):
        """
        calculate the accuracy of pred
        :param generator_output: [batch_size, max_seq_len, vocab]
        :param batch_label: [batch_size, max_seq_len]
        :return:
        """
        pred = generator_output.view(-1, generator_output.size(-1)).max(-1)[1]
        gold = batch_label.contiguous().view(-1)
        non_pad_mask = gold.ne(self.pad_idx)
        n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
        n_word = non_pad_mask.sum().item()
        return n_correct, n_word

    # @description:
    #   save the checkpoint of model
    # @params:
    #   cpt_file_name: the file name of checkpoint, is a option para
    #   epoch: the epoch index of model training
    def save_checkpoint(self, epoch_idx):
        cpt_file_name = '{}.cpt.{}'.format(self.config.cpt_file_name, epoch_idx)
        cpt_file_path = os.path.join(self.config.model_dir, cpt_file_name)

        store_dict = {
            'setting': self.config.__dict__,
        }

        if self.model:
            # only save the model_dict, not module.model_dict
            if isinstance(self.model, para.DataParallel) or \
                    isinstance(self.model, para.DistributedDataParallel):
                model_state = self.model.module.state_dict()
            else:
                model_state = self.model.state_dict()
            store_dict['model_state'] = model_state

        if self.optimizer:
            store_dict['optimizer_state'] = self.optimizer.state_dict()

        if epoch_idx:
            store_dict['epoch'] = epoch_idx

        torch.save(store_dict, cpt_file_path)

    def resume_cpt_at(self, epoch, resume_model=True, resume_optimizer=False):
        self.resume_checkpoint(cpt_file_name='{}.cpt.{}'.format(self.config.cpt_file_name, epoch),
                               resume_model=resume_model, resume_optimizer=resume_optimizer)


    # @description:
    #   resume the checkpoint of model， model has been init(already 2 device[gpu or cpu])
    # @params:
    #   cpt_file_path: the path of checkpoint file
    #   cpt_file_name: the name of checkpoint file
    #   resume_model: whether resume the model
    #   resume_optimizer: whether resume the optimizer
    #   strict: strict key value match, if False, can be copy part of model parameters, for Migration learning
    def resume_checkpoint(self, cpt_file_path=None, cpt_file_name=None,
                          resume_model=True, resume_optimizer=False, strict=False):
        # decide cpt_file_path to resume
        if cpt_file_path is None:  # use provided path with highest priority
            if cpt_file_name is None:  # no path and no name will resort to the default cpt name
                cpt_file_name = self.config.cpt_file_name
            cpt_file_path = os.path.join(self.config.model_dir, cpt_file_name)
        elif cpt_file_name is not None:  # error when path and name are both provided
            raise Exception('Confused about path {} or file name {} to resume'.format(
                cpt_file_path, cpt_file_name
            ))

        if not os.path.exists(cpt_file_path):
            raise Exception('Checkpoint does not exist, {}'.format(cpt_file_path))

        # load state_dict
        if self.device == torch.device('cpu'):
            store_dict = torch.load(cpt_file_path, map_location='cpu')
        else:
            store_dict = torch.load(cpt_file_path, map_location="cuda")


        if resume_model:
            if self.model and 'model_state' in store_dict:
                if isinstance(self.model, para.DataParallel) or \
                        isinstance(self.model, para.DistributedDataParallel):
                    self.model.module.load_state_dict(store_dict['model_state'])
                else:
                    self.model.load_state_dict(store_dict['model_state'])
            else:
                raise Exception('Resume model failed, dict.keys = {}'.format(store_dict.keys()))

        if resume_optimizer:
            if self.optimizer and 'optimizer_state' in store_dict:
                self.optimizer.load_state_dict(store_dict['optimizer_state'])
            else:
                raise Exception('Resume optimizer failed, dict.keys = {}'.format(store_dict.keys()))

    # @description:
    #   get the lastest epoch index according to startwith('{}.cpt') at setting.model_dir
    # @Returns:
    #   lastest_epoch: the lastest epoch index of model that has been saved at the setting.model_dir
    def get_lastest_cpt_epoch(self):
        previous_epochs_in_list = []
        lastest_epoch = 0
        print('self.setting.model_dir:{}'.format(self.config.model_dir))
        for fn in os.listdir(self.config.model_dir):
            print('fn:{}'.format(fn))
            if fn.startswith('{}.cpt'.format(self.config.cpt_file_name)):
                try:
                    epoch = int(fn.split('.')[-1])
                    previous_epochs_in_list.append(epoch)
                    print('epoch:{}'.format(epoch))
                except Exception as e:
                    continue
        previous_epochs_in_list.sort()
        if len(previous_epochs_in_list) > 0:
            lastest_epoch = previous_epochs_in_list[-1]
        return lastest_epoch


    # @description:
    #   init model to device(cpu or gpu)
    # @params:
    #   parallel_decorate: the sign of parallel, include single compute multi gpu,and multi compute multi gpu
    def _decorate_model(self, parallel_decorate=True):
        self.model.to(self.device)
        if self.n_gpu > 1:
            self.model = para.DataParallel(self.model)
    def test(self, test_model_epoch_idx=None):
        if test_model_epoch_idx is None:
            test_model_epoch_idx = self.get_lastest_cpt_epoch()
        assert test_model_epoch_idx >= 0
        self.resume_cpt_at(test_model_epoch_idx, resume_model=True, resume_optimizer=False)

        assert self.test_dataset is not None and self.model is not None
        self.model.eval()
        test_data_loader = self.prepare_data_loader(self.test_dataset, self.config.test_batch_size, rand_flag=True)
        total_gen_seq_str_list = []
        total_label_seq_str_list = []
        total_correct_words = 0
        total_words = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_data_loader, desc='Testing Iteration')):
                batch = self.set_batch_to_device(batch)
                batch_src_ids, batch_src_mask, batch_trg_ids, batch_trg_mask = batch
                # batch_gen_ids_list batch_idx -> [gen_seq_len]
                batch_gen_ids_list = self.model(batch_src_ids, batch_src_mask, batch_trg_ids, batch_trg_mask, mode='decode')
                n_correct, n_word = self.cal_acc_for_decode_mode(batch_gen_ids_list, batch_trg_ids)
                total_correct_words += n_correct
                total_words += n_word
                batch_gen_seq_str_list = []
                batch_label_seq_str_list = []
                # log_count = 0
                for (src_ids, gen_seq_ids, trg_seq_ids, trg_mask) in zip(batch_src_ids, batch_gen_ids_list, batch_trg_ids, batch_trg_mask):
                    gen_seq_str = self.preprocess.decode_sequence(gen_seq_ids)
                    batch_gen_seq_str_list.append(gen_seq_str)
                    label_seq_str = self.generate_label_seq(trg_seq_ids)
                    batch_label_seq_str_list.append(label_seq_str)
                    src_seq_str = self.generate_src_seq(src_ids)
                #     if log_count < 5:
                #         print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
                #         print('SRC_sent:{}'.format(src_seq_str))
                #         print('TRG_sent:{}'.format(label_seq_str))
                #         print('NMT:{}'.format(gen_seq_str))
                #         print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
                #         log_count += 1
                # local_bleu_score = compute_bleu([batch_gen_seq_str_list], batch_label_seq_str_list)
                # print('local_bleu_score:{}'.format(local_bleu_score))
                # print('  - {header:12}   batch_idx:{batch_idx: 2}, accu: {accu: 3.3f} %, bleu: {bleu:8.5f} , '.format(
                #     header=f"({'Testing'})", batch_idx=batch_idx, accu=100 * local_acc, bleu=local_bleu_score))
                total_gen_seq_str_list.extend(batch_gen_seq_str_list)
                total_label_seq_str_list.extend(batch_label_seq_str_list)
        bleu_score = compute_bleu([total_gen_seq_str_list], total_label_seq_str_list)

        accuracy = total_correct_words / total_words
        print('[ Epoch', test_model_epoch_idx, ']')
        print('  - {header:12} accu: {accu: 3.3f} %, bleu: {bleu:8.5f} , '.format(
            header=f"({'Testing'})", accu=100 * accuracy, bleu=bleu_score))
        # print('  - {header:12}   accu: {accu: 3.3f} %, bleu: {bleu:8.5f} , '.format(
        #     header=f"({'Testing'})", accu=100 * accuracy, bleu=bleu_score))

        return bleu_score

    def generate_src_seq(self, src_seq_ids):
        src_seq_ids = src_seq_ids.tolist()
        # print('trg_seq_ids:{}'.format(trg_seq_ids))
        if self.pad_idx in src_seq_ids:
            pad_loc = src_seq_ids.index(self.pad_idx)
            real_src_seq_ids = src_seq_ids[:pad_loc]
        else:
            real_src_seq_ids = src_seq_ids
        # assert self.bos_idx not in real_src_seq_ids
        # print('real_trg_seq_ids:{}'.format(real_trg_seq_ids))
        seq_str = self.preprocess.decode_sequence(real_src_seq_ids, lang='src')
        return seq_str

    def generate_label_seq(self, trg_seq_ids):
        """

        :param trg_seq_ids: [max_seq_len]
        :param trg_mask: [max_seq_len]
        :return:
        """

        trg_seq_ids = trg_seq_ids.tolist()
        # print('trg_seq_ids:{}'.format(trg_seq_ids))
        eos_loc = trg_seq_ids.index(self.eos_idx)
        real_trg_seq_ids = trg_seq_ids[1:eos_loc]
        assert self.bos_idx not in real_trg_seq_ids
        # print('real_trg_seq_ids:{}'.format(real_trg_seq_ids))
        if real_trg_seq_ids[-1] == self.eos_idx:
            real_trg_seq_ids = real_trg_seq_ids[:-1]
        # print('real_trg_seq_ids:{}'.format(real_trg_seq_ids))
        seq_str = self.preprocess.decode_sequence(real_trg_seq_ids)
        return seq_str

    def cal_acc_for_decode_mode(self, batch_gen_ids_list, batch_trg_ids):
        # batch_trg_ids [batch_size, max_seq_len]
        # print('config:max_seq_len:{}'.format(self.config.max_seq_len))
        max_seq_len = batch_trg_ids.size(-1)
        # print('max_seq_len:{}'.format(max_seq_len))
        batch_pad_gen_seq_list = []
        for gen_ids in batch_gen_ids_list:
            pad_gen_ids = list(gen_ids)
            pad_gen_ids.insert(0, self.bos_idx) # add eos
            pad_gen_ids = pad_gen_ids[:self.config.max_seq_len-1]
            pad_gen_ids.append(self.eos_idx) # add eos
            while len(pad_gen_ids) < max_seq_len:
                pad_gen_ids.append(self.pad_idx)
            assert len(pad_gen_ids) == self.config.max_seq_len
            batch_pad_gen_seq_list.append(pad_gen_ids)
        batch_pad_gen_seq = torch.tensor(batch_pad_gen_seq_list) # [batch_size, max_seq_len]
        batch_pad_gen_seq = batch_pad_gen_seq.to(batch_trg_ids.device) # [batch_size, max_seq_len]
        pred = batch_pad_gen_seq.view(-1)
        gold = batch_trg_ids.view(-1)
        non_pad_mask = gold.ne(self.pad_idx)
        n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
        n_word = non_pad_mask.sum().item()
        return n_correct, n_word




