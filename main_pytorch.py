import os
import sys
# sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import h5py
import math
from sklearn import metrics
import time
import yaml
import logging
import matplotlib.pyplot as plt
import pickle
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.data_generator import DataGenerator, InferenceDataGenerator
from utils.utilities import (get_filename, create_logging, create_folder,
                             prec_recall_fvalue, search_meta_by_mixture_name,
                             get_sed_from_meta, ideal_binary_mask)
from utils.features import LogMelExtractor
# from models_pytorch import move_data_to_gpu, VggishGMP, VggishGAP, VggishGWRP, VggishRGWRP, Unet, RUnet
from models_pytorch import move_data_to_gpu, Vggish, Unet, HornNet
from utils import config as config

from torch.utils.tensorboard import SummaryWriter

batch_size = 24


def get_model(model_type):
    if model_type == 'gwrp':
        return Vggish, 'GWRP'

    elif model_type == 'rgwrp':
        return Vggish, 'RGWRP'

    elif model_type == 'tp':
        return Vggish, 'TP'

    elif model_type == 'unet':
        return Unet, 'GWRP'

    elif model_type == 'runet':
        return Unet, 'RGWRP'

    elif model_type == 'tunet':
        return Unet, 'TP'

    elif model_type == 'hornnet':
        return HornNet, 'GWRP'
    
    elif model_type == 'rhornnet':
        return HornNet, 'RGWRP'

    elif model_type == 'thornnet':
        return HornNet, 'TP'

    elif model_type == 'ahornnet':
        return HornNet, 'AM'

    elif model_type == 'mhornnet':
        return HornNet, 'M'

    else:
        raise Exception('Incorrect model type!')


def evaluate(model, generator, data_type, max_iteration, cuda):
    """Evaluate

    Args:
      model: object.
      generator: object.
      data_type: 'train' | 'validate'.
      max_iteration: int, maximum iteration for validation
      cuda: bool.

    Returns:
      accuracy: float
    """

    # Generate function
    generate_func = generator.generate_validate(data_type=data_type,
                                                shuffle=True,
                                                max_iteration=max_iteration)

    # Forward
    dict = forward(model=model,
                   generate_func=generate_func,
                   return_target=True,
                   return_bottleneck=False,
                   cuda=cuda)

    outputs = dict['output']  # (audios_num, classes_num)
    targets = dict['target']  # (audios_num, classes_num)

    # Evaluate
    (precision, recall, f1_score) = prec_recall_fvalue(
        targets, outputs, thres=0.5, average='macro')

    try:
        auc = metrics.roc_auc_score(targets, outputs, average='macro')
    except:
        auc = 0.

    try:
        map = metrics.average_precision_score(targets, outputs, average='macro')
    except:
        map = 0.

    classes_num = outputs.shape[-1]

    loss = float(F.binary_cross_entropy(
        torch.Tensor(outputs), torch.Tensor(targets)).numpy())

    return loss, f1_score, auc, map, precision, recall


def forward(model, generate_func, return_target, return_bottleneck, cuda):
    """Forward data to a model.

    Args:
      generate_func: generate function
      return_target: bool
      return_bottleneck: bool
      cuda: bool

    Returns:
      dict, keys: 'audio_name', 'output'; optional keys: 'target'
    """

    outputs = []
    audio_names = []

    if return_target:
        targets = []

    if return_bottleneck:
        bottlenecks = []

    # Evaluate on mini-batch
    for data in generate_func:

        if return_target:
            (batch_x, batch_y, batch_audio_names) = data

        else:
            (batch_x, batch_audio_names) = data

        batch_x = move_data_to_gpu(batch_x, cuda)

        # Predict
        model.eval()

        if return_bottleneck:
            (batch_output, batch_bottleneck) = model(
                batch_x, return_bottleneck=True)

            bottlenecks.append(batch_bottleneck.data.cpu().numpy())

        else:
            batch_output = model(batch_x, return_bottleneck=False)

        # Append data
        outputs.append(batch_output.data.cpu().numpy())
        audio_names.append(batch_audio_names)

        if return_target:
            targets.append(batch_y)

    dict = {}

    outputs = np.concatenate(outputs, axis=0)
    dict['output'] = outputs

    audio_names = np.concatenate(audio_names, axis=0)
    dict['audio_name'] = audio_names

    if return_target:
        targets = np.concatenate(targets, axis=0)
        dict['target'] = targets

    if return_bottleneck:
        bottlenecks = np.concatenate(bottlenecks, axis=0)
        dict['bottleneck'] = bottlenecks

    return dict


def train(args):
    # Arugments & parameters
    workspace = args.workspace
    model_type = args.model_type
    filename = args.filename
    holdout_fold = args.holdout_fold
    scene_type = args.scene_type
    snr = args.snr
    cuda = args.cuda
    mixture_type = args.mixture_type

    labels = config.labels
    classes_num = len(labels)
    seq_len = config.seq_len
    mel_bins = config.mel_bins
    max_iteration = 10  # To speed up validation, set a maximum iteration

    # Paths
    hdf5_path = os.path.join(workspace, 'features', 'logmel',
                             'scene_type={},snr={},mixture_type={}'.format(scene_type, snr, mixture_type), 'development.h5')

    models_dir = os.path.join(workspace, 'models', filename,
                              'model_type={}'.format(model_type),
                              'scene_type={},snr={},mixture_type={}'.format(scene_type, snr, mixture_type),
                              'holdout_fold{}'.format(holdout_fold))

    create_folder(models_dir)

    # Model
    Model, pool_type = get_model(model_type)

    # model = Model(classes_num, seq_len, mel_bins, cuda)
    writer = torch.utils.tensorboard.SummaryWriter(comment=
                                                   'model_type={},scene_type={},snr={},mixture_type={},holdout_fold{}'
                                                   .format(model_type,scene_type, snr, mixture_type, holdout_fold))
    model = Model(classes_num, pool_type, writer)

    if cuda:
        model.cuda()
        # model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    # Data generator
    generator = DataGenerator(hdf5_path=hdf5_path,
                              batch_size=batch_size,
                              holdout_fold=holdout_fold)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999),
                           eps=1e-08, weight_decay=0)
    # optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay=0)

    train_bgn_time = time.time()

    # Train on mini batches
    total_time = 0.
    for iteration, (batch_x, batch_y) in enumerate(generator.generate_train()):

        # Evaluate
        if iteration % 100 == 0:
            train_fin_time = time.time()

            (tr_loss, tr_f1_score, tr_auc, tr_map, tr_prec, tr_rec) = evaluate(
                model=model, generator=generator, data_type='train',
                max_iteration=max_iteration, cuda=cuda)

            logging.info('tr_loss: {:.3f}, tr_f1_score: {:.3f}, '
                         'tr_prec: {:.3f}, tr_rec: {:.3f}'
                         ''.format(tr_loss, tr_f1_score, tr_prec, tr_rec))

            (va_loss, va_f1_score, va_auc, va_map, va_prec, va_rec) = evaluate(
                model=model, generator=generator, data_type='validate',
                max_iteration=max_iteration, cuda=cuda)

            logging.info('va_loss: {:.3f}, va_f1_score: {:.3f}, '
                         'va_prec: {:.3f}, va_rec: {:.3f}'
                         ''.format(va_loss, va_f1_score, va_prec, va_rec))

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time
            total_time += train_time

            logging.info(
                'iteration: {}, train time: {:.3f} s, validate time: {:.3f} s'
                ''.format(iteration, train_time, validate_time))

            logging.info('------------------------------------')

            train_bgn_time = time.time()

            writer.add_scalar('loss', va_loss, global_step=iteration)
            writer.add_scalar('metric/f1', va_f1_score, global_step=iteration)
            writer.add_scalar('metric/prec', va_prec, global_step=iteration)
            writer.add_scalar('metric/rec', va_rec, global_step=iteration)
            # writer.add_scalar('alpha', model.pooling.alpha, global_step=iteration)

        # Save model
        if iteration % 1000 == 0 and iteration > 0:
            save_out_dict = {'iteration': iteration,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()
                             }

            save_out_path = os.path.join(
                models_dir, 'md_{}_iters.tar'.format(iteration))

            torch.save(save_out_dict, save_out_path)
            logging.info('Model saved to {}'.format(save_out_path))

        # Reduce learning rate
        # if iteration % 250 == 0 > 0:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] *= 0.9


        # Train
        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_y = move_data_to_gpu(batch_y, cuda)

        model.train()
        batch_output = model(batch_x)

        loss = F.binary_cross_entropy(batch_output, batch_y)
        writer.add_scalar('loss', loss, global_step=iteration)

        # alphas = torch.split(model.pooling.alpha, 1)
        # for i in range(41):
        #     writer.add_scalar('alpha/%d' % (i+1), alphas[i], global_step=iteration)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Stop learning
        if iteration == 10000:
            break
    f = open('result/' + mixture_type + '_model_time_log', 'a')
    logging.info('avg train time: {:.3f} s / 100steps'.format(total_time / 100))
    f.write(model_type + '\t' + str(holdout_fold) + '\t' + str(total_time) + '\n')
    if holdout_fold==4:
        f.write('\n')
    f.close()


def inference(args):
    # Arugments & parameters
    workspace = args.workspace
    model_type = args.model_type
    holdout_fold = args.holdout_fold
    scene_type = args.scene_type
    snr = args.snr
    iteration = args.iteration
    filename = args.filename
    cuda = args.cuda
    mixture_type = args.mixture_type

    labels = config.labels
    classes_num = len(labels)
    sample_rate = config.sample_rate
    window_size = config.window_size
    overlap = config.overlap
    hop_size = window_size - overlap
    mel_bins = config.mel_bins
    seq_len = config.seq_len
    ix_to_lb = config.ix_to_lb

    thres = 0.1

    # Paths
    hdf5_path = os.path.join(workspace, 'features', 'logmel',
                             'scene_type={},snr={},mixture_type={}'.format(scene_type, snr, mixture_type), 'development.h5')

    model_path = os.path.join(workspace, 'models', filename,
                              'model_type={}'.format(model_type),
                              'scene_type={},snr={},mixture_type={}'.format(scene_type, snr, mixture_type),
                              'holdout_fold{}'.format(holdout_fold),
                              'md_{}_iters.tar'.format(iteration))
    create_folder(os.path.dirname(model_path))

    yaml_path = os.path.join(workspace, 'yaml_files', mixture_type, 'mixture.yaml')

    out_stat_path = os.path.join(workspace, 'stats', filename,
                                 'model_type={}'.format(model_type),
                                 'scene_type={},snr={},mixture_type={}'.format(scene_type, snr, mixture_type),
                                 'holdout_fold{}'.format(holdout_fold),
                                 'stat.p')
    create_folder(os.path.dirname(out_stat_path))

    pred_prob_path = os.path.join(workspace, 'pred_probs', filename,
                                  'model_type={}'.format(model_type),
                                  'scene_type={},snr={},mixture_type={}'.format(scene_type, snr, mixture_type),
                                  'holdout_fold{}'.format(holdout_fold),
                                  'pred_prob.p')
    create_folder(os.path.dirname(pred_prob_path))

    # Load yaml file
    load_yaml_time = time.time()

    with open(yaml_path, 'r') as f:
        meta = yaml.load(f)

    logging.info('Load yaml file time: {:.3f} s'.format(time.time() - load_yaml_time))

    feature_extractor = LogMelExtractor(sample_rate=sample_rate,
                                        window_size=window_size,
                                        overlap=overlap,
                                        mel_bins=mel_bins)

    # Load model
    Model, pool_type = get_model(model_type)

    # model = Model(classes_num, seq_len, mel_bins, cuda)
    model = Model(classes_num, pool_type)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    if cuda:
        model.cuda()

    # Data generator
    generator = InferenceDataGenerator(hdf5_path=hdf5_path,
                                       batch_size=batch_size,
                                       holdout_fold=holdout_fold)

    generate_func = generator.generate_validate(data_type='validate',
                                                shuffle=False,
                                                max_iteration=None)

    audio_names = []
    at_outputs = []
    at_targets = []

    sed_outputs = []
    sed_targets = []

    ss_outputs = []
    ss_targets = []

    # Evaluate on mini-batch
    for iteration, data in enumerate(generate_func):

        print('iteration: {}'.format(iteration))

        (batch_x, batch_y, batch_audio_names) = data

        batch_x = move_data_to_gpu(batch_x, cuda)

        # Predict
        model.eval()

        (batch_output, batch_bottleneck) = model(batch_x,
                                                 return_bottleneck=True)

        batch_output = batch_output.data.cpu().numpy()
        '''(batch_size, classes_num)'''

        batch_bottleneck = batch_bottleneck.data.cpu().numpy()
        '''(batch_size, classes_num, seq_len, mel_bins)'''

        audio_names.append(batch_audio_names)
        at_outputs.append(batch_output)
        at_targets.append(batch_y)

        batch_pred_sed = np.mean(batch_bottleneck, axis=-1)
        batch_pred_sed = np.transpose(batch_pred_sed, (0, 2, 1))
        '''(batch_size, seq_len, classes_num)'''

        for n in range(len(batch_audio_names)):
            gt_meta = search_meta_by_mixture_name(meta, batch_audio_names[n])
            gt_events = gt_meta['events']
            gt_sed = get_sed_from_meta(gt_events)
            '''(seq_len, classes_num)'''

            pred_classes = np.where(batch_output[n] > thres)[0]
            pred_sed = np.zeros((seq_len, classes_num))
            pred_sed[:, pred_classes] = batch_pred_sed[n][:, pred_classes]
            '''(seq_len, classes_num)'''

            sed_outputs.append(pred_sed)
            sed_targets.append(gt_sed)

            (events_stft, scene_stft, mixture_stft) = \
                generator.get_events_scene_mixture_stft(batch_audio_names[n])
            '''(seq_len, fft_bins)'''

            events_stft = np.dot(events_stft, feature_extractor.melW)
            scene_stft = np.dot(scene_stft, feature_extractor.melW)
            '''(seq_len, mel_bins)'''

            gt_mask = ideal_binary_mask(events_stft, scene_stft)
            '''(seq_len, mel_bins)'''

            gt_masks = gt_mask[:, :, None] * gt_sed[:, None, :]
            gt_masks = gt_masks.astype(np.float32)
            '''(seq_len, fft_size, classes_num)'''

            pred_masks = batch_bottleneck[n].transpose(1, 2, 0)
            '''(seq_len, fft_size, classes_num)'''

            ss_outputs.append(pred_masks)
            ss_targets.append(gt_masks)

        # if iteration == 3: break

    audio_names = np.concatenate(audio_names, axis=0)

    at_outputs = np.concatenate(at_outputs, axis=0)
    at_targets = np.concatenate(at_targets, axis=0)
    '''(audio_clips,)'''

    sed_outputs = np.array(sed_outputs)
    sed_targets = np.array(sed_targets)
    '''(audio_clips, seq_len, classes_num)'''

    ss_outputs = np.array(ss_outputs)
    ss_targets = np.array(ss_targets)
    '''(audio_clips, seq_len, mel_bins, classes_num)'''

    pred_prob = {'audio_name': audio_names,
                 'at_output': at_outputs, 'at_target': at_targets,
                 'sed_output': sed_outputs, 'sed_target': sed_targets}

    pickle.dump(pred_prob, open(pred_prob_path, 'wb'))
    logging.info('Saved stat to {}'.format(pred_prob_path))

    # Evaluate audio tagging
    at_time = time.time()

    # (at_precision, at_recall, at_f1_score) = prec_recall_fvalue(at_targets, at_outputs, thres, None)
    # at_auc = metrics.roc_auc_score(at_targets, at_outputs, average=None)
    # at_ap = metrics.average_precision_score(at_targets, at_outputs, average=None)

    (log_at_precision, log_at_recall, log_at_f1_score) = prec_recall_fvalue(at_targets, at_outputs, thres,
                                                                            average='macro')
    log_at_auc = metrics.roc_auc_score(at_targets, at_outputs, average='macro')
    log_at_ap = metrics.average_precision_score(at_targets, at_outputs, average='macro')

    logging.info('Audio tagging time: {:.3f} s'.format(time.time() - at_time))
    logging.info('acu: %.4f, ap: %.4f' % (log_at_auc, log_at_ap))
    logging.info('precision: %.4f, recall: %.4f, f1: %.4f' %
                 (log_at_precision, log_at_recall, log_at_f1_score))

    # Evaluate SED
    sed_time = time.time()

    # (sed_precision, sed_recall, sed_f1_score) = prec_recall_fvalue(
    #     sed_targets.reshape((sed_targets.shape[0] * sed_targets.shape[1], sed_targets.shape[2])),
    #     sed_outputs.reshape((sed_outputs.shape[0] * sed_outputs.shape[1], sed_outputs.shape[2])),
    #     thres=thres, average=None)
    #
    # sed_auc = metrics.roc_auc_score(
    #     sed_targets.reshape((sed_targets.shape[0] * sed_targets.shape[1], sed_targets.shape[2])),
    #     sed_outputs.reshape((sed_outputs.shape[0] * sed_outputs.shape[1], sed_outputs.shape[2])),
    #     average=None)
    #
    # sed_ap = metrics.average_precision_score(
    #     sed_targets.reshape((sed_targets.shape[0] * sed_targets.shape[1], sed_targets.shape[2])),
    #     sed_outputs.reshape((sed_outputs.shape[0] * sed_outputs.shape[1], sed_outputs.shape[2])),
    #     average=None)

    (log_sed_precision, log_sed_recall, log_sed_f1_score) = prec_recall_fvalue(
        sed_targets.reshape((sed_targets.shape[0] * sed_targets.shape[1], sed_targets.shape[2])),
        sed_outputs.reshape((sed_outputs.shape[0] * sed_outputs.shape[1], sed_outputs.shape[2])),
        thres=thres, average='macro')
    log_sed_auc = metrics.roc_auc_score(
        sed_targets.reshape((sed_targets.shape[0] * sed_targets.shape[1], sed_targets.shape[2])),
        sed_outputs.reshape((sed_outputs.shape[0] * sed_outputs.shape[1], sed_outputs.shape[2])),
        average='macro')
    log_sed_ap = metrics.average_precision_score(
        sed_targets.reshape((sed_targets.shape[0] * sed_targets.shape[1], sed_targets.shape[2])),
        sed_outputs.reshape((sed_outputs.shape[0] * sed_outputs.shape[1], sed_outputs.shape[2])),
        average='macro')

    logging.info('SED time: {:.3f} s'.format(time.time() - sed_time))
    logging.info('acu: %.4f, ap: %.4f' % (log_sed_auc, log_sed_ap))
    logging.info('precision: %.4f, recall: %.4f, f1: %.4f' %
                 (log_sed_precision, log_sed_recall, log_sed_f1_score))

    '''
    # Evaluate source separation
    ss_time = time.time()
    # ss_auc = metrics.roc_auc_score(
    #     ss_targets.reshape((ss_targets.shape[0] * ss_targets.shape[1] * ss_targets.shape[2], ss_targets.shape[3])),
    #     ss_outputs.reshape((ss_outputs.shape[0] * ss_outputs.shape[1] * ss_outputs.shape[2], ss_outputs.shape[3])),
    #     average=None)

    log_ss_auc = metrics.roc_auc_score(
        ss_targets.reshape((ss_targets.shape[0] * ss_targets.shape[1] * ss_targets.shape[2], ss_targets.shape[3])),
        ss_outputs.reshape((ss_outputs.shape[0] * ss_outputs.shape[1] * ss_outputs.shape[2], ss_outputs.shape[3])),
        average='macro')
    logging.info('SS AUC time: {:.3f} s'.format(time.time() - ss_time))

    ss_time = time.time()
    # ss_ap = metrics.average_precision_score(
    #     ss_targets.reshape((ss_targets.shape[0] * ss_targets.shape[1] * ss_targets.shape[2], ss_targets.shape[3])),
    #     ss_outputs.reshape((ss_outputs.shape[0] * ss_outputs.shape[1] * ss_outputs.shape[2], ss_outputs.shape[3])),
    #     average=None)

    log_ss_ap = metrics.average_precision_score(
        ss_targets.reshape((ss_targets.shape[0] * ss_targets.shape[1] * ss_targets.shape[2], ss_targets.shape[3])),
        ss_outputs.reshape((ss_outputs.shape[0] * ss_outputs.shape[1] * ss_outputs.shape[2], ss_outputs.shape[3])),
        average='macro')
    logging.info('SS AP time: {:.3f} s'.format(time.time() - ss_time))
    logging.info('acu: %.4f, ap: %.4f' % (log_ss_auc, log_ss_ap))

    ss_time = time.time()
    # (ss_precision, ss_recall, ss_f1_score) = prec_recall_fvalue(
    #     ss_targets.reshape((ss_targets.shape[0] * ss_targets.shape[1] * ss_targets.shape[2], ss_targets.shape[3])),
    #     ss_outputs.reshape((ss_outputs.shape[0] * ss_outputs.shape[1] * ss_outputs.shape[2], ss_outputs.shape[3])),
    #     thres=thres, average=None)

    (log_ss_precision, log_ss_recall, log_ss_f1_score) = prec_recall_fvalue(
        ss_targets.reshape((ss_targets.shape[0] * ss_targets.shape[1] * ss_targets.shape[2], ss_targets.shape[3])),
        ss_outputs.reshape((ss_outputs.shape[0] * ss_outputs.shape[1] * ss_outputs.shape[2], ss_outputs.shape[3])),
        thres=thres, average='macro')

    logging.info('SS fvalue time: {:.3f} s'.format(time.time() - ss_time))
    logging.info('precision: %.4f, recall: %.4f, f1: %.4f' %
                 (log_ss_precision, log_ss_recall, log_ss_f1_score))
    '''

    # Write stats
    stat = {'at_precision': log_at_precision, 'at_recall': log_at_recall, 'at_f1_score': log_at_f1_score, 'at_auc': log_at_auc, 'at_ap': log_at_ap,
            'sed_precision': log_sed_precision, 'sed_recall': log_sed_recall, 'sed_f1_score': log_sed_f1_score, 'sed_auc': log_sed_auc, 'sed_ap': log_sed_ap
            }

    pickle.dump(stat, open(out_stat_path, 'wb'))
    logging.info('Saved stat to {}'.format(out_stat_path))
    f = open('result/' + mixture_type + '_model_log', 'a')
    f.write(model_type + '\t' + str(holdout_fold)
            + '\t' + str(log_at_auc) + '\t' + str(log_at_ap) + '\t' + str(log_at_precision) + '\t' + str(
        log_at_recall) + '\t' + str(log_at_f1_score)
            + '\t' + str(log_sed_auc) + '\t' + str(log_sed_ap) + '\t' + str(log_sed_precision) + '\t' + str(
        log_sed_recall) + '\t' + str(log_sed_f1_score)+ '\t'  + str(model.pooling.alpha)
            # +'\t'+str(log_ss_auc)+'\t'+str(log_ss_ap)+'\t'+str(log_ss_precision)+'\t'+str(log_ss_recall)+'\t'+str(log_ss_f1_score)
            + '\n')
    f.close()


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)

        parser = argparse.ArgumentParser(description='Example of parser. ')
        subparsers = parser.add_subparsers(dest='mode')

        parser_train = subparsers.add_parser('train')
        parser_train.add_argument('--workspace', type=str, required=True)
        parser_train.add_argument('--model_type', type=str,
                                  choices=['gwrp', 'rgwrp', 'tp',
                                           'unet', 'runet', 'tunet',
                                           'hornnet', 'rhornnet', 'thornnet', 'ahornnet', 'mhornnet'], required=True)
        parser_train.add_argument('--scene_type', type=str, required=True)
        parser_train.add_argument('--snr', type=int, required=True)
        parser_train.add_argument('--mixture_type', type=str,
                                  choices=['original', 'rand_init', 'no_clip', 'mix'], required=True)
        parser_train.add_argument('--holdout_fold', type=int)
        parser_train.add_argument('--cuda', action='store_true', default=False)

        parser_inference = subparsers.add_parser('inference')
        parser_inference.add_argument('--workspace', type=str, required=True)
        parser_inference.add_argument('--model_type', type=str,
                                  choices=['gwrp', 'rgwrp', 'tp',
                                           'unet', 'runet', 'tunet',
                                           'hornnet', 'rhornnet', 'thornnet', 'ahornnet', 'mhornnet'], required=True)
        parser_inference.add_argument('--scene_type', type=str, required=True)
        parser_inference.add_argument('--snr', type=int, required=True)
        parser_inference.add_argument('--mixture_type', type=str,
                                      choices=['original', 'rand_init', 'no_clip', 'mix'], required=True)
        parser_inference.add_argument('--holdout_fold', type=int)
        parser_inference.add_argument('--iteration', type=int, required=True)
        parser_inference.add_argument('--cuda', action='store_true', default=False)

        args = parser.parse_args()
        print(args.workspace)

        args.filename = get_filename(__file__)

        # Create log
        logs_dir = os.path.join(args.workspace, 'logs', args.filename)
        create_logging(logs_dir, filemode='w')
        logging.info(args)

        if args.mode == 'train':
            train(args)

        elif args.mode == 'inference':
            inference(args)

        else:
            raise Exception('Error argument!')