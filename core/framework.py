# coding=UTF-8
"""
@Description:
@Author: Lingyong Yan
@Date: 2019-07-09 04:38:30
@LastEditTime: 2019-08-30 00:30:23
@LastEditors: Lingyong Yan
@Python release: 3.7
@Notes:
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity as torch_sim
# from torch.optim import lr_scheduler

from .util import class_mask, group_mask, neighbor_mask, predict_confidence
from .util import heuristic_pooling, cosine_sim
from .util import neighbor_adj, bidirection_adj_eff


def preprocess(device, *args):
    results = []
    for arg in args:
        if arg is not None:
            if isinstance(arg, tuple) or isinstance(arg, list):
                arg = [p_arg.to(device) for p_arg in arg]
                results.append(arg)
            else:
                results.append(arg.to(device))
        else:
            results.append(arg)
    return results


def init_teacher_data(target_pair, idx_train, device):
    target, target_t = preprocess(device, *target_pair)
    temp = torch.zeros(idx_train.size(0), target_t.size(1),
                       device=device).type_as(target_t)
    temp.scatter_(1, torch.unsqueeze(target[idx_train], 1), 1.0)
    target_t[idx_train] = temp
    return target_t


def update_teacher_data(opt, model, target_pair, inputs, device):
    idx_train, es, ps, ep_adj, pe_adj, nbs = preprocess(device, *inputs)
    target, target_t = preprocess(device, *target_pair)
    with torch.no_grad():
        model.eval()
        outputs, selects, steps, _ = model(idx_train, es, ps, ep_adj, pe_adj, nbs)
        outputs = torch.cat(outputs)
        preds = torch.full((outputs.size(0),), -1, device=outputs.device, dtype=torch.long)

        step_score = torch.ones(outputs.size(0))
        step_score = step_score.to(device)
        start = 0
        for i, (ss, step_list) in enumerate(zip(selects, steps)):
            end = start + len(ss)
            score = np.exp(- (i + 1) / 10)
            step_score[start:end] = score
            i_start = start
            for j, step in enumerate(step_list):
                i_end = i_start + step
                preds[i_start:i_end] = j
                i_start = i_end
            start = end

        selects = torch.cat(selects)
        confidence = torch.zeros(target_t.size(0), device=outputs.device)
        confidence[selects] = predict_confidence(outputs)
        confidence[selects] = confidence[selects] * step_score
        if opt['t_draw'] == 'exp':
            target_t[selects] = outputs
        elif opt['t_draw'] == 'max':
            # idx_lb = torch.argmax(outputs, dim=-1, keepdim=True)
            target_t[selects].zero_().scatter_(1, torch.unsqueeze(preds, 1), 1.0)
        else:
            raise ValueError('opt[\'s_draw\'] should be: exp, max, smp. '
                             'But recieve: %s' % opt['t_draw'])

        if opt['use_seed'] == 1:
            temp = torch.zeros(idx_train.size(0), target_t.size(1),
                               device=device).type_as(target_t)
            temp.scatter_(1, torch.unsqueeze(target[idx_train], 1), 1.0)
            target_t[idx_train] = temp
            confidence[idx_train] = 1.
    return target_t, confidence, selects


def update_teacher(opt, model, optimizer, targets, inputs, pos_mask,
                   neg_mask, non_neighbors, near_mask, far_mask, c_mask, device, epoch):
    idx_train, es, ps, ep_adj, pe_adj = preprocess(device, *inputs)
    target_t, confidence, idx_fake = preprocess(device, *targets)
    non_neighbors = preprocess(device, non_neighbors)[0]
    idx = torch.cat([idx_train, idx_fake])
    step = idx_train.size(0) // c_mask.size(0)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
    for i in range(1, epoch+1):
        model.train()
        optimizer.zero_grad()
        logits, n_es, n_ps = model(idx_train, es, ps, ep_adj, pe_adj)

        loss = 0
        if model.encoder.iter > 0:
            n_seed = n_es[idx_train]
            group_sim = cosine_sim(n_seed, n_seed)
            n_class = heuristic_pooling(n_seed, step)
            class_sim = cosine_sim(n_class, n_class)

            pos_sim = group_sim.masked_select(pos_mask)
            neg_sim = group_sim.masked_select(neg_mask)
            c_sim = class_sim.masked_select(c_mask)

            if model.sparse:
                near_sim = torch_sim(n_es[near_mask[0]], n_ps[near_mask[1]])
                non_neighbor_sim = torch_sim(n_es[non_neighbors[0]], n_es[non_neighbors[1]])
                far_sim = torch_sim(n_es[far_mask[0]], n_ps[far_mask[1]])
                # near_sim = neighbor_sim[near_mask[0], near_mask[1]]
                # far_sim = neighbor_sim[far_mask[0], far_mask[1]]
            else:
                neighbor_sim = cosine_sim(n_es, n_ps)
                far_sim = neighbor_sim.masked_select(far_mask)
                near_sim = neighbor_sim.masked_select(near_mask)
                non_neighbor_sim = cosine_sim(n_es, n_es).masked_select(non_neighbors)
            nn_scale = opt['nn_scale']

            unsup_loss = (torch.sum(far_sim) + nn_scale * torch.sum(non_neighbor_sim) - 0.1 * torch.sum(near_sim)) /\
                (far_sim.numel() + nn_scale * non_neighbor_sim.numel() + 0.1 * near_sim.numel())
            sup_loss = torch.sum(neg_sim) - torch.sum(pos_sim) + torch.sum(c_sim)
            sup_size = pos_sim.numel() + neg_sim.numel() + c_sim.numel()
            sup_loss /= sup_size
            loss += (0.5 * sup_loss + 0.5 * unsup_loss) / opt['uns_scale']

        logits = torch.log_softmax(logits, dim=-1)
        loss_pre = torch.sum(target_t[idx] * logits[idx], dim=-1)
        loss_pre = - torch.mean(loss_pre * confidence[idx])
        loss += loss_pre
        print('loss of teacher at epoch %d: %f' % (i, loss.item()))
        loss.backward()
        optimizer.step()


def update_student_data(opt, model, target_pair, inputs, device, tau=0.1):
    idx_train, es, ps, ep_adj, pe_adj = preprocess(device, *inputs)
    target, target_s = preprocess(device, *target_pair)
    with torch.no_grad():
        model.eval()
        logits, _, _ = model(idx_train, es, ps, ep_adj, pe_adj)
        preds = torch.softmax(logits, dim=-1)

        confidence = predict_confidence(preds)
        if opt['s_draw'] == 'exp':
            target_s.copy_(preds)
        elif opt['s_draw'] == 'max':
            idx_lb = torch.argmax(logits, dim=-1, keepdim=True)
            target_s.zero_().scatter_(1, idx_lb, 1.0)
        elif opt['s_draw'] == 'smp':
            idx_lb = torch.multinomial(preds, 1).squeeze(1)
            target_s.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
        else:
            raise ValueError('opt[\'s_draw\'] should be: exp, max, smp. '
                             'But recieve: %s' % opt['s_draw'])
        if opt['use_seed'] == 1:
            temp = torch.zeros(idx_train.size(0), target_s.size(1),
                               device=target_s.device).type_as(target_s)
            temp.scatter_(1, torch.unsqueeze(target[idx_train], 1), 1.0)
            target_s[idx_train] = temp
            confidence[idx_train] = 1.0
    return target_s, confidence


def update_student(model, optimizer, targets, inputs, c_mask, device, epoch, eval_step=-1, gold_target=None, multi_target=None, idx_dev=None):
    idx_train, es, ps, ep_adj, pe_adj, nbs = preprocess(device, *inputs)
    c_mask, target_s, confidence = preprocess(device, c_mask, *targets)
    idx_dev = preprocess(device, idx_dev)[0]
    for i in range(1, epoch+1):
        model.train()
        optimizer.zero_grad()
        outputs, selects, steps, hxes = model(idx_train, es, ps, ep_adj, pe_adj, nbs)
        loss = 0
        for output, select, hx in zip(outputs, selects, hxes):
            temp_select = target_s[select]
            conf_select = confidence[select]
            class_sim = cosine_sim(hx, hx)
            c_sim = class_sim.masked_select(c_mask)
            loss += torch.mean(c_sim)
            loss += - torch.mean(conf_select * torch.sum(temp_select * output.log(), dim=-1))
        print('loss of student at epoch %d: %f' % (i, loss.item()))
        loss.backward()
        optimizer.step()
        if eval_step > 0 and(i == 1 or i % eval_step == 0):
            eval_s(model, gold_target, multi_target, inputs, device, is_multi=True, idx_dev=idx_dev)


def pre_pretrain(opt, model, optimizer, inputs, pos_mask, neg_mask,
                 non_neighbors, near_mask, far_mask, c_mask, device, epoch):
    idx_train, es, ps, ep_adj, pe_adj = preprocess(device, *inputs)
    non_neighbors = preprocess(device, non_neighbors)[0]
    model.train()
    step = idx_train.size(0) // c_mask.size(0)
    for i in range(1, epoch+1):
        optimizer.zero_grad()
        _, n_es, n_ps = model(idx_train, es, ps, ep_adj, pe_adj)

        n_seed = n_es[idx_train]
        group_sim = cosine_sim(n_seed, n_seed)
        n_class = heuristic_pooling(n_seed, step)
        class_sim = cosine_sim(n_class, n_class)

        pos_sim = group_sim.masked_select(pos_mask)
        neg_sim = group_sim.masked_select(neg_mask)
        c_sim = class_sim.masked_select(c_mask)

        if model.sparse:
            near_sim = torch_sim(n_es[near_mask[0]], n_ps[near_mask[1]])
            non_neighbor_sim = torch_sim(n_es[non_neighbors[0]], n_es[non_neighbors[1]])
            far_sim = torch_sim(n_es[far_mask[0]], n_ps[far_mask[1]])
            # near_sim = neighbor_sim[near_mask[0], near_mask[1]]
            # far_sim = neighbor_sim[far_mask[0], far_mask[1]]
        else:
            neighbor_sim = cosine_sim(n_es, n_ps)
            far_sim = neighbor_sim.masked_select(far_mask)
            near_sim = neighbor_sim.masked_select(near_mask)
            non_neighbor_sim = cosine_sim(n_es, n_es).masked_select(non_neighbors)
        nn_scale = opt['nn_scale']

        unsup_loss = (torch.sum(far_sim) + nn_scale * torch.sum(non_neighbor_sim) - 0.1 * torch.sum(near_sim)) /\
            (far_sim.numel() + nn_scale * non_neighbor_sim.numel() + 0.1 * near_sim.numel())
        sup_loss = torch.sum(neg_sim) - torch.sum(pos_sim) + torch.sum(c_sim)
        sup_size = pos_sim.numel() + neg_sim.numel() + c_sim.numel()
        sup_loss /= sup_size
        loss = (0.5 * sup_loss + 0.5 * unsup_loss)

        print('loss at pre_pretrain epoch %d: %f' % (i, loss.item()))
        loss.backward()
        optimizer.step()


def pre_train(opt, model, optimizer, target_t, inputs, pos_mask, neg_mask,
              non_neighbors, near_mask, far_mask, c_mask, device, epoch, eval_step=-1, gold_target=None, teacher_save=False):
    idx_train, es, ps, ep_adj, pe_adj = preprocess(device, *inputs)
    non_neighbors = preprocess(device, non_neighbors)[0]
    step = idx_train.size(0) // c_mask.size(0)

    for i in range(1, epoch+1):
        model.train()
        optimizer.zero_grad()
        logits, n_es, n_ps = model(idx_train, es, ps, ep_adj, pe_adj)

        loss = 0
        if model.encoder.iter > 0:
            n_seed = n_es[idx_train]
            group_sim = cosine_sim(n_seed, n_seed)
            n_class = heuristic_pooling(n_seed, step)
            class_sim = cosine_sim(n_class, n_class)

            pos_sim = group_sim.masked_select(pos_mask)
            neg_sim = group_sim.masked_select(neg_mask)
            c_sim = class_sim.masked_select(c_mask)

            if model.sparse:
                near_sim = torch_sim(n_es[near_mask[0]], n_ps[near_mask[1]])
                non_neighbor_sim = torch_sim(n_es[non_neighbors[0]], n_es[non_neighbors[1]])
                far_sim = torch_sim(n_es[far_mask[0]], n_ps[far_mask[1]])
                # near_sim = neighbor_sim[near_mask[0], near_mask[1]]
                # far_sim = neighbor_sim[far_mask[0], far_mask[1]]
            else:
                neighbor_sim = cosine_sim(n_es, n_ps)
                far_sim = neighbor_sim.masked_select(far_mask)
                near_sim = neighbor_sim.masked_select(near_mask)
                non_neighbor_sim = cosine_sim(n_es, n_es).masked_select(non_neighbors)
            nn_scale = opt['nn_scale']

            unsup_loss = (torch.sum(far_sim) + nn_scale * torch.sum(non_neighbor_sim) - 0.1 * torch.sum(near_sim)) /\
                (far_sim.numel() + nn_scale * non_neighbor_sim.numel() + 0.1 * near_sim.numel())
            sup_loss = torch.sum(neg_sim) - torch.sum(pos_sim) + torch.sum(c_sim)
            sup_size = pos_sim.numel() + neg_sim.numel() + c_sim.numel()
            sup_loss /= sup_size
            loss += (0.5 * sup_loss + 0.5 * unsup_loss) / opt['uns_scale']

        logits = torch.log_softmax(logits, dim=-1)
        loss_pre = torch.sum(target_t[idx_train] * logits[idx_train], dim=-1)
        loss += - torch.mean(loss_pre)
        print('loss at pretrain epoch %d: %f' % (i, loss.item()))
        loss.backward()
        # scheduler.step(loss)
        optimizer.step()

        if eval_step > 0 and (i == 1 or i % eval_step == 0):
            eval_t(model, gold_target, inputs, device)
            if teacher_save:
                path = os.path.join(opt['save'], 'pretrained/')
                if not os.path.exists(path):
                    os.makedirs(path)
                fn = 'teacher_%d.pt' % i
                model.save(optimizer, os.path.join(path, fn))


def multi_target_eval(preds, multi_target):
    results = torch.gather(multi_target, 1, torch.unsqueeze(preds, 1)).float().view(-1)
    return results


def dev_eval(model, target, multi_target, inputs, idx_dev,  device, is_multi=False):
    idx_train, es, ps, ep_adj, pe_adj, nbs = preprocess(device, *inputs)
    idx_dev, target, multi_target = preprocess(device, idx_dev, target, multi_target)
    model.eval()
    model.decoder.dev=True
    with torch.no_grad():
        outputs, selects, steps, _ = model(idx_train, es, ps, ep_adj, pe_adj, nbs, dev=idx_dev)
        print('===== dev eval=====')
        full_count = 0
        for out in outputs:
            full_count += out.size(0)
        preds = torch.full((full_count, ), -1, dtype=torch.long, device=outputs[0].device)
        selects = torch.cat(selects, dim=0)
        full_start = 0
        for i, (output, select, step) in enumerate(zip(outputs, selects, steps)):
            full_end = full_start + output.size(0)
            for j, length in enumerate(step):
                i_end = full_start + length
                preds[full_start:i_end] = j
                full_start = i_end
            full_start = full_end
        if is_multi:
            tags = multi_target[selects]
            prec = torch.mean(multi_target_eval(preds, tags))
        else:
            tags = target[selects]
            prec = preds.eq(tags).float().mean()
        print('acc:%.4f' % prec.item())

    model.decoder.dev = False


def eval_s(model, target, multi_target, inputs, device, is_multi=False, idx_dev=None):
    idx_train, es, ps, ep_adj, pe_adj, nbs = preprocess(device, *inputs)
    target, multi_target = preprocess(device, target, multi_target)
    if idx_dev is not None:
        idx_dev = preprocess(device, idx_dev)[0]
    model.eval()
    with torch.no_grad():
        outputs, selects, steps, _ = model(idx_train, es, ps, ep_adj, pe_adj, nbs, dev=idx_dev)
        print('===== student eval=====')
        line = '\t'.join([str(i) for i in range(outputs[0].size(1))])
        print('>> step\t' + line + '\ttotal count\tp@n')
        total_count = 0
        full_count = 0
        for out in outputs:
            full_count += out.size(0)
        total_correct = .0
        preds = torch.full((full_count, ), -1, dtype=torch.long, device=outputs[0].device)
        full_start = 0
        for i, (output, select, step) in enumerate(zip(outputs, selects, steps)):
            start = 0
            full_end = full_start + output.size(0)
            s_preds = torch.full((output.size(0), ), -1, dtype=torch.long, device=output.device)
            for j, length in enumerate(step):
                end = start + length
                i_end = full_start + length
                if end == start:
                    continue
                s_preds[start:end] = j
                preds[full_start:i_end] = j
                start = end
                full_start = i_end
            full_start = full_end
            # s_count = output.size(0)
            # s_preds = torch.max(output, dim=1)[1]
            if is_multi:
                s_tag = multi_target[select]
                s_correct = multi_target_eval(s_preds, s_tag).sum()
            else:
                s_tag = target[select]
                s_correct = s_preds.eq(s_tag).float().sum()
            total_correct += s_correct
            total_count += output.size(0)
            total_acc = total_correct / total_count
            s_temp = []
            for j in range(output.size(1)):
                ss_select = s_preds == j
                ss_preds = s_preds[ss_select]
                ss_index = select[ss_select]
                if is_multi:
                    ss_tag = multi_target[ss_index]
                    ss_correct = multi_target_eval(ss_preds, ss_tag).sum()
                else:
                    ss_tag = target[ss_index]
                    ss_correct = ss_preds.eq(ss_tag).float().sum()
                ss_count = ss_preds.size(0)
                ss_acc = float(ss_correct) / max(ss_count, 1)
                s_temp.append('%d/%d(%.3f)' % (ss_correct, ss_count, ss_acc))
            s_line = '\t'.join(s_temp)
            print('>> %d\t%s\t%d\t%.4f' %
                  (i+1, s_line, total_count, total_acc))

        outputs = torch.cat(outputs, dim=0)
        selects = torch.cat(selects, dim=0)
        # preds = torch.max(outputs, dim=1)[1]
        results = [[] for _ in range(outputs.size(1))]
        for i in range(outputs.size(1)):
            sub_selects = preds == i
            s_preds = preds[sub_selects]
            sub_index = selects[sub_selects]
            if is_multi:
                s_target = multi_target[sub_index]
            else:
                s_target = target[sub_index]
            for t in range(10, 210, 10):
                if s_preds.size(0) < t:
                    break
                if is_multi:
                    correct = multi_target_eval(s_preds[:t], s_target[:t]).sum()
                else:
                    correct = s_preds[:t].eq(s_target[:t]).float().sum()
                accuracy = correct / t
                results[i].append(accuracy.item())
        print('>>> details <<<')
        line = '\t'.join([str(i) for i in range(outputs.size(1))])
        print('>> N\t' + line + '\ttotal')
        assert selects.size(0) == selects.unique().size(0)
        for s, t in enumerate(range(10, 210, 10)):
            line = '>> %d' % t
            is_total = True
            total = []
            for i in range(outputs.size(1)):
                if len(results[i]) > s:
                    total.append(results[i][s])
                    line += '\t%.3f' % results[i][s]
                else:
                    is_total = False
                    line += '\t-'
            if is_total:
                line += '\t%.4f' % (sum(total) / len(total))
            else:
                line += '\t-'
            print(line)
        '''
        for output, select in zip(outputs, selects):
            total += output.size(0)
            pred = torch.argmax(output, dim=-1)
            accuracy = torch.eq(pred, target[select])
            accuracies.append(accuracy)
            acc = torch.mean(torch.cat(accuracies).float())
            print('>>> accuracy %d is %f' % (total, acc.item()))
        '''


def eval_t(model, target, inputs, device):
    idx_train, es, ps, ep_adj, pe_adj = preprocess(device, *inputs)
    target = preprocess(device, target)[0]
    model.eval()
    mask = torch.ones_like(target, dtype=torch.bool, device=target.device)
    mask[idx_train] = 0
    with torch.no_grad():
        outputs, _, _ = model(idx_train, es, ps, ep_adj, pe_adj)
        logits = torch.softmax(outputs, dim=-1)
        preds = torch.max(outputs, dim=1)[1]
        preds = preds[mask]
        target = target[mask]
        results = [[] for _ in range(outputs.size(1))]
        entropy = - torch.sum(logits * logits.log(), dim=-1)
        entropy = entropy[mask]
        for i in range(outputs.size(1)):
            selects = preds == i
            s_entropy = entropy[selects]
            s_preds = preds[selects]
            s_target = target[selects]
            sort = torch.argsort(s_entropy)
            for t in range(10, 220, 10):
                if sort.size(0) < t:
                    break
                correct = s_preds[sort[:t]].eq(s_target[sort[:t]]).double()
                accuracy = correct.sum() / t
                results[i].append(accuracy.item())
        print('===== teacher eval =====')
        line = '\t'.join([str(i) for i in range(outputs.size(1))])
        print('>>>\tstep\t' + line + '\ttotal')
        for s, t in enumerate(range(10, 220, 10)):
            line = '>>>\t%d' % t
            is_total = True
            total = []
            for i in range(outputs.size(1)):
                if len(results[i]) > s:
                    total.append(results[i][s])
                    line += '\t%.3f' % results[i][s]
                else:
                    is_total = False
                    line += '\t-'
            if is_total:
                line += '\t%.4f' % (sum(total) / len(total))
            else:
                line += '\t-'
            print(line)

        correct = preds.eq(target).double()
        acc = correct.sum() / target.size(0)
        total_num = target.size(0)
        print('>>> accuracy of teacher [%d] is %f' % (total_num, acc.item()))


def boot_train(opt, teacher, t_optimizer, student, s_optimizer,
               target, multi_target, idx_train, es, ps, ep_adj, idx_dev=None, teacher_save=False):
    n_class, n_per_class = opt['num_class'], opt['n_per_class']
    device, s_device = opt['device'], opt['s_device']
    # n_entity, n_pattern = ep_adj.size(0), ep_adj.size(1)
    pre_epoch, prepre_epoch = opt['t_pre_epoch'], opt['t_prepre_epoch']
    t_epoch = opt['t_epoch']
    save_flag = False
    save_path = opt['quick_save']

    pos_mask, neg_mask = group_mask(n_class, n_per_class, device)
    c_mask = class_mask(n_class, device=device)
    near_mask, far_mask = neighbor_mask(ep_adj, device, 12, save_path)
    neighbors, non_neighbors = neighbor_adj(ep_adj, device, length=20, file_path=save_path)
    ep_adj, pe_adj = bidirection_adj_eff(ep_adj, file_path=save_path)
    '''
    if opt['sparse']:
        ep_step = []
        for i in range(n_entity):
            ep_step.append((ep_adj[0] == i).sum(dim=0, keepdim=True))
        ep_step = torch.cat(ep_step)
        ep_adj = (ep_adj, ep_step)
        pe_step = []
        for i in range(n_pattern):
            pe_step.append((pe_adj[0] == i).sum(dim=0, keepdim=True))
        pe_step = torch.cat(pe_step)
        pe_adj = (pe_adj, pe_step)
    '''
    inputs_t = (idx_train, es, ps, ep_adj, pe_adj)
    inputs_s = (idx_train, es, ps, ep_adj, pe_adj, neighbors)

    target_t = torch.zeros(opt['num_node'], opt['num_class'])
    target_t = target_t.to(device)
    target_s = torch.zeros(opt['num_node'], opt['num_class'])
    target_s = target_s.to(device)

    # gpu_memory_log()

    print('========== pretrain iter ==========')
    if opt['load'] is None:
        target_pt = (target, target_t)
        target_t = init_teacher_data(target_pt, idx_train, device)
        if teacher.encoder.iter > 0:
            pre_pretrain(opt, teacher, t_optimizer, inputs_t, pos_mask, neg_mask,
                         non_neighbors, near_mask, far_mask, c_mask, device, prepre_epoch)
        pre_train(opt, teacher, t_optimizer, target_t, inputs_t, pos_mask, neg_mask,
                  non_neighbors, near_mask, far_mask, c_mask, device, pre_epoch, eval_step=50, gold_target=target, teacher_save=teacher_save)
    eval_t(teacher, target, inputs_t, device)

    if opt['initialization'] != 'embedding':
        es = es.detach()
        ps = ps.detach()
        inputs_t = (idx_train, es, ps, ep_adj, pe_adj)
        inputs_s = (idx_train, es, ps, ep_adj, pe_adj, neighbors)

    for train_iter in range(1, opt['iter']+1):
        student.encoder.load_state_dict(teacher.encoder.state_dict())
        if train_iter > 1:
            s_epoch = opt['s_epoch']
        else:
            s_epoch = opt['s_pre_epoch']
        print('========== dual training iter %d ==========' % train_iter)
        if opt['save'] != '/':
            dir_ = opt['save']
            dir_ += ('/iter_%d' % train_iter)
            if not os.path.exists(dir_):
                os.makedirs(dir_)
            if train_iter == 1 or train_iter % opt['save_step'] == 0:
                save_flag = True

        if save_flag:
            teacher.save(t_optimizer, dir_+'/teacher.pt')

        target_ps = (target, target_s)
        targets_s = update_student_data(opt, teacher, target_ps, inputs_t,
                                        device)
        update_student(student, s_optimizer, targets_s, inputs_s, c_mask,
                       s_device, s_epoch, eval_step=50, gold_target=target,
                       multi_target=multi_target, idx_dev=idx_dev)
        if idx_dev is not None:
            dev_eval(student, target, multi_target, inputs_s, idx_dev, s_device, is_multi=opt['is_multi'])
        eval_s(student, target, multi_target, inputs_s, s_device, is_multi=opt['is_multi'], idx_dev=idx_dev)

        if save_flag:
            student.save(s_optimizer, dir_+'/student.pt')

        target_pt = (target, target_t)
        targets_t = update_teacher_data(opt, student, target_pt, inputs_s,
                                        s_device)
        update_teacher(opt, teacher, t_optimizer, targets_t, inputs_t, pos_mask,
                       neg_mask, non_neighbors, near_mask, far_mask, c_mask, device, t_epoch)
        eval_t(teacher, target, inputs_t, device)

    if opt['save'] != '/':
        if not os.path.exists(opt['save']):
            os.makedirs(opt['save'])
        teacher.save(t_optimizer, opt['save']+'/teacher.pt')
        student.save(s_optimizer, opt['save']+'/student.pt')
    return teacher, student


def noboot_train(opt, teacher, t_optimizer, student, s_optimizer,
                 target, multi_target, idx_train, es, ps, ep_adj):
    n_class, n_per_class = opt['num_class'], opt['n_per_class']
    device, s_device = opt['device'], opt['s_device']
    pre_epoch, prepre_epoch = opt['t_pre_epoch'], opt['t_prepre_epoch']
    save_path = opt['quick_save']

    pos_mask, neg_mask = group_mask(n_class, n_per_class, device)
    c_mask = class_mask(n_class, device=device)
    near_mask, far_mask = neighbor_mask(ep_adj, device, 12, save_path)
    neighbors, non_neighbors = neighbor_adj(ep_adj, device, file_path=save_path)
    ep_adj, pe_adj = bidirection_adj_eff(ep_adj, file_path=save_path)

    inputs_t = (idx_train, es, ps, ep_adj, pe_adj)
    inputs_s = (idx_train, es, ps, ep_adj, pe_adj, neighbors)

    target_t = torch.zeros(opt['num_node'], opt['num_class'])
    target_t = target_t.to(device)

    # gpu_memory_log()

    print('========== pretrain iter ==========')
    if opt['load'] is None:
        target_pt = (target, target_t)
        target_t = init_teacher_data(target_pt, idx_train, device)
        if teacher.encoder.iter > 0:
            pre_pretrain(opt, teacher, t_optimizer, inputs_t, pos_mask, neg_mask,
                         non_neighbors, near_mask, far_mask, c_mask, device, prepre_epoch)
        pre_train(opt, teacher, t_optimizer, target_t, inputs_t, pos_mask, neg_mask,
                  non_neighbors, near_mask, far_mask, c_mask, device, pre_epoch)
    eval_t(teacher, target, inputs_t, device)
    student.encoder.load_state_dict(teacher.encoder.state_dict())
    eval_s(student, target, multi_target, inputs_s, s_device, is_multi=opt['is_multi'])
    if opt['load'] is None:
        dir_ = opt['save']
        if not os.path.exists(dir_):
            os.makedirs(dir_)
        teacher.save(t_optimizer, dir_+'/teacher.pt')
        student.save(s_optimizer, dir_+'/student.pt')
