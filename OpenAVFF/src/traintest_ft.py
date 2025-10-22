'''
import sys
import os
import datetime
import time
from utilities import *
import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import autocast, GradScaler

def train(model, train_loader, test_loader, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_grad_enabled(True)
    
    batch_time, per_sample_time, data_time, per_sample_data_time, per_sample_dnn_time = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    loss_meter = AverageMeter()
    
    best_epoch, best_mAP, best_acc = 0, -np.inf, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.save_dir
    
    if not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)
    
    model.to(device)
    
    # possible mlp layer name list, mlp layers are newly initialized layers in the finetuning stage (i.e., not pretrained) and should use a larger lr during finetuning
    mlp_list = [
        'a2v.mlp.linear.weight',
        'a2v.mlp.linear.bias',
        'v2a.mlp.linear.weight',
        'v2a.mlp.linear.bias',
        'mlp_vision.weight',
        'mlp_vision.bias',
        'mlp_audio.weight',
        'mlp_audio.bias',
        'mlp_head.fc1.weight',
        'mlp_head.fc1.bias',
        'mlp_head.fc2.weight',
        'mlp_head.fc2.bias'
    ]
    mlp_params = list(filter(lambda kv: kv[0] in mlp_list, model.module.named_parameters()))
    base_params = list(filter(lambda kv: kv[0] not in mlp_list, model.module.named_parameters()))
    mlp_params = [i[1] for i in mlp_params]
    base_params = [i[1] for i in base_params]
    
    trainables = [p for p in model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
    optimizer = torch.optim.Adam([{'params': base_params, 'lr': args.lr}, {'params': mlp_params, 'lr': args.lr * args.head_lr}], weight_decay=5e-7, betas=(0.95, 0.999))
    base_lr = optimizer.param_groups[0]['lr']
    mlp_lr = optimizer.param_groups[1]['lr']
    lr_list = [args.lr, mlp_lr]
    print('base lr, mlp lr : ', base_lr, mlp_lr)
    
    print('Total newly initialized MLP parameter number is : {:.3f} million'.format(sum(p.numel() for p in mlp_params) / 1e6))
    print('Total pretrained backbone parameter number is : {:.3f} million'.format(sum(p.numel() for p in base_params) / 1e6))
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),gamma=args.lrscheduler_decay)
    main_metrics = args.metrics
    
    if args.loss == 'BCE':
        loss_fn = nn.BCEWithLogitsLoss()
    elif args.loss == 'CE':
        loss_fn = nn.CrossEntropyLoss()
    args.loss_fn = loss_fn
    
    epoch += 1
    scaler = GradScaler()
    
    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    result = np.zeros([args.n_epochs, 4])  # for each epoch, 10 metrics to record
    model.train()
    
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        model.train()
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))
        
        for i, (a_input, v_input, labels) in enumerate(train_loader):
            assert a_input.shape[0] == v_input.shape[0]
            B = a_input.shape[0]
            a_input = a_input.to(device, non_blocking=True)
            v_input = v_input.to(device, non_blocking=True)
            labels = labels.to(device)
            
            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / B)
            dnn_start_time = time.time()
            
            with autocast():
                output = model(a_input, v_input)
                loss = loss_fn(output, labels)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # loss_av is the main loss
            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time)/a_input.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time)/a_input.shape[0])

            print_step = global_step % args.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (args.n_print_steps/10) == 0
            print_step = print_step or early_print_step

            if print_step and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Per Sample Total Time {per_sample_time.avg:.5f}\t'
                  'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
                  'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
                  'Train Loss {loss_meter.val:.4f}\t'.format(
                   epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                      per_sample_dnn_time=per_sample_dnn_time, loss_meter=loss_meter), flush=True)
                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return

            end_time = time.time()
            global_step += 1
        
        print('start validation')
        stats, valid_loss = validate(model, test_loader, args)

        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        acc = stats[0]['acc'] # this is just a trick, acc of each class entry is the same, which is the accuracy of all classes, not class-wise accuracy

        if main_metrics == 'mAP':
            print("mAP: {:.6f}".format(mAP))
        else:
            print("acc: {:.6f}".format(acc))
        print("AUC: {:.6f}".format(mAUC))
        print("d_prime: {:.6f}".format(d_prime(mAUC)))
        print("train_loss: {:.6f}".format(loss_meter.avg))
        print("valid_loss: {:.6f}".format(valid_loss))

        result[epoch-1, :] = [acc, mAP, mAUC, optimizer.param_groups[0]['lr']]
        np.savetxt(exp_dir + '/result.csv', result, delimiter=',')
        print('validation finished')
        
        if mAP > best_mAP:
            best_mAP = mAP
            if main_metrics == 'mAP':
                best_epoch = epoch

        if acc > best_acc:
            best_acc = acc
            if main_metrics == 'acc':
                best_epoch = epoch

        if best_epoch == epoch:
            torch.save(model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))
            torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir))
        if args.save_model == True:
            torch.save(model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, epoch))
        
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if main_metrics == 'mAP':
                scheduler.step(mAP)
            elif main_metrics == 'acc':
                scheduler.step(acc)
        else:
            scheduler.step()
            
        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
        
        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time-begin_time))

        epoch += 1

        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        per_sample_dnn_time.reset()
        loss_meter.reset()
        
def validate(model, val_loader, args, output_pred=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)
    model = model.to(device)
    model.eval()

    end = time.time()
    A_predictions, A_targets, A_loss = [], [], []
    with torch.no_grad():
        for i, (a_input, v_input, labels) in enumerate(val_loader):
            a_input = a_input.to(device)
            v_input = v_input.to(device)
            labels = labels.to(device)

            with autocast():
                audio_output = model(a_input, v_input)

            predictions = audio_output.to('cpu').detach()

            A_predictions.append(predictions)
            A_targets.append(labels)

            labels = labels.to(device)
            loss = args.loss_fn(audio_output, labels)
            A_loss.append(loss.to('cpu').detach())

            batch_time.update(time.time() - end)
            end = time.time()

        audio_output = torch.cat(A_predictions)
        target = torch.cat(A_targets)
        loss = np.mean(A_loss)

        stats = calculate_stats(audio_output.cpu(), target.cpu())

    if output_pred == False:
        return stats, loss
    else:
        # used for multi-frame evaluation (i.e., ensemble over frames), so return prediction and target
        return stats, audio_output, target
    
'''

# traintest_ft.py (DDP-safe, no DataParallel, AMP on)
import os
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from utilities import *

# ---------- DDP helpers ----------
def _world():
    return int(os.environ.get("WORLD_SIZE", "1"))

def _rank():
    return int(os.environ.get("RANK", "0"))

def _local_rank():
    return int(os.environ.get("LOCAL_RANK", "0"))

def _is_dist():
    return _world() > 1

def _device():
    if torch.cuda.is_available():
        if _is_dist():
            torch.cuda.set_device(_local_rank())
            return torch.device(f"cuda:{_local_rank()}")
        return torch.device("cuda:0")
    return torch.device("cpu")

def _is_rank0():
    return (not _is_dist()) or (_rank() == 0)

# ----------------------------------

def _split_params_for_head(base_model: nn.Module):
    """
    분류 head/MLP 계열 파라미터를 따로 뽑아서 다른 lr를 줄 수 있게 분리.
    모델 래퍼(DDP) 유무에 상관없이 동작.
    """
    m = base_model.module if hasattr(base_model, "module") else base_model
    mlp_names = {
        'a2v.mlp.linear.weight',
        'a2v.mlp.linear.bias',
        'v2a.mlp.linear.weight',
        'v2a.mlp.linear.bias',
        'mlp_vision.weight',
        'mlp_vision.bias',
        'mlp_audio.weight',
        'mlp_audio.bias',
        'mlp_head.fc1.weight',
        'mlp_head.fc1.bias',
        'mlp_head.fc2.weight',
        'mlp_head.fc2.bias',
    }
    mlp_params, base_params = [], []
    for name, p in m.named_parameters():
        (mlp_params if name in mlp_names else base_params).append(p)
    return base_params, mlp_params

def train(model, train_loader, val_loader, args):
    device = _device()
    torch.set_grad_enabled(True)

    # meters
    batch_time = AverageMeter()
    per_sample_time = AverageMeter()
    data_time = AverageMeter()
    per_sample_data_time = AverageMeter()
    per_sample_dnn_time = AverageMeter()
    loss_meter = AverageMeter()

    best_epoch, best_mAP, best_acc = 0, -np.inf, -np.inf
    global_step, epoch = 0, 1
    exp_dir = args.save_dir

    # === 절대 DP 금지 ===
    # if not isinstance(model, torch.nn.DataParallel): model = torch.nn.DataParallel(model)  # 제거

    model.to(device)

    # optimizer: head_lr 해석 (multiplier or absolute)
    base_params, mlp_params = _split_params_for_head(model)
    trainables = [p for p in model.parameters() if p.requires_grad]

    if _is_rank0():
        print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in model.parameters()) / 1e6))
        print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))

    base_lr = float(args.lr)
    if float(args.head_lr) >= 1.0:
        mlp_lr = base_lr * float(args.head_lr)   # head_lr을 배수로 해석
    else:
        mlp_lr = float(args.head_lr)            # head_lr을 절대값으로 해석

    optimizer = torch.optim.Adam(
        [{'params': base_params, 'lr': base_lr},
         {'params': mlp_params, 'lr': mlp_lr}],
        weight_decay=5e-7, betas=(0.95, 0.999)
    )
    if _is_rank0():
        print('base lr, mlp lr : ', base_lr, mlp_lr)
        print('Total newly initialized MLP parameter number is : {:.3f} million'.format(sum(p.numel() for p in mlp_params) / 1e6))
        print('Total pretrained backbone parameter number is : {:.3f} million'.format(sum(p.numel() for p in base_params) / 1e6))

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(range(int(args.lrscheduler_start), 1000, int(args.lrscheduler_step))),
        gamma=float(args.lrscheduler_decay)
    )

    main_metrics = args.metrics
    if args.loss == 'BCE':
        loss_fn = nn.BCEWithLogitsLoss()
    elif args.loss == 'CE':
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss {args.loss}")
    args.loss_fn = loss_fn

    scaler = GradScaler()

    if _is_rank0():
        print("current #steps=%s, #epochs=%s" % (global_step, epoch))
        print("start training...")

    result = np.zeros([args.n_epochs, 4])
    model.train()

    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        model.train()

        if _is_rank0():
            print('---------------')
            print(datetime.datetime.now())
            print("current #epochs=%s, #steps=%s" % (epoch, global_step))

        # DDP일 때 epoch 셔플 동기화
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        for i, (a_input, v_input, labels) in enumerate(train_loader):
            B = a_input.shape[0]
            a_input = a_input.to(device, non_blocking=True)
            v_input = v_input.to(device, non_blocking=True)
            labels  = labels.to(device, non_blocking=True)

            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / max(B, 1))
            dnn_start_time = time.time()

            optimizer.zero_grad(set_to_none=True)
            with autocast(dtype=torch.float16):
                output = model(a_input, v_input)
                loss = loss_fn(output, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # meters
            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time)/max(B, 1))
            per_sample_dnn_time.update((time.time() - dnn_start_time)/max(B, 1))

            global_step += 1
            if _is_rank0():
                print_step = (global_step % int(args.n_print_steps) == 0)
                early_print_step = (epoch == 0 and global_step % max(int(args.n_print_steps)//10,1) == 0)
                if (print_step or early_print_step) and global_step != 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Per Sample Total Time {per_sample_time.avg:.5f}\t'
                          'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
                          'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
                          'Train Loss {loss_meter.val:.4f}\t'.format(
                          epoch, i, len(train_loader),
                          per_sample_time=per_sample_time,
                          per_sample_data_time=per_sample_data_time,
                          per_sample_dnn_time=per_sample_dnn_time,
                          loss_meter=loss_meter), flush=True)
                    if np.isnan(loss_meter.avg):
                        print("training diverged...")
                        return

            end_time = time.time()

        # ===== Validation (rank0만 수행) =====
        if _is_rank0():
            print('start validation')
            stats, valid_loss = validate(model, val_loader, args)

            mAP = np.mean([stat['AP'] for stat in stats])
            mAUC = np.mean([stat['auc'] for stat in stats])
            acc = stats[0]['acc']  # (레포 로직 유지)

            if main_metrics == 'mAP':
                print("mAP: {:.6f}".format(mAP))
            else:
                print("acc: {:.6f}".format(acc))
            print("AUC: {:.6f}".format(mAUC))
            print("d_prime: {:.6f}".format(d_prime(mAUC)))
            print("train_loss: {:.6f}".format(loss_meter.avg))
            print("valid_loss: {:.6f}".format(valid_loss))

            result[epoch-1, :] = [acc, mAP, mAUC, optimizer.param_groups[0]['lr']]
            np.savetxt(os.path.join(exp_dir, 'result.csv'), result, delimiter=',')
            print('validation finished')

            improved = False
            if main_metrics == 'mAP' and mAP > best_mAP:
                best_mAP, best_epoch, improved = mAP, epoch, True
            if main_metrics == 'acc' and acc > best_acc:
                best_acc, best_epoch, improved = acc, epoch, True

            # 저장은 rank0만, DDP면 model.module.state_dict()
            to_save = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            if improved:
                torch.save(to_save, f"{exp_dir}/models/best_audio_model.pth")
                torch.save(optimizer.state_dict(), f"{exp_dir}/models/best_optim_state.pth")
            if bool(args.save_model):
                torch.save(to_save, f"{exp_dir}/models/audio_model.{epoch}.pth")

            print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
            finish_time = time.time()
            print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time - begin_time))

        # 모든 rank가 step 동기화 후 스케줄러 진행
        if _is_dist():
            torch.distributed.barrier()
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            # 여기선 main_metrics 기준으로 rank0만 step하고, 그 외 rank는 동일 lr로 맞추려면 브로드캐스트 필요
            # 간단화: MultiStepLR만 사용하거나, 필요한 경우 metric을 bcast하세요.
            pass
        else:
            scheduler.step()

        # reset meters
        batch_time.reset(); per_sample_time.reset()
        data_time.reset(); per_sample_data_time.reset()
        per_sample_dnn_time.reset(); loss_meter.reset()

        epoch += 1


def validate(model, val_loader, args, output_pred=False):
    device = _device()
    batch_time = AverageMeter()

    # DP 래핑 금지
    # if not isinstance(model, nn.DataParallel): model = nn.DataParallel(model)
    model = model.to(device)
    model.eval()

    # 멀티GPU에서는 rank0만 검증을 수행하도록 run_ft.py에서 DataLoader를 구성했으면 가장 좋음.
    # 여기서는 간단히 모든 rank가 돌더라도 결과/저장은 rank0만 하도록 처리.
    end = time.time()
    A_predictions, A_targets, A_loss = [], [], []
    with torch.no_grad():
        for i, (a_input, v_input, labels) in enumerate(val_loader):
            a_input = a_input.to(device, non_blocking=True)
            v_input = v_input.to(device, non_blocking=True)
            labels  = labels.to(device, non_blocking=True)

            with autocast(dtype=torch.float16):
                audio_output = model(a_input, v_input)

            predictions = audio_output.detach().cpu()
            A_predictions.append(predictions)
            A_targets.append(labels.detach().cpu())

            loss = args.loss_fn(audio_output, labels)
            A_loss.append(loss.detach().cpu())

            batch_time.update(time.time() - end)
            end = time.time()

        audio_output = torch.cat(A_predictions) if len(A_predictions) else torch.empty(0)
        target = torch.cat(A_targets) if len(A_targets) else torch.empty(0)
        loss = float(np.mean([x.item() for x in A_loss])) if len(A_loss) else 0.0

        stats = calculate_stats(audio_output, target)

    if not output_pred:
        return stats, loss
    else:
        return stats, audio_output, target
