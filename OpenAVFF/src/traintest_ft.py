# traintest_ft.py
import sys
import os
import datetime
import time
from utilities import *
import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

def _to_probs(logits, loss_type: str):
    return torch.sigmoid(logits) if loss_type == 'BCE' else F.softmax(logits, dim=1)

def _collect_misclassified(probs, targets, paths_or_none):
    pred_cls = probs.argmax(dim=1)
    true_cls = targets.argmax(dim=1)
    wrong_idx = (pred_cls != true_cls).nonzero(as_tuple=False).view(-1).tolist()
    if paths_or_none is None or len(paths_or_none) != probs.shape[0]:
        return wrong_idx
    return [paths_or_none[i] for i in wrong_idx]

def train(model, train_loader, val_loader, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_grad_enabled(True)

    # meters 남겨두되, 에폭 요약만 출력
    batch_time, per_sample_time, data_time = AverageMeter(), AverageMeter(), AverageMeter()
    per_sample_data_time, per_sample_dnn_time = AverageMeter(), AverageMeter()
    loss_meter = AverageMeter()

    best_epoch, best_mAP, best_acc, best_AUC = 0, -np.inf, -np.inf, -np.inf
    best_wrong_paths = []
    global_step, epoch = 0, 0

    if not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)
    model.to(device)

    mlp_list = [
        'a2v.mlp.linear.weight','a2v.mlp.linear.bias',
        'v2a.mlp.linear.weight','v2a.mlp.linear.bias',
        'mlp_vision.weight','mlp_vision.bias',
        'mlp_audio.weight','mlp_audio.bias',
        'mlp_head.fc1.weight','mlp_head.fc1.bias',
        'mlp_head.fc2.weight','mlp_head.fc2.bias'
    ]
    mlp_params = [p for n,p in model.module.named_parameters() if n in mlp_list]
    base_params = [p for n,p in model.module.named_parameters() if n not in mlp_list]

    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6))
    optimizer = torch.optim.Adam(
        [{'params': base_params, 'lr': args.lr},
         {'params': mlp_params, 'lr': args.lr * args.head_lr}],
        weight_decay=5e-7, betas=(0.95, 0.999)
    )
    print('base lr, mlp lr : ', optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'])

    print('Total newly initialized MLP parameter number is : {:.3f} million'.format(sum(p.numel() for p in mlp_params) / 1e6))
    print('Total pretrained backbone parameter number is : {:.3f} million'.format(sum(p.numel() for p in base_params) / 1e6))

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),
        gamma=args.lrscheduler_decay
    )
    main_metrics = args.metrics

    # >>> 변경: 기본 CE 사용 (run_ft.py 기본값과 일치)
    loss_fn = nn.BCEWithLogitsLoss() if args.loss == 'BCE' else nn.CrossEntropyLoss()
    args.loss_fn = loss_fn

    epoch += 1
    scaler = GradScaler()

    print(f"{datetime.datetime.now()}")
    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    model.train()

    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        model.train()
        # 에폭별 로스 평균을 위해 합계/개수 집계
        train_loss_sum, train_count = 0.0, 0

        # (에폭 머리말만)
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))

        for i, batch in enumerate(train_loader):
            a_input, v_input, labels = batch
            B = a_input.shape[0]
            a_input = a_input.to(device, non_blocking=True)
            v_input = v_input.to(device, non_blocking=True)
            labels = labels.to(device)

            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / B)
            dnn_start_time = time.time()

            with autocast():
                logits = model(a_input, v_input)
                if isinstance(args.loss_fn, nn.CrossEntropyLoss):
                    # CE는 정수 라벨을 사용 → one-hot을 argmax로 변환
                    targets_ce = labels.argmax(dim=1)
                    loss = args.loss_fn(logits, targets_ce)
                else:
                    loss = args.loss_fn(logits, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 누적만 하고 중간 출력 없음!
            train_loss_sum += loss.item() * B
            train_count += B

            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time)/B)
            per_sample_dnn_time.update((time.time() - dnn_start_time)/B)

            end_time = time.time()
            global_step += 1

        # 에폭 평균 train loss
        train_loss_epoch = train_loss_sum / max(1, train_count)

        # ===== Validation on val_loader (data_val) =====
        print('start validation')
        stats, valid_loss, mAUC, wrong_paths = validate(model, val_loader, args, output_pred=True)

        mAP = np.mean([stat['AP'] for stat in stats])
        acc = stats[0]['acc']  # (구현상 동일 trick)

        # 에폭 요약 한 번만 출력
        if main_metrics == 'mAP':
            print("mAP: {:.6f}".format(mAP))
        else:
            print("acc: {:.6f}".format(acc))
        print("AUC: {:.6f}".format(mAUC))
        print("d_prime: {:.6f}".format(d_prime(mAUC) if mAUC < 1.0 else float('inf')))
        print("train_loss: {:.6f}".format(train_loss_epoch))
        print("valid_loss: {:.6f}".format(valid_loss))
        print('validation finished')

        # best AUC 시 오답 경로 저장(유지)
        if mAUC > best_AUC:
            best_AUC = mAUC
            best_epoch = epoch
            out_path = os.path.join(os.getcwd(), "misclassified_best_auc.txt")
            try:
                with open(out_path, "w") as f:
                    f.write("# Misclassified samples at best AUC\n")
                    f.write(f"# epoch={epoch}, AUC={mAUC:.6f}\n")
                    f.write("# path, prob_real, prob_fake\n")
                    for (p, pr, pf) in wrong_paths or []:
                        f.write(f"{p}, {pr:.6f}, {pf:.6f}\n")
                print(f"[+] Saved misclassified list -> {out_path} (count={len(wrong_paths or [])})")
            except Exception as e:
                print(f"[!] Failed to save misclassified list: {e}")

        # 스케줄러만 스텝
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(mAUC if main_metrics == 'mAP' else acc)
        else:
            scheduler.step()

        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
        print('epoch {:d} training time: {:.3f}'.format(epoch, time.time()-begin_time))

        epoch += 1

        # meters 리셋
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
    logits_all, targets_all, loss_list, paths_all = [], [], [], []

    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 4:
                a_input, v_input, labels, paths = batch
            else:
                a_input, v_input, labels = batch
                paths = None

            a_input = a_input.to(device)
            v_input = v_input.to(device)
            labels = labels.to(device)

            with autocast():
                logits = model(a_input, v_input)
                if isinstance(args.loss_fn, nn.CrossEntropyLoss):
                    targets_ce = labels.argmax(dim=1)
                    loss = args.loss_fn(logits, targets_ce)
                else:
                    loss = args.loss_fn(logits, labels)

            logits_all.append(logits.detach().cpu())
            targets_all.append(labels.detach().cpu())
            loss_list.append(loss.detach().cpu())

            if paths is not None:
                paths_all.extend(list(paths))

            batch_time.update(time.time() - end)
            end = time.time()

    logits_all = torch.cat(logits_all, dim=0)
    targets_all = torch.cat(targets_all, dim=0)
    loss = float(np.mean([x.item() for x in loss_list]))

    # ---- convert to probabilities for metric computation ----
    probs_all = _to_probs(logits_all, args.loss)  # (N, 2)
    stats = calculate_stats(probs_all, targets_all)
    mAUC = float(np.mean([s['auc'] for s in stats]))

    # ---- misclassified list (with probs) ----
    wrong_infos = []
    if len(paths_all) == probs_all.shape[0]:
        pred_cls = probs_all.argmax(dim=1)
        true_cls = targets_all.argmax(dim=1)
        wrong_idx = (pred_cls != true_cls).nonzero(as_tuple=False).view(-1).tolist()
        for i in wrong_idx:
            pr = float(probs_all[i, 0])
            pf = float(probs_all[i, 1])
            wrong_infos.append((paths_all[i], pr, pf))

    if not output_pred:
        return stats, loss
    else:
        # wrong_infos에 (path, pr, pf) 들어 있음
        return stats, loss, mAUC, wrong_infos
