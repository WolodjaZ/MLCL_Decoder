import os
import argparse

import shutil
import umap
import math
import wandb
import torch
import torch.nn.parallel
import torch.optim
import plotly.graph_objects as go
import torch.utils.data.distributed
from sklearn.metrics import roc_auc_score
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from src_files.helper_functions.helper_functions import mAP, CocoDetection, CutoutPIL, ModelEma, \
    TwoCropTransform, adjust_learning_rate, warmup_learning_rate, add_weight_decay
from src_files.ml_decoder.ml_decoder import add_ml_supcon_head, add_valid_linear_classification, \
    add_ml_decoder_head
from src_files.models import create_model_base
from src_files.loss_functions.losses import AsymmetricLoss, MultiSupConLoss
from randaugment import RandAugment
from torch.cuda.amp import GradScaler, autocast

def parse_option():
    parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')

    #Defult params
    parser.add_argument('--data', type=str, default='/home/MSCOCO_2014/')
    parser.add_argument('--model-name', default='tresnet_l')
    parser.add_argument('--model-path', default='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ML_Decoder/tresnet_l_pretrain_ml_decoder.pth', type=str)
    parser.add_argument('--num-classes', default=80)
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--image-size', default=448, type=int,
                        metavar='N', help='input image size (default: 448)')
    parser.add_argument('--freeze', default=True, type=bool,
                        metavar='N', help='freeze backbone')
    parser.add_argument('--run', default=0, type=int,
                        metavar='N', help='run number')
    parser.add_argument('--save_dir', default="./", type=str,
                        metavar='N', help='Save dir')
    parser.add_argument('--batch_imgs', default=0, type=int,
                        metavar='N', help='Which batch image to save')
    parser.add_argument('--vis_3d', default=True, type=bool,
                        metavar='N', help='Visualize in 3d')
    parser.add_argument('--linear', default=False, type=bool,
                        metavar='N', help='Do linear validation')

    # ML-Decoder
    parser.add_argument('--batch-size', default=56, type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('--epochs', type=int, default=40,
                        help='number of training epochs for ML-Decoder')
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--use-ml-decoder', default=1, type=int)
    parser.add_argument('--num-of-groups', default=-1, type=int)
    parser.add_argument('--decoder-embedding', default=768, type=int)
    parser.add_argument('--zsl', default=0, type=int)
    parser.add_argument('--threshold_multi', default=0.4, type=float)

    # MultiSupCon
    parser.add_argument('--batch-size_con', default=256, type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('--epochs_con', type=int, default=1000,
                        help='number of training epochs')
    parser.add_argument('--learning_rate_con', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay_con', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--method', type=str, default='MultiSupCon',
                        choices=['MultiSupCon', 'SupCon', 'SimCLR'], help='choose method')
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--feat-dim-con', type=int, default=128,
                        help='feature dimension for contrastive learning')
    parser.add_argument('--c_treshold', type=float, default=0.3,
                        help='Jaccard sim split parameter')
    
    return parser

def main():
    args = parse_option().parse_args()

    # Setup model
    print('creating model {}...'.format(args.model_name))
    model = create_model_base(args).cuda()

    print('done')
    
    wandb.login(key="c77809672cac9c98eb589447ff82854fba590ff7")
    # COCO Data loading
    instances_path_val = os.path.join(args.data, 'annotations/instances_val2014.json')
    instances_path_train = os.path.join(args.data, 'annotations/instances_train2014.json')
    data_path_val = f'{args.data}/val2014'  # args.data
    data_path_train = f'{args.data}/train2014'  # args.data
    train_backbone_dataset = CocoDetection(
        data_path_train,
        instances_path_train,
        TwoCropTransform(transforms.Compose([
                        transforms.Resize((args.image_size, args.image_size)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply([
                            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                        ], p=0.8),
                        transforms.RandomGrayscale(p=0.2),
                        transforms.ToTensor()
                        ]))
    )
    val_backbone_dataset = CocoDetection(
        data_path_val,
        instances_path_val,
        transforms.Compose([
                        transforms.Resize((args.image_size, args.image_size)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor()
                        ])
    )
    val_dataset = CocoDetection(data_path_val,
                                instances_path_val,
                                transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    # normalize, # no need, toTensor does normalization
                                ]))
    train_dataset = CocoDetection(data_path_train,
                                  instances_path_train,
                                  transforms.Compose([
                                      transforms.Resize((args.image_size, args.image_size)),
                                      CutoutPIL(cutout_factor=0.5),
                                      RandAugment(),
                                      transforms.ToTensor(),
                                      # normalize,
                                  ]))
    
    print("Contrastive len(val_dataset)): ", len(val_backbone_dataset))
    print("Contrastive len(train_dataset)): ", len(train_backbone_dataset))
    print("ML_decode len(val_dataset)): ", len(val_dataset))
    print("ML_decode len(train_dataset)): ", len(train_dataset))
    
    experiemnt_dir = os.path.join(args.save_dir, f"experiment_{args.run}")
    if os.path.exists(experiemnt_dir):
        shutil.rmtree(experiemnt_dir)
    os.makedirs(experiemnt_dir, exist_ok=False)
    os.makedirs(os.path.join(experiemnt_dir, "models"), exist_ok=False)

    # Pytorch Data loader 
    train_backbone_loader = torch.utils.data.DataLoader(
        train_backbone_dataset, batch_size=args.batch_size_con, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None)

    val_backbone_loader = torch.utils.data.DataLoader(
        val_backbone_dataset, batch_size=args.batch_size_con, shuffle=None,
        num_workers=args.workers, pin_memory=False, sampler=None)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    # Actuall Training
    print("Training backbone")
    model = add_ml_supcon_head(model, feat_dim=args.feat_dim_con)
    train_multi_label_coco_backbone(model, train_backbone_loader, val_backbone_loader,  args)
    if args.freeze:
        for name, p in model.named_parameters():
            if "body" in name or "global_pool" in name:
                p.requires_grad = False

    print("Training head")
    model = add_ml_decoder_head(model,num_classes=args.num_classes,num_of_groups=args.num_of_groups,
                                decoder_embedding=args.decoder_embedding, zsl=args.zsl)
    train_multi_label_coco(model, train_loader, val_loader,  args)


def train_multi_label_coco_backbone(model, train_loader, val_loader,  args):
    if torch.cuda.is_available():
        model = model.cuda()
    ema = ModelEma(model, 0.9997)  # 0.9997^641=0.82
    
    # set optimizer
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    
    # warm-up for large-batch training,
    if args.batch_size_con > 256:
        args.warm = True
    if args.warm:
        args.warmup_from = 0.01
        args.warm_epochs = 10
        if args.cosine:
            eta_min = args.learning_rate_con * (args.lr_decay_rate ** 3)
            args.warmup_to = eta_min + (args.learning_rate_con - eta_min) * (
                    1 + math.cos(math.pi * args.warm_epochs / args.epochs_con)) / 2
        else:
            args.warmup_to = args.learning_rate_con
    
    criterion = MultiSupConLoss(temperature=args.temp, c_treshold=args.c_treshold)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate_con, momentum=args.momentum, weight_decay=args.weight_decay_con)
    wandb.init(
        project="test-project", 
        entity="pwr-multisupcontr",
        name=f"backbone_training_{args.run}", 
        config={
            "batch-size": args.batch_size_con,
            "epochs": args.epochs_con,
            "learning_rate": args.learning_rate_con,
            "lr_decay_epochs": args.lr_decay_epochs,
            "lr_decay_rate": args.lr_decay_rate,
            "weight_decay": args.weight_decay_con,
            "momentum": args.momentum,
            "method": args.method,
            "temp": args.temp,
            "cosine": args.cosine,
            "warm": args.warm,
            "feat-dim-con": args.feat_dim_con,
            "c_treshold": args.c_treshold
      })
    wandb.watch(model, log="all")
    
    for epoch in range(1, args.epochs_con + 1):
        model.train()
        loss_total = 0
        adjust_learning_rate(args, optimizer, epoch)
        for idx, (images, labels) in enumerate(train_loader):
            images = torch.cat([images[0], images[1]], dim=0)
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            labels = labels.max(dim=1)[0]
            bsz = labels.shape[0]

            # warm-up learning rate
            warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)

            # compute loss
            features = model(images)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            f1 = torch.nn.functional.normalize(f1, dim=1)
            f2 = torch.nn.functional.normalize(f2, dim=1)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            if args.method == 'MultiSupCon':
                loss = criterion(features, labels)
            elif args.method == 'SupCon':
                loss = criterion(features, labels, multi=False)
            elif args.method == 'SimCLR':
                loss = criterion(features)
            else:
                raise ValueError('contrastive method not supported: {}'.
                                format(args.method))

            # SGD
            loss_total += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ema.update(model)

            # print info
            if idx % 100 == 0:
                print(f'Train: Epoch [{epoch-1}/{args.epochs_con}], Step [{idx}/{len(train_loader)}] loss: {loss.item():.3f}')
        try:
            torch.save(model.state_dict(), os.path.join(
                *[args.save_dir, f"experiment_{args.run}", "models", "model-backbone-{}-{}.ckpt".format(epoch + 1, idx + 1)]))
        except:
            pass
    
        fig, fig_ema = validate_contrastive(model, ema, train_loader, args.num_classes, args.vis_3d)
        
        #log
        wandb.log({
            "loss_total": loss_total,
            "loss_average": (loss_total / len(train_loader)),
            "UMAP_vis": fig,
            "UMAP_vis_ema": fig_ema
        })
    
    if args.linear:
        loss_linear, ema = train_contrastive_linear(model, train_loader,  args)
        mAP_regular, mAP_ema = validate_contrastive_linear(model, ema, val_loader,  args)
        wandb.run.summary["loss_linear"] = loss_linear
        wandb.run.summary["mAP_score"] = mAP_regular
        wandb.run.summary["mAP_score_ema"] = mAP_ema
    wandb.finish()

def validate_contrastive(model, ema_model, train_loader, numb_classes, vis_3d=True):
    model.eval()
    outputs = []
    outputs_ema = []
    labels = []
    if vis_3d:
        reducer = umap.UMAP(n_components=3)
    else:
        reducer = umap.UMAP(n_components=2)
    with torch.no_grad():
        for idx, (images, label) in enumerate(train_loader):
            images = images[0]
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)

            # compute loss
            outputs.append(torch.nn.functional.normalize(model(images), dim=1).cpu().detach())
            outputs_ema.append(torch.nn.functional.normalize(ema_model.module(images), dim=1).cpu().detach())
            labels.append(label.max(dim=1)[0].cpu().detach())
    
    embedding = reducer.fit_transform(torch.cat(outputs).numpy())
    embedding_ema = reducer.fit_transform(torch.cat(outputs_ema).numpy())
    labels = torch.cat(labels).numpy()
    
    fig = go.Figure()
    fig_ema = go.Figure()
    scatters = {x:([],[],[]) for x in range(numb_classes)}
    scatters_ema = {x:([],[],[]) for x in range(numb_classes)}
    for x in range(labels.shape[0]):
        for idx, l in enumerate(labels[x]):
            if l == 1:
                scatters[idx][0].append(embedding[x][0])
                scatters[idx][1].append(embedding[x][1])
                scatters_ema[idx][0].append(embedding_ema[x][0])
                scatters_ema[idx][1].append(embedding_ema[x][1])
                
                if vis_3d:
                    scatters[idx][2].append(embedding[x][2])
                    scatters_ema[idx][2].append(embedding_ema[x][2])

    for key in range(numb_classes):
        if vis_3d:
            fig.add_trace(go.Scatter3d(
                x=scatters[key][0], 
                y=scatters[key][1], 
                z=scatters[key][2],
                name=key,
                mode='markers',
                marker_color='rgba(152, 0, 0, .8)'
            ))
            fig_ema.add_trace(go.Scatter3d(
                x=scatters_ema[key][0], 
                y=scatters_ema[key][1], 
                z=scatters_ema[key][2],
                name=key,
                mode='markers',
                marker_color='rgba(152, 0, 0, .8)'
            ))
        else:
            fig.add_trace(go.Scatter(
                x=scatters[key][0], 
                y=scatters[key][1], 
                name=key,
                mode='markers',
                marker_color='rgba(152, 0, 0, .8)'
            ))
            fig_ema.add_trace(go.Scatter(
                x=scatters_ema[key][0], 
                y=scatters_ema[key][1],
                name=key,
                mode='markers',
                marker_color='rgba(152, 0, 0, .8)'
            ))
            
    if vis_3d:
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        fig_ema.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    return fig, fig_ema

def train_contrastive_linear(model, train_loader, args):
    model = add_valid_linear_classification(model, args.num_classes)
    if torch.cuda.is_available():
        model = model.cuda()
    
    ema_model = ModelEma(model, 0.9997)  # 0.9997^641=0.82
    for name, p in model.named_parameters():
        if "body" in name or "global_pool" in name:
            p.requires_grad = False
    
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True) #torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=model.parameters())
    
    for epoch in range(10):
        model.train()
        loss_total = 0
        for idx, (images, labels) in enumerate(train_loader):
            images = images[1]
            images = images.cuda()
            labels = labels.cuda()
            labels = labels.max(dim=1)[0]
            with autocast():  # mixed precision
                output = model(images)  # sigmoid will be done in loss !
            loss = criterion(output.float(), labels)
            loss_total += loss.item()
            model.zero_grad()
            loss.backward()
            optimizer.step()

            ema_model.update(model)
            # print info
            if idx % 100 == 0:
                print(f'Linear Train: Epoch [{epoch}/{args.epochs}], Step [{idx}/{len(train_loader)}], Loss: {loss.item():.1f}')
            
        loss_total /= len(train_loader)
    
    for name, p in model.named_parameters():
        if "body" in name or "global_pool" in name:
            p.requires_grad = True
    return loss_total, ema_model

def validate_contrastive_linear(model, ema_model, val_loader, args):
    model.eval()
    
    Sig = torch.nn.Sigmoid()
    outputs = []
    outputs_ema = []
    labels = []
    with torch.no_grad():
        for idx, (images, label) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)

            # compute loss
            outputs.append(Sig(model(images)).cpu().detach())
            outputs_ema.append(Sig(ema_model.module(images)).cpu().detach())
            labels.append(label.max(dim=1)[0].cpu().detach())
            
    mAP_score_regular = mAP(torch.cat(labels).numpy(), torch.cat(outputs).numpy())
    mAP_score_ema = mAP(torch.cat(labels).numpy(), torch.cat(outputs_ema).numpy())
    print("mAP score regular {:.2f}, mAP score EMA {:.2f}".format(mAP_score_regular, mAP_score_ema))
    return mAP_score_regular, mAP_score_ema

def train_multi_label_coco(model, train_loader, val_loader, args):
    if torch.cuda.is_available():
        model = model.cuda()

    ema = ModelEma(model, 0.9997)  # 0.9997^641=0.82

    # set optimizer
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    parameters = add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=args.learning_rate, weight_decay=0)  # true wd, filter_bias_and_bn
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate, steps_per_epoch=steps_per_epoch, epochs=args.epochs,
                                        pct_start=0.2)
    scaler = GradScaler()

    wandb.init(
        project="test-project", 
        entity="pwr-multisupcontr",
        name=f"ML_Decode_training_{args.run}", 
        config={
            "batch-size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "use-ml-decoder": args.use_ml_decoder,
            "num-of-groups": args.num_of_groups,
            "decoder-embedding": args.decoder_embedding,
            "zsl": args.zsl,
            "threshold_multi": args.threshold_multi,
            "freeze": args.freeze
      })
    wandb.watch(model, log="all")

    highest_mAP = 0
    for epoch in range(args.epochs):
        model.train()
        loss_total = 0
        for idx, (images, labels) in enumerate(train_loader):
            images = images.cuda()
            labels = labels.cuda()
            labels = labels.max(dim=1)[0]
            with autocast():  # mixed precision
                output = model(images).float()  # sigmoid will be done in loss !
            loss = criterion(output, labels)
            loss_total += loss.item()
            model.zero_grad()

            scaler.scale(loss).backward()
            # loss.backward()

            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()

            scheduler.step()

            ema.update(model)
            # print info
            if idx % 100 == 0:
                print(f'Train: Epoch [{epoch}/{args.epochs}], Step [{idx}/{len(train_loader)}], LR {scheduler.get_last_lr()[0]:.1e}, Loss: {loss.item():.1f}')
            
        try:
            torch.save(model.state_dict(), os.path.join(
                *[args.save_dir, f"experiment_{args.run}", "models", "model-{}-{}.ckpt".format(epoch + 1, idx + 1)]))
        except:
            pass

        model.eval()
        mAP_score_regular, mAP_score_ema = validate_multi(val_loader, model, ema, args.num_classes, args.batch_imgs, args.threshold_multi)
        mAP_score = max(mAP_score_regular, mAP_score_ema)
        
        #log
        wandb.log({
            "loss_total": loss_total,
            "loss_average": (loss_total / len(train_loader)),
            "mAP_score": mAP_score_regular,
            "mAP_score_ema": mAP_score_ema
        })
        
        if mAP_score > highest_mAP:
            highest_mAP = mAP_score
            try:
                torch.save(model.state_dict(), os.path.join(
                    *[args.save_dir, f"experiment_{args.run}", "models", "model-highest.ckpt"]))
            except:
                pass
        print('current_mAP = {:.2f}, highest_mAP = {:.2f}\n'.format(mAP_score, highest_mAP))
    
    wandb.run.summary["highest_mAP"] = highest_mAP
    wandb.finish()

def validate_multi(val_loader, model, ema_model, numb_classes, batch_idx=0, threshold_multi=0.4):
    print("starting validation")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    preds_ema = []
    targets = []
    table = None
    for idx, (images, labels) in enumerate(val_loader):
        labels = labels.max(dim=1)[0]
        # compute output
        with torch.no_grad():
            with autocast():
                output_regular = Sig(model(images.cuda())).cpu()
                output_ema = Sig(ema_model.module(images.cuda())).cpu()

        # for mAP calculation
        preds_regular.append(output_regular.cpu().detach())
        preds_ema.append(output_ema.cpu().detach())
        targets.append(labels.cpu().detach())
        
        if idx == batch_idx:
            pass 
            #TODO cannot log tables for multilabel
            #table = wandb.Table(columns=["image", "pred", "preds_ema", "target"])
            #for img, pred, pred_ema, targ in zip(images.to("cpu"), output_regular.to("cpu"), output_ema.to("cpu"), labels.to("cpu")):
            #    pred_index = pred <= threshold_multi
            #    pred_index = pred_index.nonzero().reshape(-1)
            #    pred_ema_index = pred_ema <= threshold_multi
            #    pred_ema_index = pred_ema_index.nonzero().reshape(-1)
            #    targ_index = targ <= threshold_multi
            #    targ_index = targ_index.nonzero().reshape(-1)
            #    table.add_data(wandb.Image(img[0].numpy()*255), pred_index, pred_ema_index, targ_index)
            #wandb.log({f"predictions_table_threshold_{threshold_multi}": table}, commit=False)
    mAP_score_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
    mAP_score_ema = mAP(torch.cat(targets).numpy(), torch.cat(preds_ema).numpy())
    print("mAP score regular {:.2f}, mAP score EMA {:.2f}".format(mAP_score_regular, mAP_score_ema))
    
    AUC_score_regular = roc_auc_score(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy(), average=None)
    AUC_score_ema = roc_auc_score(torch.cat(targets).numpy(), torch.cat(preds_ema).numpy(), average=None)
    
    AUC_table = wandb.Table(columns=[x for x in range(numb_classes)])
    AUC_table.add_data(*AUC_score_regular.tolist())
    AUC_table.add_data(*AUC_score_ema.tolist())
    wandb.log({"AUC_table": AUC_table}, commit=False)
    
    return mAP_score_regular, mAP_score_ema


if __name__ == '__main__':
    main()
