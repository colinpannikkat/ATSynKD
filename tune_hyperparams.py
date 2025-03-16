import torch
from tqdm import tqdm
from utils import Datasets, Schedulers
from losses import KLAttentionAwareKDLoss, SSIMAttentionAwareNMLoss, EuclidAttentionAwareKDLoss, KDLoss
from models import load_resnet20, load_resnet32
from argparse import ArgumentParser
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import json

import torch.nn as nn
import torch.nn.functional as F

seed = 42
torch.manual_seed(seed)

def evaluate(models, data, criterion, device, kd=False):
    loss = 0
    accuracy = 0
    with torch.no_grad():
        for inputs, labels in tqdm(data):
            inputs = inputs.to(device)
            labels = labels.to(device)
            out = [model(inputs) for model in models]
            if kd:
                teacher_layers, teacher_out = out[0]
                student_layers, student_out = out[1]
                loss += criterion(teacher_layers, teacher_out, student_layers, student_out).item()
                _, s_predicted = torch.max(F.softmax(student_out, dim=1), 1)
                _, t_predicted = torch.max(F.softmax(teacher_out, dim=1), 1)
                correct = (s_predicted == t_predicted).sum().item()
                accuracy += correct / labels.size(0)
            else:
                _, output = out[0]
                loss += criterion(output, labels).item()
                _, predicted = torch.max(F.softmax(output, dim=1), 1)
                correct = (predicted == labels).sum().item()
                accuracy += correct / labels.size(0)
    return loss / len(data), accuracy / len(data)

def train(config, checkpoint_dir=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = []
    if config["kd"]:
        teacher_weights = torch.load(config["weights"], map_location=device)
        teacher = load_resnet32(config["dataset"], weights=teacher_weights)
        student = load_resnet20(config["dataset"])
        teacher.to(device)
        student.to(device)
        models.append(teacher)
        models.append(student)
        if config["klat"]:
            criterion = KLAttentionAwareKDLoss(llambda=config["llambda"], 
                       alpha=config["alpha"], 
                       factor=config["factor"],
                       total_epochs=config["epochs"])
        elif config["euclidat"]:
            criterion = EuclidAttentionAwareKDLoss(llambda=config["llambda"])
        elif config["ssimat"]:
            criterion = SSIMAttentionAwareNMLoss(llambda=config["llambda"],
                     alpha=config["alpha"],
                     factor=config["factor"],
                     total_epochs=config["epochs"])
        else:
            criterion = KDLoss(alpha=config["alpha"])
    else:
        model = load_resnet32(config["dataset"]) if config["big"] else load_resnet20(config["dataset"])
        model = model.to(device)
        models.append(model)
        criterion = nn.CrossEntropyLoss()

    params = models[1].parameters() if config["kd"] else models[0].parameters()
    if config["sgd"]:
        optimizer = torch.optim.SGD(params, lr=config["lr"], weight_decay=config["weight_decay"], momentum=config["momentum"])
    else:
        optimizer = torch.optim.AdamW(params, lr=config["lr"], weight_decay=config["weight_decay"], eps=config["eps"])

    data = Datasets(seed=seed)
    trainset, testset = data.load(dataset=config["dataset"], n=config["n"], batch_size=config["batch"], augment=config["augment"], synth=config["synth"])
    config['lr_args']['len_train_loader'] = len(trainset)

    scheduler = None
    if config["scheduler"] or config["reducer"]:
        sched = Schedulers(optimizer, warmup=config["warmup"], reducer=config["reducer"])
        scheduler = sched.load(config["scheduler"], total_epochs=config["epochs"], **config['lr_args'])
    
    if isinstance(scheduler, tuple):
        scheduler, reduce_scheduler = scheduler
    else:
        reduce_scheduler = None

    for epoch in range(config["epochs"]):
        running_loss = 0.0
        running_acc = 0.0
        for i, data in enumerate(tqdm(trainset)):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            out = [model(inputs) for model in models]
            if config["kd"]:
                teacher_layers, teacher_out = out[0]
                student_layers, student_out = out[1]
                loss = criterion(teacher_layers, teacher_out, student_layers, student_out)
                _, s_predicted = torch.max(F.softmax(student_out, dim=1), 1)
                _, t_predicted = torch.max(F.softmax(teacher_out, dim=1), 1)
                correct = (s_predicted == t_predicted).sum().item()
                accuracy = correct / labels.size(0)
            else:
                _, output = out[0]
                loss = criterion(output, labels)
                _, predicted = torch.max(F.softmax(output, dim=1), 1)
                correct = (predicted == labels).sum().item()
                accuracy = correct / labels.size(0)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_acc += accuracy

        val_loss, val_acc = evaluate(models, testset, criterion, device, kd=config["kd"])
        tune.report({"loss" : val_loss, "accuracy": val_acc, "epoch": epoch})

        if scheduler is not None:
            scheduler.step()

        if reduce_scheduler is not None:
            reduce_scheduler.step(val_loss)

def main():
    parser = ArgumentParser()
    parser.add_argument("-dataset", choices=['cifar10', 'cifar100', 'tiny-imagenet'], required=True)
    parser.add_argument("-n", default=-1, type=int)
    parser.add_argument("-kd", action='store_true', help="Train with knowledge distillation")
    parser.add_argument("-klat", action='store_true', help="Train with attention aware distillation")
    parser.add_argument("-euclidat", action='store_true', help="Train with euclidean attention aware distillation")
    parser.add_argument("-ssimat", action='store_true', help="Train with distance metric derived from the SSIM for computing attention distances.")
    parser.add_argument("-weights", help="Weights to use for teacher model with KD")
    parser.add_argument("-small", action='store_true', help="Train student model")
    parser.add_argument("-big", action='store_true', help="Train teacher model")
    parser.add_argument("-batch", default=128, type=int) # need to add different batch sizes for train and test
    parser.add_argument("-lr", default=1e-2, type=float)
    parser.add_argument("-weight_decay", default=1e-2, type=float, help="AdamW/SGD weight decay")
    parser.add_argument("-momentum", default=0.9, type=float, help="Momentum for SGD optimizer")
    parser.add_argument("-sgd", action="store_true", help="Use SGD optimizer instead of AdamW.")
    parser.add_argument("-eps", default=1e-8, type=float, help="AdamW epsilon hyperpam")
    parser.add_argument("-epochs", default=30, type=int)
    parser.add_argument("-llambda", default=0.99, type=float, help="Tradeoff term between AT and KD Loss terms. Larger places more weight on CE. See losses.py.")
    parser.add_argument("-factor", default=None)
    parser.add_argument("-alpha", default=0.9, type=float, help="Tradeoff term between CE and KL terms in KD loss.")
    parser.add_argument("-scheduler", choices=['constant+multistep', 'lineardecay', 'constant', 'linear', 'multistep', 'onecycle'], default=None, type=str)
    parser.add_argument("-warmup", action='store_true', help="Enable warmup")
    parser.add_argument("-reducer", action='store_true', help="Enable learning rate reducer on plateau")
    parser.add_argument("-synth", default=None, type=int, help="Desired number of synthetic images, will generate M synthetic images")
    parser.add_argument("-augment", action='store_true', help="Apply AutoAugment when building dataset")
    parser.add_argument("-lr_args", help="Pass in as JSON string ex: '{'start_factor':0.5, 'warmup_period':5}'. See utils.py for more information on the arguments that can be passed in.", default="{}", type=str)
    parser.add_argument("-name", help="Designate name of training for out file", default=None)
    args = parser.parse_args()

    lr_args = json.loads(args.lr_args)
    lr_args['total_epochs'] = args.epochs

    config = {
        "dataset": args.dataset,
        "n": args.n,
        "kd": args.kd,
        "klat": args.klat,
        "euclidat": args.euclidat,
        "ssimat": args.ssimat,
        "weights": args.weights,
        "small": args.small,
        "big": args.big,
        "batch": args.batch,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "momentum" : args.momentum,
        "sgd": args.sgd,
        "eps": args.eps,
        "epochs": args.epochs,
        "llambda": tune.uniform(0.00, 1.),
        "factor" : args.factor,
        "alpha" : tune.uniform(0.00, 1.),
        "scheduler": args.scheduler,
        "warmup": args.warmup,
        "reducer": args.reducer,
        "synth": args.synth,
        "augment": args.augment,
        "lr_args": lr_args
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=args.epochs,
        grace_period=1,
        reduction_factor=2
    )

    reporter = tune.CLIReporter(
        metric_columns=["loss", "accuracy", "epoch"]
    )

    analysis = tune.run(
        train,
        resources_per_trial={"cpu": 2, "gpu": 1},
        config=config,
        num_samples=10,
        scheduler=scheduler,
        progress_reporter=reporter
    )

    print("Best hyperparameters found were: ", analysis.get_best_config(metric="loss", mode="min"))

if __name__ == "__main__":
    main()