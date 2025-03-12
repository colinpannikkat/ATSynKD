import torch
from tqdm import tqdm
from utils import Datasets, Schedulers
from losses import AttentionAwareKDLoss
from models import load_resnet20, load_resnet32
from argparse import ArgumentParser
from ray import tune
from ray.tune.schedulers import ASHAScheduler

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
        criterion = AttentionAwareKDLoss(llambda=config["llambda"])
    else:
        model = load_resnet32(config["dataset"]) if config["big"] else load_resnet20(config["dataset"])
        model = model.to(device)
        models.append(model)
        criterion = nn.CrossEntropyLoss()

    params = models[1].parameters() if config["kd"] else models[0].parameters()
    optimizer = torch.optim.AdamW(params, lr=config["lr"], weight_decay=config["weight_decay"], eps=config["eps"])

    data = Datasets(seed=seed)
    trainset, testset = data.load(dataset=config["dataset"], n=config["n"], batch_size=config["batch"], augment=config["augment"], synth=config["synth"])

    scheduler = None
    if config["scheduler"] or config["reducer"]:
        sched = Schedulers(optimizer, warmup=config["warmup"], reducer=config["reducer"])
        scheduler = sched.load(config["scheduler"], total_epochs=config["epochs"], len_train_loader=len(trainset))
    
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
    parser.add_argument("-weights", help="Weights to use for teacher model with KD")
    parser.add_argument("-small", action='store_true', help="Train student model")
    parser.add_argument("-big", action='store_true', help="Train teacher model")
    parser.add_argument("-batch", default=128, type=int)
    parser.add_argument("-lr", default=1e-2, type=float)
    parser.add_argument("-weight_decay", default=1e-2, type=float, help="AdamW weight decay")
    parser.add_argument("-eps", default=1e-8, type=float, help="AdamW epsilon hyperpam")
    parser.add_argument("-epochs", default=30, type=int)
    parser.add_argument("-llambda", default=0.1, type=float, help="Tradeoff term between CE and AT Loss")
    parser.add_argument("-scheduler", choices=['constant+multistep', 'lineardecay', 'constant', 'linear', 'multistep', 'onecycle'], default=None, type=str)
    parser.add_argument("-warmup", action='store_true', help="Enable warmup")
    parser.add_argument("-reducer", action='store_true', help="Enable learning rate reducer on plateau")
    parser.add_argument("-synth", default=None, type=int, help="Desired dataset size, will generate M - N synthetic images")
    parser.add_argument("-augment", action='store_true', help="Apply AutoAugment when building dataset")
    parser.add_argument("-lr_args", help="Pass in as JSON string ex: '{\"start_factor\":0.5, \"warmup_period\":5}'. See utils.py for more information on the arguments that can be passed in.", default="{}", type=str)
    args = parser.parse_args()

    def generate_milestones(config):
        return [int(config['epochs'] * 0.3), int(config['epochs'] * 0.6), int(config['epochs'] * 0.9)]

    config = {
        "dataset": args.dataset,
        "n": args.n,
        "kd": args.kd,
        "weights": args.weights,
        "small": args.small,
        "big": args.big,
        "batch": tune.choice([64, 128, 256]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "weight_decay": tune.loguniform(1e-4, 1e-2),
        "eps": tune.loguniform(1e-8, 1e-6),
        "epochs": tune.choice([10, 20, 30, 40, 50]),
        "llambda": tune.uniform(0.01, 0.1),
        "scheduler": tune.choice(['constant+multistep', 'lineardecay', 'constant', 'linear', 'multistep', 'onecycle']),
        "warmup": args.warmup,
        "reducer": args.reducer,
        "synth": args.synth,
        "augment": args.augment,
        "lr_args": {
            "start_factor": tune.uniform(0.1, 0.5),
            "end_factor": tune.uniform(0.5, 1.0),
            "total_iters": tune.choice([5, 10, 20]),
            "warmup_period": tune.choice([5, 10, 15]),
            "constant_epochs": tune.choice([5, 10]),
            "factor": tune.uniform(0.1, 0.5),
            "milestones": tune.sample_from(generate_milestones),
            "gamma": tune.uniform(0.1, 0.5),
            "max_lr": tune.loguniform(1e-4, 1e-2),
            "steps_per_epoch": tune.choice([100, 200, 300]),
            "pct_start": tune.uniform(0.1, 0.3)
        }
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