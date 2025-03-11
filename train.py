import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import Datasets, Schedulers, DataLoader, plot_metrics, save_metrics, save_parameters, save_checkpoint, load_checkpoint
from losses import AttentionAwareKDLoss
from models import load_resnet20, load_resnet32
from argparse import ArgumentParser
import json
import datetime

seed = 42
torch.manual_seed(seed)

# Given a model, X/Y dataset, and batch size, return the average cross-entropy loss and accuracy over the set
def evaluate(models, data, criterion, device, kd=False):
    loss = 0
    accuracy = 0
    with torch.no_grad():
        for inputs, labels in tqdm(data):

            # Move inputs and labels to gpu
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            out = [model(inputs) for model in models]
            if kd:
                teacher_layers, teacher_out = out[0]
                student_layers, student_out = out[1]
                loss += criterion(teacher_layers, teacher_out, student_layers, student_out).item()

                # Compute accuracy for student and teacher model pred
                _, s_predicted = torch.max(F.softmax(student_out, dim=1), 1)
                _, t_predicted = torch.max(F.softmax(teacher_out, dim=1), 1)
                correct = (s_predicted == t_predicted).sum().item()
                accuracy += correct / labels.size(0)
            else:
                _, output = out[0]
                loss += criterion(output, labels).item()

                # Compute accuracy
                _, predicted = torch.max(F.softmax(output, dim=1), 1)
                correct = (predicted == labels).sum().item()
                accuracy += correct / labels.size(0)

    return loss / len(data), accuracy / len(data)

def train(models: list[nn.Module], train_dataloader: DataLoader, test_dataloader: DataLoader, 
          optimizer, criterion, device, kd=False, scheduler=None, num_epochs=30, prefix=".",
          checkpoint=None):
    
    best_val_loss = (10e12, 0, 0) # storing val_loss, acc, and epoch number
    # Some lists for book-keeping for plotting later
    losses = []
    val_losses = []
    accs = []
    val_accs = []
    lrs = []
    best_model = None

    if len(models) > 1 and kd == False:
        raise(Exception("Cannot run normal training with more than one model."))
    if len(models) == 1 and kd == True:
        raise(Exception("Cannot do knowledge distllation with one model."))
    
    models = [model.train() for model in models]
    
    reduce_scheduler = None
    if type(scheduler) is tuple:
        scheduler, reduce_scheduler = scheduler

    for epoch in range(num_epochs):

        running_loss = 0.0
        running_acc = 0.0
        for i, data in enumerate(tqdm(train_dataloader)):

            inputs, labels = data

            # Move inputs and labels to gpu
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            out = [model(inputs) for model in models]
            if kd:
                teacher_layers, teacher_out = out[0]
                student_layers, student_out = out[1]
                loss = criterion(teacher_layers, teacher_out, student_layers, student_out)

                # Compute accuracy for student and teacher model pred
                _, s_predicted = torch.max(F.softmax(student_out, dim=1), 1)
                _, t_predicted = torch.max(F.softmax(teacher_out, dim=1), 1)
                correct = (s_predicted == t_predicted).sum().item()
                accuracy = correct / labels.size(0)
            else:
                _, output = out[0]
                loss = criterion(output, labels)

                # Compute accuracy
                _, predicted = torch.max(F.softmax(output, dim=1), 1)
                correct = (predicted == labels).sum().item()
                accuracy = correct / labels.size(0)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            running_acc += accuracy

        val_loss, val_acc = evaluate(models, test_dataloader, criterion, device, kd=kd)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/len(train_dataloader)}, Train Accuracy: {running_acc/len(train_dataloader)}, Val Loss: {val_loss}, Val Accuracy: {val_acc}")

        # Learning rate (and warmup) step
        if scheduler is not None:
            scheduler.step()

        if reduce_scheduler is not None:
            reduce_scheduler.step(val_loss)
                
        losses.append(running_loss/len(train_dataloader))
        accs.append(running_acc/len(train_dataloader))
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        lrs.append(optimizer.param_groups[0]['lr'])

        save_metrics(running_loss/len(train_dataloader), running_acc/len(train_dataloader), val_loss, val_acc, optimizer.param_groups[0]['lr'], prefix, epoch)

        if kd:

            if val_loss < best_val_loss[0]:
                best_model = models[1].state_dict()
                best_val_loss = (val_loss, val_acc, epoch)
            save_checkpoint(models[1], optimizer, (scheduler, reduce_scheduler), epoch, filepath=f"{prefix}/models")

        else:

            if val_loss < best_val_loss[0]:
                best_model = models[0].state_dict()
                best_val_loss = (val_loss, val_acc, epoch)
            save_checkpoint(models[0], optimizer, (scheduler, reduce_scheduler), epoch, filepath=f"{prefix}/models")

    best_epoch = best_val_loss[2]
    save_metrics(losses[best_epoch], accs[best_epoch], val_losses[best_epoch], val_accs[best_epoch], lrs[best_epoch], prefix, best_epoch, best=True)

    print(f"Lowest loss of {best_val_loss[0]} found in epoch {best_epoch+1} with accuracy {best_val_loss[1]}.")

    return losses, accs, val_losses, val_accs, lrs, best_model

def main():

    parser = ArgumentParser()
    parser.add_argument("-dataset", choices=['cifar10', 'cifar100', 'tiny-imagenet'], required=True)
    parser.add_argument("-n", default=-1, type=int)
    parser.add_argument("-kd", action='store_true', help="Train with knowledge distillation")
    parser.add_argument("-weights", help="Weights to use for teacher model with KD")
    parser.add_argument("-small", action='store_true', help="Train student model")
    parser.add_argument("-big", action='store_true', help="Train teacher model")
    parser.add_argument("-batch", default=128, type=int) # need to add different batch sizes for train and test
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
    parser.add_argument("-lr_args", help="Pass in as JSON string ex: '{'start_factor':0.5, 'warmup_period':5}'. See utils.py for more information on the arguments that can be passed in.", default="{}", type=str)

    args = parser.parse_args()

    lr_args = json.loads(args.lr_args)
    lr_args['total_epochs'] = args.epochs

    hparams = {
        "lr" : args.lr,
        "batch_size" : args.batch,
        "num_epochs" : args.epochs
    }

    # Standard loss
    criterion = nn.CrossEntropyLoss()

    # Initialize the model, loss function, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    models = []
    if args.kd:
        teacher_weights = torch.load(args.weights, map_location=device)
        teacher = load_resnet32(args.dataset, weights=teacher_weights)
        student = load_resnet20(args.dataset)
        teacher.to(device)
        student.to(device)
        models.append(teacher)
        models.append(student)
        criterion = AttentionAwareKDLoss(llambda=args.llambda)
    else:
        model = None
        if args.big:
            model = load_resnet32(args.dataset)
        if args.small:
            model = load_resnet20(args.dataset)
        if model is None:
            raise(Exception("No model specified"))
        model = model.to(device)
        models.append(model)

    params = models[0].parameters()
    if args.kd:
        params = models[1].parameters()

    optimizer = torch.optim.AdamW(params, lr=hparams['lr'], weight_decay=args.weight_decay, eps=args.eps)

    # Get Data
    data = Datasets(seed=seed)
    trainset, testset = data.load(dataset=args.dataset, n=args.n, batch_size=hparams['batch_size'], augment=args.augment, synth=args.synth)
    lr_args['len_train_loader'] = len(trainset)

    # Define scheduler
    scheduler = None
    if (args.scheduler):
        sched = Schedulers(optimizer, warmup=args.warmup, reducer=args.reducer)
        scheduler = sched.load(args.scheduler, **lr_args)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{args.dataset}_{'kd' if args.kd else 'reg'}_{timestamp}"

    save_parameters(args, hparams, prefix)

    # Run training loop
    train_losses, train_accs, val_losses, val_accs, lrs, best_model = train(models, 
                                                                       trainset, 
                                                                       testset, 
                                                                       optimizer, 
                                                                       criterion, 
                                                                       device, 
                                                                       num_epochs=hparams['num_epochs'], 
                                                                       kd=args.kd, 
                                                                       scheduler=scheduler,
                                                                       prefix=prefix)

    plot_metrics(train_accs, train_losses, val_accs, val_losses, out=f"{prefix}/metrics.png")
    if args.kd:
        torch.save(best_model, f"{args.dataset}_student_model_{timestamp}.pt")
        l, a = evaluate([models[1]], testset, nn.CrossEntropyLoss(), device, kd=False)
        print(f"Student model test: Loss: {l}, Accuracy: {a}")
        l, a = evaluate([models[0]], testset, nn.CrossEntropyLoss(), device, kd=False)
        print(f"Teacher model test: Loss: {l}, Accuracy: {a}")
    else:
        torch.save(best_model, f"{prefix}/{args.dataset}_{"big" if args.big else "small"}_teacher_model_{timestamp}.pt")

if __name__ == "__main__":
    main()