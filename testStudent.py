from models import load_resnet20
from utils import Datasets
import torch
from tqdm import tqdm
from argparse import ArgumentParser

def testModel(model_path, dataset):
    if dataset == "cifar100":
        trainloader, testloader = Datasets().load_cifar100(batch_size=256)
    elif dataset == "tiny-imagenet":
        trainloader, testloader = Datasets().load_tinyimagenet(batch_size=256)

    student = load_resnet20(dataset, weights=torch.load(model_path))
    student = student.to('cuda')
    student.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(testloader):
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = student(images)[0]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: {:.2f} %'.format(100 * correct / total))

def main():
    parser = ArgumentParser()
    parser.add_argument('-model_path', type=str, required=True, help='Path to the model file')
    parser.add_argument('-dataset', type=str, required=True, help='Name of the dataset')
    args = parser.parse_args()

    testModel(args.model_path, args.dataset)

if __name__ == "__main__":
    main()
