import os
import random
import argparse
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms

import time
import copy
import numpy as np

from Resnet18 import resnet18
from Vgg13 import vgg
from Mobilenetv2_quant import mobilenet_v2


class QuantizedModel(nn.Module):
    def __init__(self, model_fp32):
        super(QuantizedModel, self).__init__()
        # QuantStub converts tensors from floating point to quantized. This will only be used for inputs.
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point. This will only be used for outputs.
        self.dequant = torch.quantization.DeQuantStub()
        self.model_fp32 = model_fp32

    def forward(self, x):
        # manually specify where tensors will be converted from floating point to quantized in the quantized model
        x = self.quant(x)
        x = self.model_fp32(x)
        # manually specify where tensors will be converted from quantized to floating point in the quantized model
        x = self.dequant(x)
        return x

#######################################################################


def logger(msg):
    print(datetime.datetime.now(), '-->', msg)


def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


def create_model(model_name, num_classes=10):
    if model_name == 'resnet18':
        model = resnet18(num_classes=num_classes, pretrained=False)
    elif model_name == 'vgg13':
        model = vgg(num_classes=num_classes)
    elif model_name == 'mobilenetv2':
        model = mobilenet_v2(num_classes=num_classes)
    return model


def prepare_dataloader(num_workers, train_batch_size, eval_batch_size):
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                               std=(0.229, 0.224, 0.225)), ])

    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                              std=(0.229, 0.224, 0.225)), ])

    train_set = torchvision.datasets.CIFAR10(root="data",
                                             train=True,
                                             download=True,
                                             transform=train_transform)

    test_set = torchvision.datasets.CIFAR10(root="data",
                                            train=False,
                                            download=True,
                                            transform=test_transform)

    train_sampler = torch.utils.data.RandomSampler(train_set)
    test_sampler = torch.utils.data.SequentialSampler(test_set)

    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=train_batch_size,
                                               sampler=train_sampler,
                                               num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=eval_batch_size,
                                              sampler=test_sampler,
                                              num_workers=num_workers)

    return train_loader, test_loader


def evaluate_model(model, test_loader, device, criterion=None):
    model.eval()
    model.to(device)

    running_loss = 0
    running_corrects = 0

    for inputs, labels in test_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        if criterion is not None:
            loss = criterion(outputs, labels).item()
        else:
            loss = 0

        # statistics
        running_loss += loss * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    eval_loss = running_loss / len(test_loader.dataset)
    eval_accuracy = running_corrects / len(test_loader.dataset)

    return eval_loss, eval_accuracy


def train_model(model,
                train_loader,
                test_loader,
                device,
                learning_rate=1e-1,
                num_epochs=1):
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    # It seems that SGD optimizer is better than Adam optimizer for ResNet18 training on CIFAR10.
    optimizer = optim.SGD(model.parameters(),
                          lr=learning_rate,
                          momentum=0.9,
                          weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[100, 150],
                                                     gamma=0.1,
                                                     last_epoch=-1)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    # Evaluation
    model.eval()
    eval_loss, eval_accuracy = evaluate_model(model=model,
                                              test_loader=test_loader,
                                              device=device,
                                              criterion=criterion)
    logger("Epoch: {:02d} | Eval Loss: {:.3f} | Eval Acc: {:.3f}".format(-1, eval_loss, eval_accuracy))

    for epoch in range(num_epochs):
        # Training
        model.train()

        running_loss = 0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = running_corrects / len(train_loader.dataset)

        # Evaluation
        model.eval()
        eval_loss, eval_accuracy = evaluate_model(model=model,
                                                  test_loader=test_loader,
                                                  device=device,
                                                  criterion=criterion)

        # Set learning rate scheduler
        scheduler.step()

        logger("Epoch: {:03d} | Train Loss: {:.3f} | Train Acc: {:.3f} | Eval Loss: {:.3f} | Eval Acc: {:.3f}"
               .format(epoch, train_loss, train_accuracy, eval_loss, eval_accuracy))

    return model


def model_equivalence(model_1,
                      model_2,
                      device,
                      rtol=1e-05,
                      atol=1e-08,
                      num_tests=100,
                      input_size=(1, 3, 32, 32)):
    model_1.to(device)
    model_2.to(device)

    for _ in range(num_tests):
        x = torch.rand(size=input_size).to(device)
        y1 = model_1(x).detach().cpu().numpy()
        y2 = model_2(x).detach().cpu().numpy()
        if not np.allclose(a=y1, b=y2, rtol=rtol, atol=atol, equal_nan=False):
            print("Model equivalence test sample failed: ")
            print(y1)
            print(y2)
            return False

    return True


def calibrate_model(model, loader, device=torch.device("cpu:0")):
    model.to(device)
    model.eval()

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        _ = model(inputs)


def measure_inference_latency(model,
                              device,
                              input_size=(1, 3, 32, 32),
                              num_samples=100,
                              num_warmups=10):
    model.to(device)
    model.eval()

    x = torch.rand(size=input_size).to(device)

    with torch.no_grad():
        for _ in range(num_warmups):
            _ = model(x)
    # torch.cpu.synchronize()

    with torch.no_grad():
        start_time = time.time()
        for _ in range(num_samples):
            _ = model(x)
            # torch.cpu.synchronize()
        end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_ave = elapsed_time / num_samples

    return elapsed_time_ave


def save_model(model, model_dir, model_filename):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.save(model.state_dict(), model_filepath)


def load_model(model, model_filepath, device):
    model.load_state_dict(torch.load(model_filepath, map_location='cpu'))
    return model


def save_torchscript_model(model, model_dir, model_filename):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.jit.save(torch.jit.script(model), model_filepath)


def load_torchscript_model(model_filepath, device):
    model = torch.jit.load(model_filepath, map_location='cpu')
    return model


def print_model_size(model):
    torch.save(model.state_dict(), "tmp.pt")
    size = os.path.getsize("tmp.pt")/1e6
    os.remove('tmp.pt')
    return size

##############################################################################


def main(args):
    # initializing variables
    random_seed = 0
    num_classes = 10
    # cuda_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu:0")

    model_name = args.model
    logger(f'Performing quantization for {model_name}')
    model_dir = "saved_models"
    model_filename = f"{model_name}_cifar10.pt"
    quantized_model_filename = f"{model_name}_quantized_cifar10.pt"
    model_filepath = os.path.join(model_dir, model_filename)
    quantized_model_filepath = os.path.join(model_dir, quantized_model_filename)

    # setting seed value
    set_random_seeds(random_seed=random_seed)
    logger(f'Random seed set to {random_seed}')

    # Create an untrained model.
    model = create_model(model_name=model_name, num_classes=num_classes)
    logger(f'Untrained model created')

    # preparing train and test datasets
    num_workers = args.num_workers
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
    train_loader, test_loader = prepare_dataloader(num_workers=num_workers,
                                                   train_batch_size=train_batch_size,
                                                   eval_batch_size=eval_batch_size)
    logger('Dataloader prepared')

    pretrained = bool(args.pretrained)
    if not pretrained:
        # train model
        num_epochs = args.num_epochs
        model = train_model(model=model,
                            train_loader=train_loader,
                            test_loader=test_loader,
                            device=cpu_device,
                            learning_rate=1e-1,
                            num_epochs=num_epochs)
        logger(f'Model training completed with {num_epochs} epochs')

        # Save model
        save_model(model=model, model_dir=model_dir, model_filename=model_filename)

    # load pretrained model
    model = load_model(model=model,
                       model_filepath=model_filepath,
                       device=cpu_device)
    logger(f'Loaded pretrained model from {model_filepath}')

    # Move the model to CPU since static quantization does not support CUDA currently
    model.to(cpu_device)
    # Make a copy of the model for layer fusion
    fused_model = copy.deepcopy(model)
    model.eval()
    # The model has to be switched to evaluation mode before fusion, else quantization will not work correctly
    fused_model.eval()

    # no need of layer fusion for vgg13 model
    if model_name == 'resnet18':
        # Fuse the model in place
        fused_model = torch.quantization.fuse_modules(fused_model, [["conv1", "bn1", "relu"]], inplace=True)
        for module_name, module in fused_model.named_children():
            if "layer" in module_name:
                for basic_block_name, basic_block in module.named_children():
                    torch.quantization.fuse_modules(
                        basic_block, [["conv1", "bn1", "relu1"], ["conv2", "bn2"]], inplace=True)
                    for sub_block_name, sub_block in basic_block.named_children():
                        if sub_block_name == "downsample":
                            torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)
    elif model_name == 'mobilenetv2':
        fused_model.fuse_model()

    # Print FP32 model
    logger(f'FP32 model \n {model}')
    # Print fused model
    logger(f'Layer fused model \n {fused_model}')

    # Model and fused model should be equivalent.
    assert model_equivalence(
        model_1=model,
        model_2=fused_model,
        device=cpu_device,
        rtol=1e-03,
        atol=1e-06,
        num_tests=100,
        input_size=(
            1, 3, 32,
            32)), "Fused model is not equivalent to the original model!"

    # Prepare the model for static quantization. This inserts observers in
    # the model that will observe activation tensors during calibration.
    quantized_model = QuantizedModel(model_fp32=fused_model)

    quantization_config = torch.quantization.get_default_qconfig("qnnpack")
    quantized_model.qconfig = quantization_config
    # Print quantization configurations
    logger(f'Quantized model qconfig \n {quantized_model.qconfig}')

    # prepare for quantization
    torch.quantization.prepare(quantized_model, inplace=True)

    # Use training data for calibration.
    calibrate_model(model=quantized_model,
                    loader=train_loader,
                    device=cpu_device)

    quantized_model = torch.quantization.convert(quantized_model, inplace=True)

    # Using high-level static quantization wrapper
    # The above steps prepare, calibrate_model, and convert are also equivalent to
    # quantized_model = torch.quantization.quantize(model=quantized_model, run_fn=calibrate_model, run_args=[train_loader], mapping=None, inplace=False)

    quantized_model.eval()

    # Print quantized model
    logger(f'Quantized model \n {quantized_model}')

    # Save quantized model
    save_torchscript_model(model=quantized_model,
                           model_dir=model_dir,
                           model_filename=quantized_model_filename)

    # Load quantized model
    quantized_jit_model = load_torchscript_model(model_filepath=quantized_model_filepath, device=cpu_device)

    _, fp32_eval_accuracy = evaluate_model(model=model,
                                           test_loader=test_loader,
                                           device=cpu_device,
                                           criterion=None)
    _, int8_eval_accuracy = evaluate_model(model=quantized_jit_model,
                                           test_loader=test_loader,
                                           device=cpu_device,
                                           criterion=None)

    logger("FP32 evaluation accuracy: {:.3f}%".format(fp32_eval_accuracy * 100))
    logger("INT8 evaluation accuracy: {:.3f}%".format(int8_eval_accuracy * 100))

    fp32_cpu_inference_latency = measure_inference_latency(model=model,
                                                           device=cpu_device,
                                                           input_size=(1, 3, 32, 32),
                                                           num_samples=100)
    int8_cpu_inference_latency = measure_inference_latency(model=quantized_model,
                                                           device=cpu_device,
                                                           input_size=(1, 3, 32, 32),
                                                           num_samples=100)
    int8_jit_cpu_inference_latency = measure_inference_latency(model=quantized_jit_model,
                                                               device=cpu_device,
                                                               input_size=(1, 3, 32, 32),
                                                               num_samples=100)
    """
    fp32_gpu_inference_latency = measure_inference_latency(model=model,
                                                           device=cpu_device,
                                                           input_size=(1, 3,
                                                                       32, 32),
                                                           num_samples=100)
    """

    logger("FP32 CPU Inference Latency: {:.2f} ms / sample".format(fp32_cpu_inference_latency * 1000))
    # print("FP32 CUDA Inference Latency: {:.2f} ms / sample".format(fp32_gpu_inference_latency * 1000))
    logger("INT8 CPU Inference Latency: {:.2f} ms / sample".format(int8_cpu_inference_latency * 1000))
    logger("INT8 JIT CPU Inference Latency: {:.2f} ms / sample".format(int8_jit_cpu_inference_latency * 1000))

    logger("FP32 model size: %.2f MB" % print_model_size(model))
    logger("INT8 model size: %.2f MB" % print_model_size(quantized_model))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', type=int, default=0, choices=[0, 1])
    parser.add_argument('--model', type=str, choices=['resnet18', 'vgg13', 'mobilenetv2'])
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--eval_batch_size', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=5)
    arguments = parser.parse_args()

    main(args=arguments)
