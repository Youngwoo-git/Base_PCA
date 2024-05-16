import torch
import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub

def prepare_quantization(model, primary_layers):

    fuse_modules = torch.ao.quantization.fuse_modules_qat

    for l, _ in primary_layers:
        activation_idx = int(l.split(".")[-1])
        conv_idx = activation_idx-1
        fuse_modules(model, [l.replace(str(activation_idx), str(conv_idx)), l], inplace=True)

        model.features[conv_idx] = nn.Sequential(QuantStub(), model.features[conv_idx], DeQuantStub())
        model.features[conv_idx].qconfig = torch.ao.quantization.default_qconfig

    torch.ao.quantization.prepare(model, inplace=True)


def calibrate(model, data_loader, num_calibration_batches = 1):
    model.eval()
    calibration_count = min(num_calibration_batches, len(data_loader))
    # print(calibration_count)

    with torch.no_grad():
        for idx, (image, target) in enumerate(data_loader):
            if idx == calibration_count:
                break
            model(image)

# if num_calibration_batches, exclude calibration 
def quantize_model(model, primary_layers, data_loader, num_calibration_batches):
    prepare_quantization(model, primary_layers)
    calibrate(model, data_loader, num_calibration_batches)
    torch.ao.quantization.convert(model, inplace=True)