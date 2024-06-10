import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
from torchvision.models.vgg import vgg16, vgg19 
from torch.ao.quantization import QuantStub, DeQuantStub
from utils.utils import calculate_op
from utils.train_utils import create_dataset,compute_accuracy
from utils.pca import extract_primary_layers, produce_return_nodes
from utils.quantization import quantize_model, print_size_of_model
from tqdm import tqdm
import json, os
from glob import glob

x = torch.randn(1, 3, 224, 224)
_, _, trainloader_cifar10, testloader_cifar10 = create_dataset(dataset = "cifar10", data_root = "./data", batch_size = 2048, num_workers = 2)
_, _, trainloader_cifar100, testloader_cifar100 = create_dataset(dataset = "cifar100", data_root = "./data", batch_size = 2048, num_workers = 2)

threshold = 0.99
save_dir = "./result/"

vgg_16_cifar_10_weight_path = "result/05_27_175941_vgg16_cifar10/best_model.pt"
vgg_16_cifar_100_weight_path = "result/05_28_142643_vgg16_cifar100/best_model.pt"
vgg_19_cifar_10_weight_path = "result/05_23_153908_vgg19_cifar10/best_model.pt"
vgg_19_cifar_100_weight_path = "result/05_23_154059_vgg19_cifar100/best_model.pt"

weight_path_list = [vgg_16_cifar_10_weight_path,
                   vgg_16_cifar_100_weight_path,
                   vgg_19_cifar_10_weight_path,
                   vgg_19_cifar_100_weight_path]

return_nodes_vgg_16 = produce_return_nodes(vgg16())
return_nodes_vgg_19 = produce_return_nodes(vgg19())

return_nodes_list = [return_nodes_vgg_16, 
                     return_nodes_vgg_19]


def extract_result_data(model, model_path, return_nodes, train_loader, test_loader, num_primary_layers, inverse = False):
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    model_size = print_size_of_model(model)
    
    ori_test_acc = compute_accuracy(test_loader, model, "cpu")
    ori_macs, ori_params = calculate_op(x, model)
    
    # for random layer quantization, replace the function "primary_layers" by selecting num_primary_layers amount of layers randomly.
    # the example format is [('features.1', 0.578125), ('features.4', 0.3828125), ('features.7', 0.34375)], list of tuples containing layer name and its pca score. 
    # Note that pca score can be any float value as long as the format is maintained as it is not used in quantization process.
    primary_layers = extract_primary_layers(x, model, return_nodes, threshold = threshold, num_layers = num_primary_layers, verbose = False, inverse=inverse)
    quantize_model(model, primary_layers, train_loader, 1)
    
    test_acc = compute_accuracy(test_loader, model, "cpu")
    q_macs, q_params = calculate_op(x, model)
    
    q_model_size = print_size_of_model(model)
    
    return [ori_test_acc, test_acc, model_size, q_model_size, ori_macs, q_macs, ori_params, q_params]

for i, weight_path in enumerate(weight_path_list):
    model_index = i//2
    return_nodes = return_nodes_list[model_index]
    full_layer_count = len(return_nodes)
    
    if "cifar100" in weight_path:
        trainloader, testloader = trainloader_cifar100, testloader_cifar100
    else:
        trainloader, testloader = trainloader_cifar10, testloader_cifar10
    
    main_key = '_'.join(weight_path.split("/")[1].split("_")[-2:])
    
    if i//2 == 0:
        # continue
        result_list = []
        for num_primary_layers in tqdm(range(1, full_layer_count+1)):
            result = extract_result_data(vgg16(), weight_path, return_nodes, trainloader, testloader, num_primary_layers)
            result_list.append({num_primary_layers: result})        
        with open(os.path.join(save_dir, main_key + '.json'), 'w') as f:
            json.dump(result_list, f)
    
    elif i//2 == 1:
        # continue
        result_list = []
        for num_primary_layers in tqdm(range(1, full_layer_count+1)):
            result = extract_result_data(vgg19(), weight_path, return_nodes, trainloader, testloader, num_primary_layers)
            result_list.append({num_primary_layers: result})
        with open(os.path.join(save_dir, main_key + '.json'), 'w') as f:
            json.dump(result_list, f)
