import torch
from torchvision.models.vgg import vgg16, vgg11
from torchvision.models.feature_extraction import create_feature_extractor

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# paper reference: "https://arxiv.org/abs/1906.01493"
# calculation reference: "https://www.baeldung.com/cs/pca"

def compute_PCA(feature, threshold = 0.95, verbose = False):

    total_channel = feature.shape[1]

    activations = (feature.data).cpu().numpy()
    # print('shape of activations are:',activations.shape)
    a=activations.swapaxes(1,2).swapaxes(2,3)
    a_shape=a.shape
    # print('reshaped ativations are of shape',a.shape)
    # raw_input()

    pca = PCA() #number of components should be equal to the number of filters
    pca.fit(a.reshape(a_shape[0]*a_shape[1]*a_shape[2],a_shape[3]))
    a_trans=pca.transform(a.reshape(a_shape[0]*a_shape[1]*a_shape[2],a_shape[3]))
    # print('explained variance ratio is:',pca.explained_variance_ratio_)
    # raw_input()
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    d = np.argmax(cumsum >= threshold)
    
    # print(cumsum.shape)
    # print("pca: ", pca.components_.shape)
    
    # importance_ratio = d/total_channel

    if verbose:
        print('need at least {} filter(s) out of {} components to exceed threshold. So {:.02f}% of filters needed minimum to exceed threshold'.format(d, total_channel, d/total_channel*100))
    
    return d,  d/total_channel


def extract_primary_layers(x, model, return_nodes, threshold = 0.99, num_layers = 3, verbose = False, inverse = False): # num_layers for how many layers to quantize
    if num_layers > len(return_nodes.keys()):
        print("num layers cannot exceed number of activation layers within model")
        return
    
    
    model_nodes = {v: k for k, v in return_nodes.items()}
    
    extractor_model = create_feature_extractor(model, return_nodes=return_nodes)

    intermediate_outputs = extractor_model(x)

    # more important layer carries smaller pca_score
    # pca_score가 작을수록 더 중요한 레이어 입니다
    # 최소 fitler 필요 percentge가 작을수록 중요한 레이어 입니다
    
    pca_result_list = []
    
    for k in intermediate_outputs.keys():
        if verbose:
            print(model_nodes[k] + " analysis result")
        feature = intermediate_outputs[k]
        pca_score, pca_ratio = compute_PCA(feature, threshold, verbose = verbose)
        pca_result_list.append((model_nodes[k], pca_ratio))
        
    sorted_pca_result_list = sorted(pca_result_list, key=lambda x: -x[1])
    if inverse:
        sorted_pca_result_list = sorted(pca_result_list, key=lambda x: x[1])
    return sorted_pca_result_list[:num_layers]
    


