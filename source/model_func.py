import torch
import torch.nn as nn
import torch.nn.functional as F

from models import VoxelMorph
from loss_func import Grad


def load_model(cfg):
    if cfg['ModelType'] == 'VoxelMorph':
        model = VoxelMorph(
            backbone=cfg['BackBone'],
            feat_num=cfg['FeatNum'],
            img_size=cfg['ImgSize'],
            integrate_cfg=cfg['IntegrateConfig']
        )
    else:
        raise NotImplementedError

    return model


class GradientDescent(object):
    def __init__(self, model, optimizer, sim_loss, reg_loss, loss_weight, gradient_surgery):
        self.model = model
        self.optimizer = optimizer

        if sim_loss == 'MSE':
            self.similarity_cost = nn.MSELoss()
        elif sim_loss == 'MAE':
            self.similarity_cost = nn.L1Loss()
        else:
            raise NotImplementedError

        if reg_loss == 'Grad':
            self.smoothness_cost = Grad()
        else:
            raise NotImplementedError

        self.loss_weight = loss_weight
        self.gradient_surgery = gradient_surgery

    def update_gradient(self, fwd):

        self.optimizer.zero_grad()
        similarity_loss = self.similarity_cost(fwd['Moved'], fwd['Fixed'])
        similarity_loss.backward(retain_graph=True)
        similarity_gradient = self.model.get_gradient()

        self.optimizer.zero_grad()
        smoothness_loss = self.smoothness_cost(fwd['Flow'])
        smoothness_loss.backward()
        smoothness_gradient = self.model.get_gradient()

        print_info = 'Loss ==> Similarity: {:.5f} Smoothness: {:.5f} '.format(
            similarity_loss.item(), smoothness_loss.item())

        loss_info = {
            'Similarity': similarity_loss.item(),
            'Smoothness': smoothness_loss.item(),
        }

        aggregated_similarity_gradient = torch.cat(similarity_gradient)
        aggregated_smoothness_gradient = torch.cat(smoothness_gradient)

        signed_similarity_gradient = torch.sign(aggregated_similarity_gradient)
        signed_smoothness_gradient = torch.sign(aggregated_smoothness_gradient)

        overlap = signed_similarity_gradient == signed_smoothness_gradient
        overlap_ratio = torch.sum(overlap) * 100 / aggregated_similarity_gradient.size(-1)
        print_info += 'Overlap: {:.2f}% '.format(overlap_ratio.item())
        loss_info['OverlapRatio'] = overlap_ratio.item()

        if self.gradient_surgery is None:
            if 'Lambda' in fwd.keys():
                gradient = 1 / (0.05 ** 2) * aggregated_similarity_gradient + \
                           fwd['Lambda'].item() * aggregated_smoothness_gradient
            else:
                gradient = self.loss_weight['Similarity'] * aggregated_similarity_gradient + \
                           self.loss_weight['Smoothness'] * aggregated_smoothness_gradient

        elif self.gradient_surgery == 'AgrSum':
            gradient = overlap * aggregated_similarity_gradient

        elif self.gradient_surgery == 'AgrRand':
            gradient = overlap * aggregated_similarity_gradient

            rand_gradient = torch.randn_like(gradient, device=gradient.device)
            rand_gradient *= ~overlap

            scale = gradient.abs().mean()

            rand_gradient *= scale
            gradient += rand_gradient

        elif self.gradient_surgery == 'PCGrad':
            gradient = self.__pcgrad(aggregated_similarity_gradient, aggregated_smoothness_gradient)

        elif self.gradient_surgery == 'LayerWise':
            gradient = []
            for layer_idx, (similarity_layer_gradient, smoothness_layer_gradient) in enumerate(zip(similarity_gradient, smoothness_gradient)):
                cosine = F.cosine_similarity(similarity_layer_gradient, smoothness_layer_gradient, dim=0)
                if cosine < 0:
                    inner_prod = torch.dot(similarity_layer_gradient, smoothness_layer_gradient)

                    proj_gradient = inner_prod / torch.linalg.vector_norm(smoothness_layer_gradient).pow(2) * smoothness_layer_gradient
                    norm_gradient = similarity_layer_gradient - proj_gradient
                    gradient.append(norm_gradient)

                else:
                    gradient.append(similarity_layer_gradient)
            gradient = torch.cat(gradient)
        else:
            raise NotImplementedError

        self.model.set_gradient(gradient)
        self.optimizer.step()

        return similarity_loss.item(), loss_info, print_info

    def __pcgrad(self, main_gradient, auxiliary_gradient):
        cosine = F.cosine_similarity(main_gradient, auxiliary_gradient, dim=0)

        if cosine < 0:
            inner_prod = torch.dot(main_gradient, auxiliary_gradient)
            proj_gradient = inner_prod / torch.linalg.vector_norm(auxiliary_gradient).pow(
                2) * auxiliary_gradient
            norm_gradient = main_gradient - proj_gradient
            gradient = norm_gradient
        else:
            gradient = main_gradient

        return gradient

    @property
    def device(self):
        return self.model.model_device

