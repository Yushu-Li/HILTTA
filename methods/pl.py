"""
Builds upon: https://github.com/DequanWang/tent
Corresponding paper: https://arxiv.org/abs/2006.10726
"""


import torch.nn as nn
import torch.jit

from methods.base import TTAMethod



import math
import torch



import torch.nn.functional as F

def get_weight(total, each):
    num_batch = int(total/each)
    sum_up=0
    for i in range(num_batch):
        sum_up +=(i+2)/(num_batch+1) 
    sum_up*=each
    weight=[]
    for i in reversed(range(num_batch)):
        for j in range(each):
            weight.append((i+2)/(num_batch+1)/sum_up)
    return weight

def uncertainty_entropy_based(output):
    """ 
    Returns the uncertainty score of a probability distribution using
    entropy 
                
    """
    
    return -(output.softmax(1) * output.log_softmax(1)).sum(1)


class PL(TTAMethod):
    """SSL adapt model using cross-entropy loss on softmax outputs, with pseudo labels.
    """
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        self.random_seed = cfg.RNG_SEED
        self.threshold = 1.0
        self.num_classes = num_classes
        

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        imgs_test = x[0]
        outputs = self.model(imgs_test)
        
        pseudo_cls = torch.argmax(outputs, dim=1)
        scores = uncertainty_entropy_based(outputs)
        high_confidence_index = torch.where(scores < self.threshold * math.log(outputs.shape[1]))[0]
        loss = F.cross_entropy(outputs[high_confidence_index], pseudo_cls[high_confidence_index])
        # loss = F.cross_entropy(outputs, pseudo_cls)


        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return outputs

    def configure_model(self):
        """Configure model for use with tent."""
        # model.train()
        self.model.eval()  # eval mode to avoid stochastic depth in swin. test-time normalization is still applied
        # disable grad, to (re-)enable only what we update
        self.model.requires_grad_(False)
        # enable all trainable
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()   # always forcing train mode in bn1d will cause problems for single sample tta
                m.requires_grad_(True)
            # Train all other layers
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)
            # elif isinstance(m, nn.Linear):
            #     m.requires_grad_(False)
            else:
                m.requires_grad_(True)


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)
