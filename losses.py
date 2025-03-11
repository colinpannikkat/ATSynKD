import torch
from torch import nn, Tensor
import torch.nn.functional as F

class AttentionAwareKDLoss(nn.Module):
    '''
    Attention-aware knowledge distillation loss that compares internal representation
    of two models.

    Lambda closer to 1 puts more weight on cross-entropy and less on kl divergence.
    '''
    def __init__(self, llambda: float = 0.5, alpha: float = 100., *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.kl_div = torch.nn.KLDivLoss(reduction="batchmean", log_target=False)
        self.ce = torch.nn.CrossEntropyLoss()
        self.mse = torch.nn.MSELoss()

        assert(llambda >= 0 and llambda <= 1)
        self.llambda = torch.tensor(llambda)
        self.alpha = torch.tensor(alpha)

    def forward(self, teacher_layers: list[Tensor], teacher_out: Tensor, 
                      student_layers: list[Tensor], student_out: Tensor):
        
        kl_div_loss = 0
        for t_layer, s_layer in zip(teacher_layers, student_layers):

            teacher = t_layer.flatten(start_dim=1)
            student = s_layer.flatten(start_dim=1)

            # teacher = F.normalize(teacher, p=1)
            # student = F.normalize(student, p=1)

            kl_div_loss += self.kl_div(F.log_softmax(student, dim=1), F.softmax(teacher, dim=1))

        ce_loss = self.ce(student_out, teacher_out.argmax(1))

        return (1 - self.llambda) * kl_div_loss + (self.llambda) * ce_loss
    
def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def reconstruction_loss(x_recon, x):
    x = x.view(x.size(0), -1)  # Flatten input to [batch, input_dim]
    return F.mse_loss(x_recon, x, reduction='sum') / x.shape[0]

def elbo_loss(x, x_recon, mu, logvar):
    recon_loss = reconstruction_loss(x_recon, x)
    kl_loss = kl_divergence(mu, logvar)
    return recon_loss + kl_loss, recon_loss, kl_loss