import torch
from torch import nn, Tensor
import torch.nn.functional as F

class KLAttentionAwareKDLoss(nn.Module):
    '''
    Attention-aware knowledge distillation loss that compares internal representation
    of two models using KL-Divergence.

    Lambda closer to 1 puts more weight on cross-entropy and less on kl divergence.
    '''
    def __init__(self, llambda: float = 0.99, alpha: float = 0.9, factor: float = 0.01, total_epochs: int = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.kl_div = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.kd = KDLoss(alpha=alpha)

        assert(llambda >= 0 and llambda <= 1)
        self.llambda = torch.tensor(llambda)
        self.factor = factor
        self.total_epochs = total_epochs

        self.step_size = -(self.llambda / self.total_epochs) if total_epochs else 0

    def forward(self, outputs: list[Tensor | list[Tensor]]):

        teacher_out, teacher_layers = outputs[0]
        student_out, student_layers = outputs[1]
        
        at_loss = 0
        for t_layer, s_layer in zip(teacher_layers, student_layers):

            teacher = t_layer.flatten(start_dim=1)
            student = s_layer.flatten(start_dim=1)

            # teacher = F.normalize(teacher, p=2)
            # student = F.normalize(student, p=2)

            at_loss += self.kl_div(F.log_softmax(student, dim=1), F.log_softmax(teacher, dim=1))

        ce_loss = self.kd([[teacher_out], [student_out]])

        return (1 - self.llambda) * at_loss + (self.llambda) * ce_loss
    
    def step(self):
        self.llambda += self.step_size
    
class EuclidAttentionAwareKDLoss(nn.Module):
    '''
    Attention-aware knowledge distillation loss that compares internal representation
    of two models using euclidean distance/MSE.

    Lambda closer to 1 puts more weight on cross-entropy and less on kl divergence.
    '''
    def __init__(self, llambda: float = 1e3, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ce = torch.nn.CrossEntropyLoss()
        self.llambda = torch.tensor(llambda)

    def forward(self, outputs: list[Tensor | list[Tensor]]):

        teacher_out, teacher_layers = outputs[0]
        student_out, student_layers = outputs[1]
        
        at_loss = 0
        for t_layer, s_layer in zip(teacher_layers, student_layers):

            teacher = t_layer.flatten(start_dim=1)
            student = s_layer.flatten(start_dim=1)

            teacher = F.normalize(teacher, p=2)
            student = F.normalize(student, p=2)

            at_loss += torch.norm(student - teacher, p=2, dim=1).mean()

        ce_loss = self.ce(student_out, teacher_out.argmax(1))

        return self.llambda/2 * at_loss + ce_loss

class SSIMAttentionAwareNMLoss(nn.Module):
    '''
    Attention-aware knowledge distillation loss that compares internal representation
    of two models using NMSE (normalized MSE, derived from SSIM).

    Lambda closer to 1 puts more weight on cross-entropy and less on SSIM/NMSE.
    '''
    def __init__(self, llambda: float = 0.99, alpha: float = 0.9, factor: float = 0.01, total_epochs: int = None, stability_const: float = 1e-8, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.kd = KDLoss(alpha=alpha)

        assert 0 <= llambda <= 1, "Lambda must be between 0 and 1"
        self.llambda = torch.tensor(llambda)
        self.factor = factor
        self.total_epochs = total_epochs
        self.stability_const = stability_const  # Added stability constant

        self.step_size = -(self.llambda / self.total_epochs) if total_epochs else 0

    def nmse(self, student: torch.Tensor, teacher: torch.Tensor):
        """Compute NMSE loss between student and teacher activations, with stability constant."""
        num = torch.norm(student - teacher, p=2, dim=1) ** 2
        denom = (torch.norm(student, p=2, dim=1) + torch.norm(teacher, p=2, dim=1) + self.stability_const) ** 2
        return (num / denom).mean()

    def forward(self, outputs: list[torch.Tensor | list[torch.Tensor]]):
        teacher_out, teacher_layers = outputs[0]
        student_out, student_layers = outputs[1]

        at_loss = 0
        for t_layer, s_layer in zip(teacher_layers, student_layers):
            teacher = t_layer.flatten(start_dim=1)
            student = s_layer.flatten(start_dim=1)
            at_loss += self.nmse(student, teacher)

        ce_loss = self.kd([[teacher_out], [student_out]])

        return (1 - self.llambda) * at_loss + (self.llambda) * ce_loss

    def step(self):
        self.llambda += self.step_size
    
class KDLoss(nn.Module):
    '''
    Regular unsupervised knowledge distillation loss with soft-label and 
    pseudo-hard-label comparison.

    From Equation 1 of Black-box Few-shot Knowledge Distillation.

    '''
    def __init__(self, alpha: float = 0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.kl_div = torch.nn.KLDivLoss(reduction="batchmean", log_target=False)
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, outputs: list[Tensor | list[Tensor]]):

        teacher_out = outputs[0][0]
        student_out = outputs[1][0]

        # Soft targets
        soft_targets = F.log_softmax(student_out, dim=1)
        soft_labels = F.softmax(teacher_out, dim=1)
        kl_div_loss = self.kl_div(soft_targets, soft_labels)

        # Hard targets
        hard_label = teacher_out.argmax(dim=1)
        ce_loss = self.ce(student_out, hard_label)

        return self.alpha * kl_div_loss + (1.0 - self.alpha) * ce_loss
    
def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def reconstruction_loss(x_recon, x):
    x = x.view(x.size(0), -1)  # Flatten input to [batch, input_dim]
    return F.mse_loss(x_recon, x, reduction='sum') / x.shape[0]

def elbo_loss(x, x_recon, mu, logvar):
    recon_loss = reconstruction_loss(x_recon, x)
    kl_loss = kl_divergence(mu, logvar)
    return recon_loss + kl_loss, recon_loss, kl_loss