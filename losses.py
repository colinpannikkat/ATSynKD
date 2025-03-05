import torch
from torch import nn, Tensor
import torch.nn.functional as F

class AttentionAwareKDLoss(nn.Module):
    '''
    Attention-aware knowledge distillation loss that compares internal representation
    of two models.

    Lambda closer to 1 puts more weight on cross-entropy and less on kl divergence.
    '''
    def __init__(self, llambda: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.kl_div = torch.nn.KLDivLoss(reduction="batchmean", log_target=False)
        self.ce = torch.nn.CrossEntropyLoss()
        self.llambda = nn.Parameter(torch.tensor(llambda), requires_grad=True)

    def forward(self, teacher_layers: list[Tensor], teacher_out: Tensor, 
                      student_layers: list[Tensor], student_out: Tensor):
        
        kl_div_loss = 0
        for t_layer, s_layer in zip(teacher_layers, student_layers):
            teacher_dim = t_layer.shape[-1]
            student_dim = s_layer.shape[-1]

            teacher = t_layer.reshape((t_layer.shape[0], teacher_dim ** 2))
            student = s_layer.reshape((t_layer.shape[0], student_dim ** 2))

            # teacher = teacher/torch.norm(teacher, p=2)
            teacher = F.normalize(teacher, p=1)
            # student = student/torch.norm(student, p=2)
            student = F.normalize(student, p=1)

            kl_div_loss += self.kl_div(F.log_softmax(student, dim=1), F.softmax(teacher, dim=1))

        ce_loss = self.ce(teacher_out, student_out)

        return (1 - self.llambda) * kl_div_loss + (self.llambda) * ce_loss