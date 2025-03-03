import torch
from torch import nn, Tensor
import torch.nn.functional as F

class AttentionAwareKDLoss(nn.Module):
    def __init__(self, lambda_val: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.kl_div = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.ce = torch.nn.CrossEntropyLoss()
        self.lambda_val = nn.Parameter(torch.tensor(lambda_val), requires_grad=True)

    def forward(self, teacher_layers: list[Tensor], teacher_out: Tensor, 
                      student_layers: list[Tensor], student_out: Tensor):
        
        kl_div_loss = 0
        for t_layer, s_layer in zip(teacher_layers, student_layers):
            teacher_dim = t_layer.shape[-1]
            student_dim = s_layer.shape[-1]

            teacher = t_layer.reshape((t_layer.shape[0], teacher_dim ** 2))
            student = s_layer.reshape((t_layer.shape[0], student_dim ** 2))

            teacher = teacher/torch.norm(teacher, p=2)
            student = student/torch.norm(student, p=2)

            kl_div_loss += self.kl_div(F.log_softmax(student, dim=1), F.log_softmax(teacher, dim=1))

        ce_loss = self.ce(F.softmax(teacher_out, dim=1), F.softmax(student_out, dim=1))

        return (1 - self.lambda_val) * kl_div_loss + (self.lambda_val) * ce_loss