import numpy as np
import torch
import torch.nn as nn


def generator_loss(x: torch.Tensor, teacher_model, student_model, device):
    teacher_model.eval().to(device)
    student_model.eval().to(device)
    teacher_label = teacher_model(x.float().to(device))
    student_pred = student_model(x.float().to(device))
    return -(teacher_label - student_pred) ** 2, student_pred


def generator_loss_numpy(x: np.array, teacher_model, student_model):
    x = torch.from_numpy(x)
    teacher_model.eval().to(device)
    student_model.eval().to(device)

    with torch.no_grad():
        teacher_label = teacher_model(x.float().to(device))
        student_pred = student_model(x.float().to(device))
    teacher_label = teacher_label.cpu().detach().numpy()
    student_pred = student_pred.cpu().detach().numpy()
    return -(teacher_label - student_pred) ** 2


class FunctionWrapper(nn.Module):
    """
    Wrapper for generator loss function with pytorch model
    Weights for the model represent the input X to be optimized
    """

    def __init__(self, m, d, device):
        super().__init__()
        weights = np.random.normal(loc=0.0, scale=1.0, size=(m, d))
        self.weights = nn.Parameter(torch.from_numpy(weights).float())
        self.device = device

    def forward(self, teacher_model, student_model):
        teacher_model.eval().to(self.device)
        student_model.eval().to(self.device)
        teacher_label = teacher_model(self.weights.to(self.device))
        student_pred = student_model(self.weights.to(self.device))
        return -(teacher_label - student_pred) ** 2, student_pred
