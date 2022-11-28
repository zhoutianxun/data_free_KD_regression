import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from scipy.stats import qmc
from models import Generator
from utils import generator_loss, FunctionWrapper


def train_epoch(model, device, dataloader, optimizer):
    model.train()
    train_loss = 0.0
    for x, y in dataloader:
        y_pred = model(x.float().to(device))
        loss = nn.MSELoss()(y_pred.reshape(-1), y.float().to(device))
        train_loss += loss.cpu().item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return train_loss / len(dataloader)


def valid_epoch(model, device, dataloader):
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for x, y in dataloader:
            y_pred = model(x.float().to(device))
            loss = nn.MSELoss()(y_pred.reshape(-1), y.float().to(device))
            valid_loss += loss.cpu().item()
    return valid_loss / len(dataloader)


def predict(model, device, dataloader):
    model.eval().to(device)
    y_pred = np.array([])
    y_true = np.array([])
    with torch.no_grad():
        for x, y in dataloader:
            pred = model(x.float().to(device))
            pred = pred.cpu().detach().numpy().reshape(-1)
            y_pred = np.concatenate((y_pred, pred))
            y = y.numpy().reshape(-1)
            y_true = np.concatenate((y_true, y))
    return y_pred, y_true


def train_model(model, train_loader, valid_loader, lr, weight_decay, device, save_path, epochs=100):
    model = model.to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)

    train_losses = []
    valid_losses = []
    min_val_loss = np.inf
    patience = 500
    patience_step = 0
    for epoch in tqdm(range(epochs)):
        train_loss = train_epoch(model, device, train_loader, optimizer)
        valid_loss = valid_epoch(model, device, valid_loader)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        if valid_loss < min_val_loss:
            min_val_loss = valid_loss
            torch.save(model.state_dict(), save_path)
        else:
            scheduler.step(valid_loss)
            patience_step += 1
        if patience_step > patience:
            break
    return train_losses, valid_losses


def distillate_model(teacher_model, student_model,
                     method, valid_loader,
                     alpha, device, save_path,
                     bounds=None,
                     student_train_epochs=2000,
                     student_lr=0.001,
                     student_weight_decay=1e-6,
                     generator_input_size=50,
                     generator_hidden_size=500,
                     generator_lr=0.001,
                     generator_weight_decay=1e-6,
                     direct_optimizer_epochs=2,
                     direct_optimizer_lr=0.1,
                     ng=1, ns=10, m=50,
                     beta=1e-5, gamma=1e-5):
    """
    :param teacher_model: teacher model to distill
    :param student_model: student model to fit to teacher
    :param method: either 'generator', 'direct_optimizer' or 'qmc_sampler'
    :param valid_loader: torch dataloader for computing validation loss
    :param alpha: either 'increasing', 'decreasing' or a float between 0 - 1
    :param device: torch device
    :param save_path: save path of student model
    :param bounds: only for qmc_sampler, [(lower bound, upper bound), ...] for all dimensions
    :param student_train_epochs: number of epochs for training
    :param student_lr: learning rate for student model
    :param student_weight_decay: weight decay (L2 regularization) for student model
    :param generator_input_size: random noise sample input size for generatior
    :param generator_hidden_size: hidden size of generator model
    :param generator_lr: learning rate for generator model
    :param generator_weight_decay: weight decay (L2 regularization) for generator model
    :param direct_optimizer_epochs: number of steps for direct optimization
    :param direct_optimizer_lr: learning rate for direct optimization
    :param ng: number of training batches for generator per epoch
    :param ns: number of training batches for student per epoch
    :param m: batch size for training generator / student
    :param beta: regularization strength for xg
    :param gamma: regularization strength for student_model(xg)
    :return: valid_losses, generator_train_losses, student_train_losses
    """
    dim = next(iter(valid_loader))[0].shape[1]
    if method == "generator":
        generator = Generator(generator_input_size, generator_hidden_size, dim).to(device)
        optimizer_generator = torch.optim.RMSprop(generator.parameters(), lr=generator_lr,
                                                  weight_decay=generator_weight_decay)
    elif method == "qmc_sampler":
        sampler = qmc.LatinHypercube(dim)
    elif method != "direct_optimizer":
        raise ValueError("Please input a valid method: 'generator', 'direct_optimizer' or 'qmc_sampler'. "
                         "Baseline model can be trained by choosing any method and setting alpha=0")

    if type(alpha) == str:
        if alpha not in ['increasing', 'decreasing']:
            raise ValueError("Please input a valid alpha: 'increasing', 'decreasing' or a float between 0 - 1")
    elif not (0 <= alpha <= 1):
        raise ValueError("Value of alpha not between 0 - 1")
    else:
        a = alpha

    teacher_model.eval()

    optimizer_student = torch.optim.RMSprop(student_model.parameters(), lr=student_lr,
                                            weight_decay=student_weight_decay)
    generator_train_losses = []
    student_train_losses = []
    valid_losses = []
    min_val_loss = np.inf

    for epoch in tqdm(range(student_train_epochs)):

        # Generate xg with method specified
        if method == 'generator':
            for i in range(ng):
                # generate synthetic data
                generator.train()
                xg = np.random.normal(loc=0.0, scale=1.0, size=(m, generator_input_size))
                xg = torch.from_numpy(xg).float()
                xg = generator(xg.to(device))

                loss_G, student_pred = generator_loss(xg, teacher_model, student_model, device)
                loss_G = torch.mean(loss_G + beta * torch.norm(xg, dim=1) ** 2 + gamma * student_pred ** 2)

                # Backward pass
                optimizer_generator.zero_grad()
                loss_G.backward()
                optimizer_generator.step()

        student_train_loss = 0.0
        xg_train_loss = 0.0

        # alpha (factor controlling proportion of loss for xg and xp)
        if alpha == 'increasing':
            a = epoch / student_train_epochs
        elif alpha == 'decreasing':
            a = 1 - epoch / student_train_epochs

        # Train student model for ns iterations
        for i in range(ns):
            # generate synthetic data
            xp = np.random.normal(loc=0.0, scale=1.0, size=(m, dim))
            xp = torch.from_numpy(xp).float().to(device)
            if method == 'generator':
                generator.eval()
                xg = np.random.normal(loc=0.0, scale=1.0, size=(m, generator_input_size))
                xg = torch.from_numpy(xg).float().to(device)
                xg = generator(xg)

            elif method == 'qmc_sampler':
                xg = sampler.random(n=m)
                xg = qmc.scale(xg, bounds[:, 0], bounds[:, 1])
                xg = torch.from_numpy(xg).float().to(device)

            elif method == 'direct_optimizer':
                fun = FunctionWrapper(m, dim, device)
                optimizer_direct = torch.optim.RMSprop(fun.parameters(), lr=direct_optimizer_lr, weight_decay=beta)
                for i in range(direct_optimizer_epochs):
                    direct_loss, student_pred = fun(teacher_model, student_model)
                    direct_loss = torch.mean(direct_loss + gamma * student_pred ** 2)
                    optimizer_direct.zero_grad()
                    direct_loss.backward()
                    optimizer_direct.step()
                xg = nn.utils.parameters_to_vector(fun.weights).view(m, -1).to(device)

            with torch.no_grad():
                teacher_model.eval().to(device)
                teacher_label_xg = teacher_model(xg)
                teacher_label_xp = teacher_model(xp)

            # train student model
            student_model.train().to(device)
            student_pred_xg = student_model(xg)
            student_pred_xp = student_model(xp)
            loss_xg = nn.MSELoss()(student_pred_xg.reshape(-1), teacher_label_xg.reshape(-1))
            loss_xp = nn.MSELoss()(student_pred_xp.reshape(-1), teacher_label_xp.reshape(-1))
            loss_S = a * loss_xg + (1 - a) * loss_xp
            student_train_loss += loss_S.cpu().item()
            xg_train_loss += a * loss_xg.cpu().item()

            # Backward pass
            optimizer_student.zero_grad()
            loss_S.backward()
            optimizer_student.step()

        valid_loss = valid_epoch(student_model, device, valid_loader)
        student_train_losses.append(student_train_loss / ns)
        generator_train_losses.append(xg_train_loss / ns)
        valid_losses.append(valid_loss)
        if valid_loss < min_val_loss:
            torch.save(student_model.state_dict(), save_path)
            min_val_loss = valid_loss

    return valid_losses, generator_train_losses, student_train_losses
