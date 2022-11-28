import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

import torch
from data_loader import load_data, process_data, convert_to_pytorch
from models import Regressor
from trainer import train_model, distillate_model, predict

# global variables
random_state = 42
batch_size = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_plots = True

# Experiment settings
# types of models to train:
# 1. teacher
# 2. baseline (simple gaussian sampling)
# 3. student (1) with generator sampling, decreasing alpha
# 4. student (2) with sampling by direct optimization of the generator loss function, decreasing alpha
# 5. student (3) with generator sampling, alpha=1
# 6. student (4) with sampling by direct optimization of the generator loss function, alpha=1
dataset = 'cpusmall'
train_teacher = True
train_baseline = True
train_student_1 = True
train_student_2 = True
train_student_3 = True
train_student_4 = True

teacher_hidden_size = 500
teacher_train_epochs = 500
teacher_lr = 0.001
teacher_weight_decay = 1e-5

student_hidden_size = 25
student_train_epochs = 2000
student_lr = 0.001
student_weight_decay = 1e-5

generator_input_size = 50
generator_hidden_size = 500
generator_lr = 0.001
generator_weight_decay = 1e-5
beta = 1e-5
gamma = 1e-5

direct_optimizer_epochs = 2
direct_optimizer_lr = 0.1

ng = 1
ns = 10
m = 50


def run_experiment(random_state, batch_size, device, save_plots, dataset, train_teacher, train_baseline,
                   train_student_1, train_student_2, train_student_3, train_student_4, teacher_hidden_size,
                   teacher_train_epochs, teacher_lr, teacher_weight_decay, student_hidden_size, student_train_epochs,
                   student_lr, student_weight_decay, generator_input_size, generator_hidden_size, generator_lr,
                   generator_weight_decay, beta, gamma, direct_optimizer_epochs, direct_optimizer_lr, ng, ns, m):
    # Results: teacher, baseline, generator (decreasing alpha), direct_optimizer (decreasing alpha),
    # generator (alpha=1), direct_optimizer (alpha=1)
    results = np.zeros(6)

    # Load data
    print(f"############## Dataset: {dataset}, Random State: {random_state} ##############")
    X, y = load_data(dataset)
    X_train, X_test, y_train, y_test = process_data(X, y, random_state=random_state)
    train_loader, valid_loader, test_loader = convert_to_pytorch(X_train, X_test, y_train, y_test,
                                                                 batch_size=batch_size,
                                                                 valid_required=True, valid_ratio=0.1,
                                                                 random_state=random_state)

    # Train teacher model
    print("############## Teacher Model ##############")

    teacher_model = Regressor(input_size=X.shape[1], hidden_size=teacher_hidden_size)
    teacher_save_path = os.path.join("teacher_models", f"T_{dataset}.pt")

    if train_teacher:
        train_losses, valid_losses = train_model(teacher_model, train_loader, valid_loader, teacher_lr,
                                                 teacher_weight_decay, device, teacher_save_path,
                                                 epochs=teacher_train_epochs)
        if save_plots:
            plt.plot(np.arange(len(valid_losses)), np.sqrt(np.array(valid_losses)))
            plt.ylabel("RMSE on validation set")
            plt.xlabel("Epochs")
            plt.savefig(os.path.join("plots", "teacher_model", f"T_{dataset}_RMSE.png"))
            plt.clf()

    teacher_model.load_state_dict(torch.load(teacher_save_path))

    y_pred, y_true = predict(teacher_model, device, test_loader)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    results[0] = rmse
    print(f"RMSE: {rmse}\n")

    # Train baseline model (simple sampling method)
    print("############# Baseline Model ##############")

    baseline_model = Regressor(input_size=X.shape[1], hidden_size=student_hidden_size)
    baseline_save_path = os.path.join("baseline_models", f"S_{student_hidden_size}_{dataset}_baseline.pt")
    bounds = np.array(list(zip(np.min(X_train, axis=0),
                               np.max(X_train,
                                      axis=0) + 0.01)))  # for qmc sampler, doesn't have effect if 0 used for alpha

    # Set alpha=0, i.e. loss contributed by xp only. Since xg is ignored, set batch size to m*2 to maintain same number of samples
    if train_baseline:
        valid_losses, _, _ = distillate_model(teacher_model, baseline_model, 'qmc_sampler', valid_loader,
                                              0, device, baseline_save_path, bounds,
                                              student_train_epochs, student_lr, student_weight_decay,
                                              generator_input_size, generator_hidden_size, generator_lr,
                                              generator_weight_decay, direct_optimizer_epochs,
                                              direct_optimizer_lr, ng, ns, m * 2, beta, gamma)
        np.save(os.path.join("plots", "baseline_model", f"S_{student_hidden_size}_{dataset}_valid_loss.npy"),
                np.array(valid_losses))

        if save_plots:
            plt.plot(np.arange(len(valid_losses)), np.sqrt(np.array(valid_losses)), label='valid RMSE')
            plt.ylabel("RMSE on validation set")
            plt.xlabel("Epochs")
            plt.savefig(os.path.join("plots", "baseline_model", f"S_{student_hidden_size}_{dataset}_baseline_RMSE.png"))
            plt.clf()

    baseline_model.load_state_dict(torch.load(baseline_save_path))

    y_pred, y_true = predict(baseline_model, device, test_loader)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    results[1] = rmse
    print(f"RMSE: {rmse}\n")

    # Train model using generative adversarial training
    print("#### Student Model (Generator method) #####")

    student_model = Regressor(input_size=X.shape[1], hidden_size=student_hidden_size)
    student_save_path = os.path.join("student_models", "generator", f"S_{student_hidden_size}_{dataset}_generator.pt")

    if train_student_1:
        valid_losses, generator_train_losses, _ = distillate_model(teacher_model, student_model, 'generator',
                                                                   valid_loader,
                                                                   'decreasing', device, student_save_path, None,
                                                                   student_train_epochs, student_lr,
                                                                   student_weight_decay,
                                                                   generator_input_size, generator_hidden_size,
                                                                   generator_lr,
                                                                   generator_weight_decay, direct_optimizer_epochs,
                                                                   direct_optimizer_lr, ng, ns, m, beta, gamma)
        np.save(
            os.path.join("plots", "student_generator_model", f"S_{student_hidden_size}_{dataset}_g_valid_losses.npy"),
            np.array(valid_losses))
        np.save(
            os.path.join("plots", "student_generator_model", f"S_{student_hidden_size}_{dataset}_g_gen_train_loss.npy"),
            np.array(generator_train_losses))

        if save_plots:
            plt.plot(np.arange(len(valid_losses)), np.sqrt(np.array(valid_losses)), label='valid RMSE')
            plt.ylabel("RMSE on validation set")
            plt.xlabel("Epochs")
            plt.savefig(
                os.path.join("plots", "student_generator_model", f"S_{student_hidden_size}_{dataset}_g_RMSE.png"))
            plt.clf()
            plt.plot(np.arange(len(generator_train_losses)), np.array(generator_train_losses),
                     label='generator train loss')
            plt.ylabel("Generator train loss")
            plt.xlabel("Epochs")
            plt.savefig(
                os.path.join("plots", "student_generator_model", f"S_{student_hidden_size}_{dataset}_g_genloss.png"))
            plt.clf()

    student_model.load_state_dict(torch.load(student_save_path))

    y_pred, y_true = predict(student_model, device, test_loader)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    results[2] = rmse
    print(f"RMSE: {rmse}\n")
    del student_model

    # Train model using direct optimization training method
    print("### Student Model (direct opt method) #####")

    student_model = Regressor(input_size=X.shape[1], hidden_size=student_hidden_size)
    student_save_path = os.path.join("student_models", "direct_optimizer",
                                     f"S_{student_hidden_size}_{dataset}_direct.pt")

    if train_student_2:
        valid_losses, generator_train_losses, _ = distillate_model(teacher_model, student_model, 'direct_optimizer',
                                                                   valid_loader,
                                                                   'decreasing', device, student_save_path, None,
                                                                   student_train_epochs, student_lr,
                                                                   student_weight_decay,
                                                                   generator_input_size, generator_hidden_size,
                                                                   generator_lr,
                                                                   generator_weight_decay, direct_optimizer_epochs,
                                                                   direct_optimizer_lr, ng, ns, m, beta, gamma)
        np.save(os.path.join("plots", "student_optimizer_model", f"S_{student_hidden_size}_{dataset}_d_valid_loss.npy"),
                np.array(valid_losses))
        np.save(
            os.path.join("plots", "student_generator_model", f"S_{student_hidden_size}_{dataset}_d_gen_train_loss.npy"),
            np.array(generator_train_losses))

        if save_plots:
            plt.plot(np.arange(len(valid_losses)), np.sqrt(np.array(valid_losses)), label='valid RMSE')
            plt.ylabel("RMSE on validation set")
            plt.xlabel("Epochs")
            plt.savefig(
                os.path.join("plots", "student_optimizer_model", f"S_{student_hidden_size}_{dataset}_d_RMSE.png"))
            plt.clf()
            plt.plot(np.arange(len(generator_train_losses)), np.array(generator_train_losses),
                     label='generator train loss')
            plt.ylabel("Generator train loss")
            plt.xlabel("Epochs")
            plt.savefig(
                os.path.join("plots", "student_optimizer_model", f"S_{student_hidden_size}_{dataset}_d_genloss.png"))
            plt.clf()

    student_model.load_state_dict(torch.load(student_save_path))

    y_pred, y_true = predict(student_model, device, test_loader)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    results[3] = rmse
    print(f"RMSE: {rmse}\n")
    del student_model

    # Train model using generative adversarial training with all xg and no xp
    # Set alpha=1, i.e. loss contributed by xg only. Since xp is ignored, set batch size to m*2 to maintain same number of samples
    print("### Student Model (generator method, alpha=1) #####")

    student_model = Regressor(input_size=X.shape[1], hidden_size=student_hidden_size)
    student_save_path = os.path.join("student_models", "direct_optimizer",
                                     f"S_{student_hidden_size}_{dataset}_generator2.pt")

    if train_student_3:
        valid_losses, generator_train_losses, _ = distillate_model(teacher_model, student_model, 'generator',
                                                                   valid_loader,
                                                                   1, device, student_save_path, None,
                                                                   student_train_epochs, student_lr,
                                                                   student_weight_decay,
                                                                   generator_input_size, generator_hidden_size,
                                                                   generator_lr,
                                                                   generator_weight_decay, direct_optimizer_epochs,
                                                                   direct_optimizer_lr, ng, ns, m * 2, beta, gamma)
        np.save(
            os.path.join("plots", "student_generator_model", f"S_{student_hidden_size}_{dataset}_g2_valid_loss.npy"),
            np.array(valid_losses))
        np.save(os.path.join("plots", "student_generator_model",
                             f"S_{student_hidden_size}_{dataset}_g2_gen_train_loss.npy"),
                np.array(generator_train_losses))

        if save_plots:
            plt.plot(np.arange(len(valid_losses)), np.sqrt(np.array(valid_losses)), label='valid RMSE')
            plt.ylabel("RMSE on validation set")
            plt.xlabel("Epochs")
            plt.savefig(
                os.path.join("plots", "student_generator_model", f"S_{student_hidden_size}_{dataset}_g2_RMSE.png"))
            plt.clf()
            plt.plot(np.arange(len(generator_train_losses)), np.array(generator_train_losses),
                     label='generator train loss')
            plt.ylabel("Generator train loss")
            plt.xlabel("Epochs")
            plt.savefig(
                os.path.join("plots", "student_generator_model", f"S_{student_hidden_size}_{dataset}_g2_genloss.png"))
            plt.clf()

    student_model.load_state_dict(torch.load(student_save_path))

    y_pred, y_true = predict(student_model, device, test_loader)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    results[4] = rmse
    print(f"RMSE: {rmse}\n")
    del student_model

    # Train model using direct optimization training method with all xg and no xp
    # Set alpha=1, i.e. loss contributed by xg only. Since xp is ignored, set batch size to m*2 to maintain same number of samples
    print("### Student Model (direct opt method, alpha=1) #####")

    student_model = Regressor(input_size=X.shape[1], hidden_size=student_hidden_size)
    student_save_path = os.path.join("student_models", "direct_optimizer",
                                     f"S_{student_hidden_size}_{dataset}_direct2.pt")

    if train_student_4:
        valid_losses, generator_train_losses, _ = distillate_model(teacher_model, student_model, 'direct_optimizer',
                                                                   valid_loader,
                                                                   1, device, student_save_path, None,
                                                                   student_train_epochs, student_lr,
                                                                   student_weight_decay,
                                                                   generator_input_size, generator_hidden_size,
                                                                   generator_lr,
                                                                   generator_weight_decay, direct_optimizer_epochs,
                                                                   direct_optimizer_lr, ng, ns, m * 2, beta, gamma)
        np.save(
            os.path.join("plots", "student_optimizer_model", f"S_{student_hidden_size}_{dataset}_d2_valid_loss.npy"),
            np.array(valid_losses))
        np.save(os.path.join("plots", "student_optimizer_model",
                             f"S_{student_hidden_size}_{dataset}_d2_gen_train_loss.npy"),
                np.array(generator_train_losses))

        if save_plots:
            plt.plot(np.arange(len(valid_losses)), np.sqrt(np.array(valid_losses)), label='valid RMSE')
            plt.ylabel("RMSE on validation set")
            plt.xlabel("Epochs")
            plt.savefig(
                os.path.join("plots", "student_optimizer_model", f"S_{student_hidden_size}_{dataset}_d2_RMSE.png"))
            plt.clf()
            plt.plot(np.arange(len(generator_train_losses)), np.array(generator_train_losses),
                     label='generator train loss')
            plt.ylabel("Generator train loss")
            plt.xlabel("Epochs")
            plt.savefig(
                os.path.join("plots", "student_optimizer_model", f"S_{student_hidden_size}_{dataset}_d2_genloss.png"))
            plt.clf()

    student_model.load_state_dict(torch.load(student_save_path))

    y_pred, y_true = predict(student_model, device, test_loader)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    results[5] = rmse
    print(f"RMSE: {rmse}\n")

    # save results
    np.save(os.path.join("results", f"results_{student_hidden_size}_{dataset}_seed{random_state}.npy"), results)


if __name__ == "__main__":
    run_experiment(random_state, batch_size, device, save_plots, dataset, train_teacher, train_baseline,
                   train_student_1, train_student_2, train_student_3, train_student_4, teacher_hidden_size,
                   teacher_train_epochs, teacher_lr, teacher_weight_decay, student_hidden_size,
                   student_train_epochs,
                   student_lr, student_weight_decay, generator_input_size, generator_hidden_size, generator_lr,
                   generator_weight_decay, beta, gamma, direct_optimizer_epochs, direct_optimizer_lr, ng, ns, m)
