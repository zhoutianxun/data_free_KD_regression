import pandas as pd
from run_experiment import *

datasets = ['compactiv', 'cpusmall', 'CTScan', 'Indoorloc', 'mv', 'pole', 'puma32h']
random_states = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
student_hidden_size = 50
rerun = True

if rerun:
    for dataset in datasets:
        for random_state in random_states:
            run_experiment(random_state, batch_size, device, save_plots, dataset, train_teacher, train_baseline,
                           train_student_1, train_student_2, train_student_3, train_student_4, teacher_hidden_size,
                           teacher_train_epochs, teacher_lr, teacher_weight_decay, student_hidden_size,
                           student_train_epochs, student_lr, student_weight_decay, generator_input_size,
                           generator_hidden_size, generator_lr, generator_weight_decay, beta, gamma,
                           direct_optimizer_epochs, direct_optimizer_lr, ng, ns, m)

# collate results
results = {}
for dataset in datasets:
    result_dataset = np.zeros(6)
    for random_state in random_states:
        replicate = np.load(os.path.join("results", f"results_{student_hidden_size}_{dataset}_seed{random_state}.npy"))
        result_dataset = result_dataset + replicate
    result_dataset = result_dataset/len(random_states)
    results[dataset] = result_dataset

results = pd.DataFrame(results).transpose()
print(results)