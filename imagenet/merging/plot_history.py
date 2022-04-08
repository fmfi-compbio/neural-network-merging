import matplotlib.pyplot as plt
import torch


def plot_history(history, indices, names, save_path):
    plt.clf()
    for index, name in zip(indices, names):
        losses = [x[index] for x in history]
        plt.plot(losses, label=name)
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(save_path)


def plot_gate_values(gate_values, student_num):
    # epoch x layer x gate
    for layer_num in range(len(gate_values[0])):
        layer_gates_by_epoch = [x[layer_num] for x in gate_values]
        arr = torch.stack(layer_gates_by_epoch).detach().cpu().numpy()
        plt.clf()
        plt.plot(arr[:,:arr.shape[1]//2], c='blue', alpha=0.2)
        plt.plot(arr[:,arr.shape[1]//2:], c='red', alpha=0.2)
        plt.title('layer number {}'.format(layer_num))
        plt.savefig('students/gates{}_{}.png'.format(student_num, layer_num))
