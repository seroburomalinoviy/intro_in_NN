import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as f_act

#1
def init_net():
    input_nodes = 784 # 28x28 pixels
    print('Input the number of hidden neurens: ')
    hidden_nodes = int(input())
    out_nodes = 10 # от 0 до 9
    print('Input the training speed (0.5): ')
    learn_speed = float(input())
    return input_nodes,hidden_nodes,out_nodes,learn_speed

#2
def create_net(input_nodes,hidden_nodes,out_nodes):
    w_in2hidden=np.random.uniform(0,0.5,(hidden_nodes,input_nodes))
    w_hidden2out=np.random.uniform(0,0.5,(out_nodes,hidden_nodes))
    return w_in2hidden,w_hidden2out

#5
def net_output(w_in2hidden,w_hidden2out,input_signal,return_hidden):
    inputs=np.array(input_signal,ndmin=2).T
    hidden_in=np.dot(w_in2hidden,inputs)
    hidden_out=f_act(hidden_in)
    final_in=np.dot(w_hidden2out,hidden_out)
    final_out=f_act(final_in)
    if return_hidden==0:
        return final_out
    else:
        return final_out,hidden_out

#4
def net_train(target_list,input_signal,w_in2hidden,w_hidden2out,learn_speed):
    targets = np.array(target_list, ndmin=2).T
    inputs = np.array(input_signal, ndmin=2).T
    final_out, hidden_out = net_output(w_in2hidden, w_hidden2out, input_signal, 1)
    out_errors = targets - final_out
    hidden_errors = np.dot(w_hidden2out.T, out_errors)
    w_hidden2out += learn_speed * np.dot((out_errors * final_out * (1 - final_out)), hidden_out.T)
    w_in2hidden += learn_speed * np.dot((hidden_errors * hidden_out * (1 - hidden_out)), inputs.T)
    return w_in2hidden, w_hidden2out

#3
def train_set(w_in2hidden, w_hidden2out, learn_speed):
    data_file = open("mnist_train.csv", 'r')
    training_list = data_file.readlines()
    data_file.close()
    for record in training_list:
        all_values = record.split(',')
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.999) + 0.001
    targets = np.zeros(10) + 0.001
    # FIXME: target[0]
    targets[int(all_values[0])] = 1.0
    w_in2hidden, w_hidden2out = net_train(targets, inputs, w_in2hidden, w_hidden2out, learn_speed)
    return w_in2hidden, w_hidden2out


def test_set(w_in2hidden, w_hidden2out):
    data_file = open("mnist_test.csv", 'r')
    test_list = data_file.readlines()
    data_file.close()
    test = []
    for record in test_list:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.999) + 0.001
        print(inputs)
        out_session = net_output(w_in2hidden, w_hidden2out, inputs, 0)

        if int(all_values[0]) == np.argmax(out_session):
            test.append(1)
        else:
            test.append(0)
    test = np.asarray(test)
    print('Net efficiency % =', (test.sum() / test.size) * 100)


def plot_image(pixels: np.array):
    plt.imshow(pixels.reshape((28, 28)), cmap='gray')
    plt.show()

from pprint import pprint
input_nodes, hidden_nodes, out_nodes, learn_speed = init_net()
w_in2hidden, w_hidden2out = create_net(input_nodes, hidden_nodes, out_nodes)

for i in range(5):
    print('Test #', i + 1)
    w_in2hidden, w_hidden2out = train_set(w_in2hidden, w_hidden2out, learn_speed)
    test_set(w_in2hidden, w_hidden2out)


data_file = open("mnist_test.csv", 'r')
test_list = data_file.readlines()
data_file.close()
# all_values = test_list[int(np.random.uniform(0, 9999))].split(',')
all_values = test_list[0].split(',')
inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.999) + 0.001
out_session = net_output(w_in2hidden, w_hidden2out, inputs, 0)
print(np.argmax(out_session))
plot_image(np.asfarray(all_values[1:]))

