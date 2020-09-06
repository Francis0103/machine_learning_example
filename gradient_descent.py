import numpy as np
import matplotlib.pyplot as plt

epochs = 1000
learning_rate = 0.01

# gradinent descent example, mean square error
# linear regression: loss = 1/2*((input * weight + bias) - target)^2
# deep learning: loss = 1/2*(activate_function(input * weight + bias) - target)^2

def sigmoid(data):
    res = 1 / (1 + np.exp(-data))
    return res

def cross_entropy(logit, pred):
    loss = np.sum(np.multiply(logit, np.log(pred)))
    return loss

def gradient_updata(input_data, y_true, weight_data, bias_data, learning_rate):
    gradient_w = np.dot(input_data.T, (np.dot(input_data, weight_data) + bias_data - y_true))
    weight_data = weight_data - learning_rate * gradient_w
    gradient_b = np.dot(input_data, weight_data) + bias_data - y_true
    bias_data = bias_data - learning_rate * gradient_b
    return weight_data, bias_data

def mean_square_error(logit, pred):
    loss_value = sum(pow((logit - pred), 2)) / len(pred)
    return loss_value

def mm_forward(input_data, weight_data, bias_data):
    return np.dot(input_data, weight_data) + bias_data

def linear_regression():
    sample_num = 10
    n_in = 8
    n_out = 1
    np.random.seed(123)
    input_data = np.random.randn(sample_num, n_in) # (10, 8)
    weight_data = np.random.randn(n_in, n_out) # (8, 1)
    bias_data = np.random.randn(sample_num, n_out) # (1)
    output_data = mm_forward(input_data, weight_data, bias_data)
    y_true = np.random.randn(sample_num, n_out)
    loss = mean_square_error(y_true, output_data) 
    print("initial loss is {}".format(loss))
    loss_list = []
    loss_list.append(loss)
    for i in range(epochs):
        weight_data, bias_data = gradient_updata(input_data, y_true, weight_data,
                                                 bias_data, learning_rate)
        output_data = mm_forward(input_data, weight_data, bias_data)
        loss = mean_square_error(y_true, output_data)
        loss_list.append(loss)

    final_output = mm_forward(input_data, weight_data, bias_data)
    final_loss = mean_square_error(y_true, final_output)
    print("final loss is {}".format(final_loss))
    print("weight value is {}".format(weight_data.T))
    test_sample = input_data[3, :]
    test_label = y_true[3]
    test_pred = np.dot(test_sample, weight_data) + bias_data[3]
    print("test label is {} vs pred label is {}".format(test_label, test_pred))
    plt.figure()
    plt.plot(range(len(loss_list)), loss_list)
    plt.show()

def dnn():
    pass


if __name__ == '__main__':
    linear_regression()
    #dnn()