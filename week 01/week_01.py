"""
Week 1: Neural network basics in pure Python.
Tasks:
1. Single neuron forward pass
2. Small multi-neuron layer
3. Mean squared error (MSE) loss
4. Loss curve by manually varying one weight
5. Gradient descent using a numerical derivative
"""
import math
import matplotlib.pyplot as plt

# Task 1: Single neuron forward pass

inputs= [0.7, 1.45, -0.85]
weights=[0.4, 0.6, 0.3]
bias= 0.68
summation=0
for i, x in enumerate (inputs):  # enumerate gives both the index and the value
    weight= weights[i]
    summation+= x * weight       # weighted sum of the inputs
summation+=bias                  # sum with bias

output= 1/(1+math.exp(-summation)) # sigmoid activation
print(output)


# Task 2: Layer with multiple neurons

# Inputs stay the same, but each of the 3 neurons has its own weights and bias.
weights=[[0.4, 0.6, 0.3],  # neuron 1
         [0.1, 0.5, -0.6], # neuron 2
         [0.2, -0.7, 0.8]] # neuron 3
biases=[0.68, -0.45, 0.56]

outputs_layer= []          # one output per neuron
for i, neuron_weights in enumerate(weights):
    summation = 0
    for j, x in enumerate(inputs):
        summation += x * neuron_weights[j]
    summation += biases[i]
    outputs_layer.append(1 / (1 + math.exp(-summation)))
print(outputs_layer)



# Task 3: Loss function

# The second target is 0.5 instead of 0 or 1: a sigmoid output can never reach 0 or 1,
# so with those targets the loss curve in Task 4 would have no clear minimum.
targets=[1.0,0.5,0.0]
def loss_function(targets,outputs_layer): # function that calculates the loss (mean squared)
    loss= 0
    for target, output in zip(targets, outputs_layer):
        loss+= ((target - output)**2) # squared error per output
    loss/= len(outputs_layer) # mean over all inputs
    return loss
print(loss_function(targets,outputs_layer))


# Task 4: Vary one weight manually and plot the loss

def forward_pass(inputs, weights, biases):
    # We rewrite task 1 and 2 as a function, so we can re-run it after changing a weight.
    outputs = []
    for neuron_weights, bias in zip(weights, biases):
        summation = 0
        for x, w in zip(inputs, neuron_weights):
            summation += x * w
        summation += bias
        outputs.append(1 / (1 + math.exp(-summation)))
    return outputs

original_weight = weights[1][1]  # remember the original value

test_weights = []   # x axis
losses = []         # y axis
for i in range (-50,51):    # weight from -5.0 to 5.0 in steps of 0.1
    test_weight=i/10.0
    weights[1][1]= test_weight
    current_guess= forward_pass(inputs, weights, biases)
    current_loss=loss_function(targets,current_guess)
    print(f"Weight is: {test_weight} Loss is: {current_loss}")
    test_weights.append(test_weight)
    losses.append(current_loss)

weights[1][1]= original_weight # restore the original weight

# graph
plt.plot(test_weights, losses, label="Loss curve", color="purple")
plt.title("Loss as a function of weights[1][1]")
plt.xlabel("Weight value")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid(True)
plt.savefig("loss_curve.png", dpi=150)
plt.show()


# Task 5: Gradient descent with a numerical derivative

w=weights[1][1]     # the parameter we optimize
h=0.0001            # small nudge used to estimate the slope
learning_rate=0.4   # step size: scales the gradient (even if our slope is very large, this will fix it.)
steps=100
loss_history=[]

for step in range(steps):
    weights[1][1]=w
    output1=forward_pass(inputs, weights, biases)
    loss1=loss_function(targets,output1) # our first loss

    weights[1][1]=w+h
    output2=forward_pass(inputs, weights, biases)
    loss2=loss_function(targets,output2) # our second loss

    gradient=(loss2-loss1)/h     # numerical derivative: (f(w+h) - f(w)) / h

    loss_history.append(loss1)   # we add our loss to the history
    print(f"Step {step}: w = {w:.6f}, loss = {loss1:.6f}")

    w=w-(learning_rate*gradient)
    # positive slope means the loss grows with w, so we move the opposite way.


weights[1][1] = w  # keep the final value

plt.plot(loss_history, color="green")
plt.title("Loss during gradient descent")
plt.xlabel("Step")
plt.ylabel("Loss (MSE)")
plt.grid(True)
plt.savefig("gradient_descent.png", dpi=150)
plt.show()