# Breakthrough Concept:

# Simulating Neural Plasticity and Memory Formation in Spiking Neural Networks (SNNs)

# We’ll create a small-scale SNN that models Hebbian learning, showing how neurons adjust their synaptic weights based on activity — simulating memory formation. This combines computational neuroscience principles with Python coding.

# Python Implementation (using Brian2 for spiking neural networks)

# Install Brian2 if you haven't:
# pip install brian2

from brian2 import *
import matplotlib.pyplot as plt

# Simulation parameters
duration = 1*second
num_inputs = 10
num_neurons = 5

# Input spikes: Poisson spike trains
input_rates = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]*Hz
input_group = PoissonGroup(num_inputs, rates=input_rates)

# Neuron model: Leaky Integrate-and-Fire
tau = 10*ms
eqs = '''
dv/dt = ( - v ) / tau : 1
'''

neurons = NeuronGroup(num_neurons, eqs, threshold='v>1', reset='v=0', method='linear')
neurons.v = 0

# Synapses with Hebbian learning (Spike-Timing Dependent Plasticity)
synapses = Synapses(input_group, neurons,
                    model='''
                    w : 1
                    dApre/dt = -Apre/tau : 1 (event-driven)
                    dApost/dt = -Apost/tau : 1 (event-driven)
                    ''',
                    on_pre='''
                    v_post += w
                    Apre += 0.01
                    w = clip(w + Apost, 0, 1)
                    ''',
                    on_post='''
                    Apost += 0.01
                    w = clip(w + Apre, 0, 1)
                    ''')
synapses.connect(p=0.5)
synapses.w = 'rand()'

# Monitors
spike_mon = SpikeMonitor(neurons)
state_mon = StateMonitor(synapses, 'w', record=True)

# Run simulation
run(duration)

# Plot results
plt.figure(figsize=(12,5))

plt.subplot(121)
plt.title("Neuron Spikes")
plt.plot(spike_mon.t/ms, spike_mon.i, '.k')
plt.xlabel("Time (ms)")
plt.ylabel("Neuron index")

plt.subplot(122)
plt.title("Synaptic Weight Evolution")
for i in range(len(state_mon.w)):
    plt.plot(state_mon.t/ms, state_mon.w[i])
plt.xlabel("Time (ms)")
plt.ylabel("Weight")
plt.tight_layout()
plt.show()

# What This Code Does

# Creates 10 input neurons firing randomly (Poisson spike trains).

# Connects them to 5 output neurons modeled with Leaky Integrate-and-Fire.

# Uses Hebbian plasticity (STDP) to simulate learning.

# Monitors spikes and synaptic weights over time.

# Visualizes how neurons “learn” connections — memory formation in action.

# Breakthrough Potential

# By scaling this up, you could simulate network-level memory storage, pattern recognition, or disease models (e.g., Alzheimer’s synaptic decay).

# This bridges experimental neuroscience with computational modeling.
