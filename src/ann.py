# How to get this shit.
# git clone git://github.com/pybrain/pybrain.git
# sudo python3 setup.py install
# sudo apt-get install python3-numpy python3-scipy
# Should be enough.

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

net = buildNetwork(4, 10, 2)
result = net.activate([1,3,2,0])
print(result)
ds = SupervisedDataSet(4, 2)

ds.addSample([1,3,2,0], [1,4])
ds.addSample([1,3,2,0], [1,4])

ds.addSample([1,3,2,1], [1,5])



trainer = BackpropTrainer(net, ds)
trainer.trainUntilConvergence()

result = net.activate([1,3,2,0])
print(result)