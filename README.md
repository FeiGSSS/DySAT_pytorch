# DySAT: Deep Neural Representation Learning on Dynamic Graphs via Self-Attention Networks
This is a pytorch implementation of DySAT. All codes are adapted from official [implementation in TensorFlow](https://github.com/aravindsankar28/DySAT). This implementation is only tested using dataset Enron, and the results is inconsistent with official results (better than that). Code review and contribution is welcome!

# Raw Data Process
"""
cd raw_data/Enron
pyhton process.py
"""
The processed data will stored at 'data/Enron"

# Training
'''
python train --dataset Enron --time_steps 16
'''

