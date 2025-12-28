# MLP Plant Classifier

A PyTorch implementation of a simple feedforward neural network that classifies plants based on stem height and leaf radius.

## Overview
This project demonstrates how a MLP can classify data by adding a hidden layer.

## Features
- **Simple Architecture** - 2-input -> hidden layer -> 3 outputs with ReLU activation
- **GPU Support** - Automatically uses CUDA if available
- **Checkpointing** - Save and load model states during training
- **Progress Tracking** - Configurable loss reporting intervals
- **Training Visualization** - Displays predictions after training