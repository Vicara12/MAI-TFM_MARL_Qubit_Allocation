# How are checkpoints structured?

This folder contains all the checkpoints saved during training.
Each folder corresponds to a type of model.
For example, the folder `russo` contains all checkpoints corresponding to the implementation of
Enrico Russo's paper.

Inside a model's folder one finds one folder for each training execution.
The format of the folder name is the number of logical qubits in the circuit, followed by the date
in the format `yymmdd` and the time as `hhmmss`.
Note that there can be no more than one checkpoint per second, otherwise the program will crash.

Inside a training instance's folder one finds all the checkpoints recorded during a given training
instance.
The folder name convention is the checkpoint number followed by an R and the cost obtained on
average when running the validation batch on it.

Inside each checkpoint folder there is the `.pth` file with the model's parameters, a svg image of
a sample circuit and another svg image with its corresponding allocation using the given model.

If you implement a new model type, remember to include the corresponding trained folder in `.gitignore`,
as the models take too much space.

Below is an example of a file structure for reference
```
trained
├── README.md
└── russo
    ├── 16lq_250726_114850
    │   ├── 0_R_129
    │   │   ├── allocations_cost_111.svg
    │   │   ├── circuit.svg
    │   │   └── model.pth
    │   ├── 1_R_111
    │   │   ├── allocations_cost_137.svg
    │   │   ├── circuit.svg
    │   │   └── model.pth
    │   └── 2_R_105
    │       ├── allocations_cost_97.svg
    │       ├── circuit.svg
    │       └── model.pth
    └── 16lq_250726_115057
        ├── 0_R_104
        │   ├── allocations_cost_119.svg
        │   ├── circuit.svg
        │   └── model.pth
        ├── 1_R_93
        │   ├── allocations_cost_111.svg
        │   ├── circuit.svg
        │   └── model.pth
        └── 2_R_91
            ├── allocations_cost_96.svg
            ├── circuit.svg
            └── model.pth
```