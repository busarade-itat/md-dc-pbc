# MD-DC-PBC Multi-dimensional Discriminant Chronicles Pattern-Based Classification

This Python script is dedicated to classify temporal sequence sets 
using discriminant chronicle as features of a global classifier. 

The **_main.py_** is the entry point of the application: 
- it allows to classify datasets using the discriminant chronicles or episodes as features. 
- The path of the **DCM-MD** executable used to extract chronicles has to be written in the variable **_CPP_DCM_** in the file **_GLOBAL.py_**. 

## Prerequisites

Your Python runtime should have `sklearn` and `numpy` available.

The path of the **DCM-MD** executable used to extract chronicles has to be written in the variable **_CPP_DCM_** in the file **_GLOBAL.py_**. 
The path to **WEKA** JAR file used for classifying the chronicles has to be written in the variable **_WEKA_** in the file **_GLOBAL.py_**. 

## Running the program 

The command `./main.py datasets_new/datasets/proportionality_2D_legacy --fmin 0.3 --gmin 2 --mincs 2 --maxcs 5 --fold 5 --classifier svc --n 90 --k 1 --legacy --vecsize 2` will produce a directory with a name beginning by **_xp_** followed by a timestamp. 
In this directory a file **_cmd_** will contain the command. 
Five directories (ex0, ex1, ex2, ex3 and ex4) will contain the results of the 5 folds (`--fold 5`). 
Two files are contained by each of those directories.

The **_chronicles_** file contains the extracted chronicles to classify the `proportionality_2D` dataset (`datasets/BIDE-D/blocks`). 
Those chronicles were extracted with a minimal frequency of 30% (`--fmin 0.3`) and a minimal growth rate of 2 (`-gmin 2`). 
They were also containing at least 2 events (`--mincs 2`) and at most 5 (`--maxcs 5`). 
Only the 90 most discriminant per class are kept (`--n 90`) and their growth rate is biased with 1 negative example (`--k 1`). 
The global classifier using those chronicles is a linear SVM (`-classifier svc`). 
The dataset is in a legacy *one-sequence-per-line* format (`--legacy`).
Lastly, the size of the vectors in the dataset is equal to 2 (`--vecsize 2`).

The **_res\_classify_** file contains the classification results. 
An example obtained from the previous command is: 

```
Accuracy: 0.85
Absolute accuracy: 34.0/40.0
Recall: 0.85
Precision: 0.85
pos accuracy: 19.0/20.0 (0.95)
neg accuracy: 15.0/20.0 (0.75)
```

The **_chronicles_** file contains the classification results. 
An example obtained from the previous command is:

```
C0: {A(0), B(1), C(2), D(3), }
A(0),B(1): (<-inf,-inf>, <inf,inf>)
A(0),C(2): (<-inf,-inf>, <inf,inf>)
A(0),D(3): (<-inf,-inf>, <inf,inf>)
B(1),C(2): (<-inf,-inf>, <inf,inf>)
B(1),D(3): (<-inf,-inf>, <inf,inf>)
C(2),D(3): (<10,10>, <inf,inf>)
class: pos
sup(c,pos)/sup(c,neg): 42.0/0.0

C1: {A(0), C(1), D(2), }
A(0),C(1): (<-inf,-inf>, <inf,inf>)
A(0),D(2): (<-inf,-inf>, <inf,inf>)
C(1),D(2): (<10,10>, <inf,inf>)
class: pos
sup(c,pos)/sup(c,neg): 42.0/0.0

C2: {B(0), C(1), D(2), }
B(0),C(1): (<-inf,-inf>, <inf,inf>)
B(0),D(2): (<-inf,-inf>, <inf,inf>)
C(1),D(2): (<10,10>, <inf,inf>)
class: pos
sup(c,pos)/sup(c,neg): 42.0/0.0

C3: {C(0), D(1), }
C(0),D(1): (<10,10>, <inf,inf>)
class: pos
sup(c,pos)/sup(c,neg): 42.0/0.0

C4: {C(0), D(1), }
C(0),D(1): (<-inf,-inf>, <9,9>)
class: neg
sup(c,pos)/sup(c,neg): 80.0/38.0

C5: {A(0), C(1), D(2), }
A(0),C(1): (<-inf,-inf>, <inf,inf>)
A(0),D(2): (<-inf,-inf>, <10,10>)
C(1),D(2): (<-inf,-inf>, <9,9>)
class: neg
sup(c,pos)/sup(c,neg): 59.0/20.0

C6: {A(0), B(1), C(2), D(3), }
A(0),B(1): (<5,5>, <inf,inf>)
A(0),C(2): (<-inf,-inf>, <inf,inf>)
A(0),D(3): (<-inf,-inf>, <inf,inf>)
B(1),C(2): (<-inf,-inf>, <inf,inf>)
B(1),D(3): (<-inf,-inf>, <inf,inf>)
C(2),D(3): (<-inf,-inf>, <9,9>)
class: neg
sup(c,pos)/sup(c,neg): 44.0/0.0

C7: {B(0), C(1), D(2), }
B(0),C(1): (<-inf,-inf>, <-1,-1>)
B(0),D(2): (<-inf,-inf>, <inf,inf>)
C(1),D(2): (<-inf,-inf>, <9,9>)
class: neg
sup(c,pos)/sup(c,neg): 59.0/16.0

C8: {A(0), D(1), }
A(0),D(1): (<-7,-7>, <-0,-0>)
class: neg
sup(c,pos)/sup(c,neg): 27.0/7.0

C9: {A(0), B(1), D(2), }
A(0),B(1): (<-inf,-inf>, <inf,inf>)
A(0),D(2): (<-inf,-inf>, <-0,-0>)
B(1),D(2): (<-inf,-inf>, <inf,inf>)
class: neg
sup(c,pos)/sup(c,neg): 32.0/11.0
```

Those chronicles are the discriminant multidimensional chronicles for S+ (`pos.dat`) and S- (`neg.dat`).

The meaning of the output is as follows:
- For example, for the first chronicle, `C0: {A(0), B(1), C(2), D(3), }` corresponds to the multiset of the chronicle,
while the `C0` identifier denotes an ordinal number of the mined discriminant multidimensional chronicle.
- The events are separated by commas.
- The line `C(2),D(3): (<10,10>, <inf,inf>)` corresponds to a hyperrectangle constraint of the chronicle.
- `C(2)` and `D(3)` correspond to indices and event type names in the multiset, it is so a temporal constraint between
- event type `C` and event type `D`. The temporal interval is defined by
- `(<10,10>, <inf,inf>)` what means that the temporal constraint is `C [<10,10>, <inf,inf>] D`.
- The line `class: pos` says that the chronicle is discriminant for S+ (for S- it would be `class: neg`).
- Finally, `sup(c,pos)/sup(c,neg): 42.0/0.0` corresponds to computed supports for S+/S-.

## Running unit tests

The unit tests are located in the `tests/` directory. Running the test
entry point `tests/main.py` will perform all tests.

Prior to running the tests, your Python environment should contain *MD-DC-PBC*
source roots for the tests to work correctly.

```
cd tests
python ./main.py
```

## Datasets

The directories **_datasets_old_** and **_datasets_new_** contain the datasets supplied with the original version of DCM
by Yann Dauxais et al. and a dataset with crystal growth data.

## Usage

```
usage: main.py [-h] [--fmin FMIN] [--gmin GMIN] [--mincs MINCS]
               [--maxcs MAXCS] [--test TEST] [--fold FOLD] [--k K]
               [--n N] [--C C] [--kernel KERNEL] [--verbose]
               [--classifier CLASSIFIER] [--vecsize VECTOR_SIZE]
               [--legacy] [--component-debug-in <MOCK_1_neg>,<MOCK_1_pos>,...,<MOCK_k_neg>,<MOCK_k_pos>]
               [--out OUTDIR] [--convert-only] [--disable-randomness]
               datasets
```

Note that the the `--component-debug-in`, `--convert-only` and `--disable-randomness` parameters should be used for debugging and testing only.
