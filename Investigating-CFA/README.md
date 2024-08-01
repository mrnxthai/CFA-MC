# Investigating-CFA
Reproduction of the counterfactual data augmentation (CFA) algorithm and experiments presented in 
"Solving the Class Imbalance Problem Using a Counterfactual Method for Data Augmentation" 
(Machine Learning with Applications, Volume 9, 15 September 2022, 100375).

## Abstract
The problem of class imbalance is a well-known issue in machine learning, where the number of samples in one class is 
significantly larger than the number of samples in the other class. In this report, we re-implemented the counterfactual 
data augmentation algorithm presented in the paper "Solving the Class Imbalance Problem Using a Counterfactual Method for Data Augmentation". 
The algorithm proposes a novel approach to augment the minority class by creating counterfactual samples. 
Our experiments were conducted on five datasets from the UCI repository and compared the results obtained 
by the authors in their paper with the results obtained by our re-implementation of the algorithm. 
We found that the method introduced in the paper did not yield improved results in our experiments. 
Additionally, we encountered difficulties in reimplementing the algorithm from the paper alone. 
Despite these challenges, the report provides an insight on the current state of the research in the field of class imbalance 
and counterfactual data augmentation and highlights the potentials and limitations of the proposed method.

# Repository structure
* `exp`: Experiments with the algorithm
* `src`: Our re-implementation of the algorithm
* `doc`: Contains the final [report](doc/XAI_Report_Mantiuk_Kurth.pdf) and the figures used in it.
