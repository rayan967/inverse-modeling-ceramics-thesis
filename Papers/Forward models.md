# Forward Models Papers

- [Adaptive Gaussian Process Regression for Efficient Building of
Surrogate Models in Inverse Problems](https://arxiv.org/pdf/2303.05824.pdf) 

In summary, the authors propose an approach that involves constructing a GPR surrogate model through adaptive sampling of design space, selecting the next training data using a greedy-type strategy, evaluating the surrogate model's approximation error based on its impact on parameter accuracy, and optimizing the combination of sample points and evaluation tolerance using estimates of computational work. They also consider improving the accuracy of existing sample points through continued finite element solution procedures.

- [Mechanical behavior predictions of additively manufactured microstructures using functional Gaussian process surrogates](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line)

In summary, the authors propose an architecture to form GPR forward models within the context of RVE microstructures to model functional data.

- [Neural Network Based Inverse System Identification from Small Data Sets](https://ieeexplore.ieee.org/document/8851722)

The authors propose an architecture to form NN forward models using virtual sample generation to address small data learning problem for neural networks.

- [A predictive machine learning approach for microstructure optimization and materials design](https://www.nature.com/articles/srep11551)

By incorporating ML-based preprocessing, the authors aim to efficiently explore the high-dimensional space of microstructures and identify multiple optimal solutions, making it a promising approach for microstructure-sensitive design in materials engineering. (Not very relevant)

- [An Intuitive Tutorial to Gaussian Processes Regression](https://arxiv.org/abs/2009.10862)

Comprehensive paper on GPR, discussing the basic concepts, distributions, kernels, non-parametric models, etc.

- [A Novel Methodology for Hydrocarbon Depth Prediction in Seabed Logging: Gaussian Process-Based Inverse Modeling of Electromagnetic Data](https://www.mdpi.com/2076-3417/11/4/1492)

A similar methodology for an application in seabed logging. They use GPRs to build a forward model based on data from FE simulations for EM responses from hydrocarbon depths. Then they apply gradient descent routines to predict the hydrocarbon depth in the SBL.  (very relevant)

- [Direct Inverse Modeling for Electromagnetic Components Using Gaussian Kernel Regression](https://ieeexplore.ieee.org/abstract/document/9714401?casa_token=V7SmxE68jgcAAAAA:Xi0NbZjBNzXuV5V1loayVbFJyI-looNpYtRy6wjFrM2ygmZi56r_GeeCZ-bTU_9MLdPMoupa3Q)

Another similar methodology to inverse modeling for design of electromagnetic (EM) devices by using a gaussian kernel regression and applying Newton's method as an optimization routine.

- [A machine learning framework for real-time inverse modeling and multi-objective process optimization of composites for active manufacturing control](https://www.sciencedirect.com/science/article/abs/pii/S135983682100531X)

A similar methodology to solve the inverse heat transfer problem, using a surrogate model for optimizing an inverse solution; parts of the paper are not accessible through uni credentials.