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

- [A deep learning–based method for the design of microstructural materials](https://link.springer.com/article/10.1007/s00158-019-02424-2)

A DNN application to inverse design of microstructures based on deep convolutional generative adversarial network (DCGAN) and convolutional neural network (CNN). The DCGAN is used to generate design candidates that satisfy geometrical constraints and the CNN is used as a surrogate model to link the microstructure to its properties. Once trained, the two networks are combined to form the design network which is utilized to for the inverse design.

- [Multi-target regression via input space expansion: treating targets as inputs](https://link.springer.com/article/10.1007/s10994-016-5546-z)

Multitarget regression by statistical dependencies between targets is exploited by building a separate model for each target on an expanded input space where other targets are treated as additional input variables (might not be relevant since there might not be any statistical dependence between variables).

- [Latin supercube sampling for very high-dimensional simulations](https://dl.acm.org/doi/abs/10.1145/272991.273010)

- [Orthogonal-maximin latin hypercube designs](https://www.jstor.org/stable/24308251?casa_token=1fWZyBmRZ8oAAAAA%3AKiDlhe557GpFpPduSTlIwHXtVy9EW9cf2mYwXcZ-i1takK8ENVNutFZHUq3ctR9vRN3J_XOzYfH1wpYO7A2S5yXlLARqqKLuaDZ2g3L2LtKbYBLln5w)

Literature on latin hypercube sampling, showcasing its use cases in high dimensional problems.

- [Parameter Estimation and Inverse Problems](https://books.google.de/books?hl=en&lr=&id=VuRyDwAAQBAJ&oi=fnd&pg=PP1&dq=Parameter+Estimation+and+Inverse+Problems%5D(R.+Aster,+B.+Borchers,+and+C.+Thurber&ots=PFTUQ1Cx7-&sig=g5aZS8kBxauvu_pROXy0mZW01Ik&redir_esc=y#v=onepage&q=Parameter%20Estimation%20and%20Inverse%20Problems%5D(R.%20Aster%2C%20B.%20Borchers%2C%20and%20C.%20Thurber&f=false )

- [An Introduction to Inverse Problems with Applications](https://books.google.de/books?hl=en&lr=&id=Oc8_N1PmYnYC&oi=fnd&pg=PP2&dq=F.+Neto+and+A.+da+Silva+Neto.+An+Introduction+to+Inverse+Problems+with+Applications.+Springer,+2012&ots=MmGjqNXOiu&sig=bSsbQOMc8ZMlMJsxNIUflrYSINs&redir_esc=y#v=onepage&q=F.%20Neto%20and%20A.%20da%20Silva%20Neto.%20An%20Introduction%20to%20Inverse%20Problems%20with%20Applications.%20Springer%2C%202012&f=false)

Comprehensive books on the topic of inverse modeling.

- [Surrogates: Gaussian Process Modeling, Design and Optimization for the Applied
Sciences](https://books.google.de/books?hl=en&lr=&id=1w_WDwAAQBAJ&oi=fnd&pg=PP1&dq=Surrogates:+Gaussian+Process+Modeling,+Design+and+Optimization+for+the+Applied+Sciences&ots=vRrs7y7vGa&sig=klhPa8XSvYRPQT-TCEWMloBmoJo&redir_esc=y#v=onepage&q=Surrogates%3A%20Gaussian%20Process%20Modeling%2C%20Design%20and%20Optimization%20for%20the%20Applied%20Sciences&f=false)

Comprehensive book on GPR, discussing the basic concepts, distributions, kernels, non-parametric models, etc.


- [Adaptive sampling applied to multivariate, multiple
output rational interpolation models with application to microwave circuits.](https://onlinelibrary.wiley.com/doi/abs/10.1002/mmce.10032)

The paper presents an adaptive sampling algorithm for building multivariate, multiple output rational interpolation models as surrogate models.

- [Using Gaussian process regression for efficient parameter reconstruction](https://arxiv.org/abs/1903.12128)

Paper by Philipp-Immanuel Schneider and Martin Hammerschmidt on inverse modeling with bayesian optimization and gaussian process regressors.

[Gaussian Processes for Machine Learning Book by Carl Edward Rasmussen and Christopher K. I. Williams]()

[Answer on gradient calculation with RBF kernel](https://stats.stackexchange.com/questions/373446/computing-gradients-via-gaussian-process-regression)

[Surrogate models based on machine learning methods for parameter estimation of left ventricular myocardium](https://royalsocietypublishing.org/doi/10.1098/rsos.20112)

[An overview of gradient descent optimization algorithms](https://arxiv.org/pdf/1609.04747.pdf)