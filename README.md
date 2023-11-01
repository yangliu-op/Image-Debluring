# Image-Debluring

A pytorch implementation for the image debluring problem of the paper [Obtaining Pseudo-inverse Solutions With MINRES](https://arxiv.org/abs/2309.17096).

Bluring matrices can be full-rank but extremely ill-conditioned, e.g., Gaussian bluring matrix. Therefore, the bluring matrix behaves like a singular matrix numerically and the image debluring problem can be regarded as to find a good approximation of the ``pseudo-inverse solution'' of the ``singular'' least-squares problem. However, such problem will be very hard to solve as the blured images can contain large noise, say, from the communication loss. 

In this experiment, we use the canonical MINRES methods (and preconditioned MINRES) with lifting strategies from [Obtaining Pseudo-inverse Solutions With MINRESG](https://arxiv.org/abs/2309.17096) to approximately find such a solution iteratively and efficiently.
