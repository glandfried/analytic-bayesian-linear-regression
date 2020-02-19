Maximum likelihood estimation can lead to severe over-fitting if complex
models (e.g. polynomial regression models of high order) are fit to
datasets of limited size. Common approachs to prevent over-fitting is to
add a regularization term to the error function or to perform
corss-validation.

Bayesian treatment of linear regression avoid over-fitting problem of
maximum likelihood, and which will also leads to automatic methods of
determining model complexity using the training data alone.

For a Bayesian treatment of linear regression we need a prior
probability distribution over model parameters $w$.

$$p(\vm{w}) = \N(\vm{w}|\vm{m}_0, \vm{S}_0)$$

Due to the choice of a conjugate Gaussian prior, the posterior will also
be Gaussian. We can evaluate this distribution by the usal procedure of
completing the square in the exponential, and then finding the
normalization coefficient using the standard result for normalized
Gaussian. The work for deriving the general result is at equation
(\[eq:post\])

$$p(\vm{w}|\vm{t}) = \N(\vm{w}|\vm{m}_N, \vm{S}_N)$$

where

$$\vm{m}_N = \vm{S}_N (\vm{S}_0^{-1} \vm{m}_0 + \beta \vm{\Phi}^T \vm{t})$$

$$\vm{S}_N^{-1} = \vm{S}_0^{-1} + \beta \vm{\Phi}^T\vm{\Phi}$$

For simplicity, we consider a zero-mean isotropic Gaussian prior
governed by a single precision paramater $\alpha$ so that

$$p(\vm{w}) = N(\vm{w}|\vm{0}, \alpha^{-1} \vm{I})$$

then corresponding posterior distribution over $\vm{w}$ is then

$$\vm{m}_N = \beta  \vm{S}_N\vm{\Phi}^T \vm{t}$$

$$\vm{S}_N^{-1} = \alpha \vm{I} + \beta \vm{\Phi}^T\vm{\Phi}$$

To obtain an analytical solution we will treat $\beta$ as a known
constant. Note that in supervised learning problems such as regression
we are not seeking to model the distribution of the input variables, so
we will treat the input $\vm{x}$ as a known constant.

  ![image](figures/png/linearModel.png)

    def posterior(alpha, beta, t, Phi):
        S_N_inv = alpha * np.eye(Phi.shape[1]) + beta * Phi.T.dot(Phi)
        S_N = np.linalg.inv(S_N_inv)
        m_N = beta * S_N.dot(Phi.T).dot(t)
        return m_N, S_N

  ---------------------------------------------------------- ---------------------------------------------------------- ----------------------------------------------------------
                          Likelihood                                              Prior/Posterior                                               Data Space
                                               ![image](figures/png/linearModel.png)                           ![image](figures/png/linearModel.png)
2        4                              4
  ---------------------------------------------------------- ---------------------------------------------------------- ----------------------------------------------------------

 

 

![Ilustration of a sequential Bayesian learning for a simple linear
model of the form
$y(x,\vm{w})= w_0 + w_1 x$](figures/png/example1_posterior_0.png)

 

![Ilustration of a sequential Bayesian learning for a simple linear
model of the form
$y(x,\vm{w})= w_0 + w_1 x$](figures/png/example1_dataSpace_0.png)

  ![Ilustration of a sequential Bayesian learning for a simple
linear model of the form
$y(x,\vm{w})= w_0 + w_1 x$](figures/png/example1_likelihood_1.png "fig:")

  ![Ilustration of a sequential Bayesian learning for a simple
linear model of the form
$y(x,\vm{w})= w_0 + w_1 x$](figures/png/example1_posterior_1.png "fig:")

  ![Ilustration of a sequential Bayesian learning for a simple
linear model of the form
$y(x,\vm{w})= w_0 + w_1 x$](figures/png/example1_dataSpace_1.png "fig:")

  ![Ilustration of a sequential Bayesian learning for a simple
linear model of the form
$y(x,\vm{w})= w_0 + w_1 x$](figures/png/example1_likelihood_2.png "fig:")

  ![Ilustration of a sequential Bayesian learning for a simple
linear model of the form
$y(x,\vm{w})= w_0 + w_1 x$](figures/png/example1_posterior_2.png "fig:")

  ![Ilustration of a sequential Bayesian learning for a simple
linear model of the form
$y(x,\vm{w})= w_0 + w_1 x$](figures/png/example1_dataSpace_2.png "fig:")

  ![Ilustration of a sequential Bayesian learning for a simple
linear model of the form
$y(x,\vm{w})= w_0 + w_1 x$](figures/png/example1_likelihood_3.png "fig:")

  ![Ilustration of a sequential Bayesian learning for a simple
linear model of the form
$y(x,\vm{w})= w_0 + w_1 x$](figures/png/example1_posterior_3.png "fig:")

  ![Ilustration of a sequential Bayesian learning for a simple
linear model of the form
$y(x,\vm{w})= w_0 + w_1 x$](figures/png/example1_dataSpace_3.png "fig:")

  ![Ilustration of a sequential Bayesian learning for a simple
linear model of the form
$y(x,\vm{w})= w_0 + w_1 x$](figures/png/example1_likelihood_4.png "fig:")

  ![Ilustration of a sequential Bayesian learning for a simple
linear model of the form
$y(x,\vm{w})= w_0 + w_1 x$](figures/png/example1_posterior_4.png "fig:")

  ![Ilustration of a sequential Bayesian learning for a simple
linear model of the form
$y(x,\vm{w})= w_0 + w_1 x$](figures/png/example1_dataSpace_4.png "fig:")
