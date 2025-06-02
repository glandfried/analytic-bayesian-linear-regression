import numpy as np
import GPy
from scipy import linalg, stats
from scipy.optimize import minimize
from scipy.special import gammaln
import matplotlib.pyplot as plt
from scipy.stats import invgamma


class BayesianLinearRegressionGPy:
    def __init__(self, variance_prior=1.0, noise_variance=1.0, estimate_noise=False,
                 noise_prior_shape=1e-4, noise_prior_rate=1e-4):
        """
        Inicializa la regresión lineal bayesiana usando GPy
        variance_prior: varianza del prior sobre los pesos (1/alpha en la notación de Bishop)
        noise_variance: varianza inicial del ruido (1/beta en la notación de Bishop)
        estimate_noise: si es True, estima la varianza del ruido desde los datos
        noise_prior_shape, noise_prior_rate: parámetros de la prior Gamma-Inversa para la varianza del ruido
        """
        self.variance_prior = variance_prior
        self.noise_variance = noise_variance
        self.estimate_noise = estimate_noise
        self.noise_prior_shape = noise_prior_shape
        self.noise_prior_rate = noise_prior_rate
        self.model = None
        self.posterior_mean = None
        self.posterior_cov = None
        self.noise_posterior_params = None
    #
    def fit(self, X, y):
        """
        Ajusta el modelo usando GPy
        X: matriz de diseño de forma (n_samples, n_features)
        y: vector objetivo de forma (n_samples, 1)
        """
        # Asegurarse de que y sea 2D
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        #
        # Estimar la varianza del ruido si es necesario
        if self.estimate_noise:
            self.noise_variance = self._estimate_noise_variance(X, y)
            print(f"Varianza del ruido estimada: {self.noise_variance}")
        #
        # Crear el kernel lineal con la varianza correcta
        kernel = GPy.kern.Linear(input_dim=X.shape[1])
        kernel.variances = self.variance_prior
        #
        # Crear y optimizar el modelo
        self.model = GPy.models.GPRegression(
            X=X,
            Y=y,
            kernel=kernel,
            noise_var=self.noise_variance
        )
        #
        # Optimizar hiperparámetros si se estima el ruido
        if self.estimate_noise:
            self.model.optimize()
            # Actualizar la varianza de ruido después de la optimización
            self.noise_variance = float(self.model.likelihood.variance)
        #
        # Calcular el posterior de los parámetros
        self._compute_parameter_posterior(X, y)
        #
        # Calcular el posterior de la varianza del ruido
        self._compute_noise_posterior(X, y)
        #
        return self
    #
    def _estimate_noise_variance(self, X, y):
        """
        Estima la varianza del ruido observacional usando evidencia marginal máxima
        """
        def neg_log_evidence(log_noise_var):
            noise_var = np.exp(log_noise_var)
            beta = 1.0 / noise_var
            alpha = 1.0 / self.variance_prior
            n_samples, n_features = X.shape
            #
            # Calcular la precisión posterior
            S_N_inv = alpha * np.eye(n_features) + beta * X.T @ X
            #
            try:
                # Usar Cholesky para estabilidad numérica
                L = linalg.cho_factor(S_N_inv)
                #
                # Calcular determinantes
                log_det_prior = n_features * np.log(alpha)
                log_det_posterior = 2 * np.sum(np.log(np.diag(L[0])))
                #
                # Calcular términos de la evidencia
                m_N = beta * linalg.cho_solve(L, X.T @ y)
                data_fit = beta * np.sum((y - X @ m_N)**2)
                complexity_penalty = m_N.T @ (alpha * np.eye(n_features)) @ m_N
                #
                # Log evidencia negativo
                neg_log_ev = 0.5 * (log_det_posterior - log_det_prior - n_samples * np.log(beta) + 
                                   data_fit + complexity_penalty + n_samples * np.log(2 * np.pi))
                #
                return float(neg_log_ev)
            except np.linalg.LinAlgError:
                # Si hay problemas de estabilidad numérica, devolver un valor grande
                return 1e10
        #
        # Optimizar la log-evidencia con respecto a log(noise_variance)
        initial_log_noise = np.log(self.noise_variance)
        result = minimize(neg_log_evidence, initial_log_noise, method='L-BFGS-B')
        #
        # Devolver la varianza del ruido estimada
        return np.exp(result.x[0])
    #
    def _compute_parameter_posterior(self, X, y):
        """
        Calcula la distribución posterior de los parámetros
        Usando las fórmulas del libro de Bishop (PRML)
        """
        n_features = X.shape[1]
        #
        # Precisión del prior (alpha) y del ruido (beta)
        alpha = 1.0 / self.variance_prior
        beta = 1.0 / self.noise_variance
        #
        # Matriz de precisión posterior
        S_N_inv = alpha * np.eye(n_features) + beta * X.T @ X
        #
        # Usar Cholesky para estabilidad numérica
        L = linalg.cho_factor(S_N_inv)
        #
        # Media posterior
        self.posterior_mean = beta * linalg.cho_solve(L, X.T @ y)
        #
        # Covarianza posterior
        self.posterior_cov = linalg.cho_solve(L, np.eye(n_features))
    #
    def _compute_noise_posterior(self, X, y):
        """
        Calcula la distribución posterior de la varianza del ruido
        Utilizando el posterior en forma de distribución Gamma Inversa
        """
        n_samples = X.shape[0]
        #
        # Calcular la media predictiva usando la media posterior de los parámetros
        y_pred = X @ self.posterior_mean
        #
        # Suma de errores cuadrados
        sum_sq_errors = np.sum((y - y_pred) ** 2)
        #
        # Parámetros posterior para distribución Gamma Inversa
        post_shape = self.noise_prior_shape + n_samples / 2
        post_rate = self.noise_prior_rate + sum_sq_errors / 2
        #
        self.noise_posterior_params = {
            'shape': float(post_shape),
            'rate': float(post_rate),
            'mode': float(post_rate / (post_shape + 1)),  # Moda de la distribución
            'mean': float(post_rate / (post_shape - 1)) if post_shape > 1 else float('inf'),  # Media
            'variance': float(post_rate ** 2 / ((post_shape - 1) ** 2 * (post_shape - 2))) if post_shape > 2 else float('inf')  # Varianza
        }
    #
    def get_noise_posterior(self):
        """
        Retorna los parámetros de la distribución posterior de la varianza del ruido
        """
        if self.noise_posterior_params is None:
            raise ValueError("El modelo debe ser ajustado primero")
        #
        return self.noise_posterior_params
    #
    def sample_from_noise_posterior(self, n_samples=1000):
        """
        Genera muestras de la distribución posterior de la varianza del ruido
        """
        if self.noise_posterior_params is None:
            raise ValueError("El modelo debe ser ajustado primero")
        #
        # Generar muestras de una distribución Gamma Inversa
        shape = self.noise_posterior_params['shape']
        rate = self.noise_posterior_params['rate']
        #
        # Para muestrear de una Gamma Inversa, muestreamos de una Gamma y tomamos el recíproco
        gamma_samples = np.random.gamma(shape, 1/rate, size=n_samples)
        inv_gamma_samples = 1 / gamma_samples
        #
        return inv_gamma_samples
    #
    def get_parameter_posterior(self):
        """
        Retorna la distribución posterior de los parámetros
        """
        if self.posterior_mean is None or self.posterior_cov is None:
            raise ValueError("El modelo debe ser ajustado primero")
        #
        return {
            'mean': self.posterior_mean,
            'covariance': self.posterior_cov
        }
    #
    def sample_from_posterior(self, n_samples=1000):
        """
        Genera muestras de la distribución posterior de los parámetros
        """
        if self.posterior_mean is None or self.posterior_cov is None:
            raise ValueError("El modelo debe ser ajustado primero")
        #
        return np.random.multivariate_normal(
            mean=self.posterior_mean.flatten(),
            cov=self.posterior_cov,
            size=n_samples
        )
    #
    def predict(self, X_new):
        """
        Realiza predicciones
        Retorna: media y varianza predictiva
        """
        if self.model is None:
            raise ValueError("El modelo debe ser ajustado primero")
        #
        mean, var = self.model.predict(X_new)
        return mean, var
    #
    def log_marginal_likelihood(self):
        """
        Calcula el logaritmo de la evidencia marginal
        """
        if self.model is None:
            raise ValueError("El modelo debe ser ajustado primero")
        #
        return float(self.model.log_likelihood())
    #
    def get_parameters(self):
        """
        Obtiene los parámetros del modelo
        """
        if self.model is None:
            raise ValueError("El modelo debe ser ajustado primero")
        #
        return {
            'kernel_variance': float(self.model.kern.variances),
            'noise_variance': float(self.model.likelihood.variance)
        }

# Función de ayuda para calcular intervalos de credibilidad del ruido
def noise_credible_interval(noise_posterior, confidence=0.95):
    """
    Calcula el intervalo de credibilidad para la varianza del ruido
    """
    shape = noise_posterior['shape']
    rate = noise_posterior['rate']
    #
    # Puntos para el intervalo de credibilidad de la distribución Gamma Inversa
    lower = stats.invgamma.ppf((1 - confidence) / 2, shape, scale=rate)
    upper = stats.invgamma.ppf(1 - (1 - confidence) / 2, shape, scale=rate)
    #
    return lower, upper

# Función de ayuda para comparar modelos
def compare_models(X, y, prior_variances, estimate_noise=True):
    """
    Compara modelos con diferentes priors usando la evidencia marginal
    """
    results = []
    #
    for var_prior in prior_variances:
        model = BayesianLinearRegressionGPy(
            variance_prior=var_prior,
            noise_variance=0.1,
            estimate_noise=estimate_noise
        )
        model.fit(X, y)
        #
        log_evidence = model.log_marginal_likelihood()
        results.append({
            'prior_variance': var_prior,
            'log_evidence': log_evidence,
            'parameters': model.get_parameters(),
            'noise_posterior': model.get_noise_posterior()
        })
    #
    return results



import pandas as pd
Alturas = pd.read_csv("alturas.csv")

N, _ = Alturas.shape
Y_alturas = np.array(Alturas.altura)
X = {}


X["base"] = np.array(pd.DataFrame({"Base": [1 for _ in range(N)],    # Origen
                       "Altura": Alturas.altura_madre,  # Pendiente
             }))

X["biologico"] = np.array(pd.DataFrame({"Base": [1 for _ in range(N)],    # Origen
                            "Altura": Alturas.altura_madre,  # Pendiente
                            "Sexo": (Alturas.sexo=="M")+0     # Sexo
             }))

X_identidad = {"Base": [1 for _ in range(N)],    # Origen
               "Altura": Alturas.altura_madre  # Pendiente
            }
for i in range(25):
    X_identidad[f'id{i}'] = [ ((j % 25) == i)+0 for j in range(N)]

X["identidad"] = np.array(pd.DataFrame(X_identidad))



modelos = {}
for m in ["biologico", "base", "identidad"]:
    #
    # Ajustar modelo
    modelos[m] = BayesianLinearRegressionGPy(
        variance_prior=100,
        noise_variance=10,  # Valor inicial
        estimate_noise=False,  # Usamos False para obtener el posterior completo
        noise_prior_shape=1e-4,
        noise_prior_rate=1e-4
    )
    modelos[m].fit(X[m], Y_alturas )
    print(f"log(P(Modelo_{m}|Datos)) = ", modelos[m].log_marginal_likelihood())
    print(f"log(P(Modelo_{m}|Datos)) = ", modelos[m].log_marginal_likelihood())


plt.figure(figsize=(8, 5))
# Generate x values
x = np.linspace(0.01, 100, 1000)
for m in ["biologico", "base", "identidad"]:
    # Parameters
    noise_post = modelos[m].get_noise_posterior()
    alpha = noise_post["shape"]  # Shape parameter
    beta = noise_post["rate"]   # Rate parameter (scale = 1/beta)
    scale = beta
    # Compute PDF
    pdf = invgamma.pdf(x, a=alpha, scale=scale)
    # Plot
    plt.plot(x, pdf, label=f'Inverse Gamma (α={alpha}, β={beta})')


plt.xlabel('x')
plt.ylabel('Density')
plt.title('Inverse Gamma Distribution')
plt.legend()
plt.grid()
plt.show()

