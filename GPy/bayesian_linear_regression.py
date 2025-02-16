import numpy as np
import GPy
from scipy import linalg

class BayesianLinearRegressionGPy:
    def __init__(self, variance_prior=1.0, noise_variance=1.0):
        """
        Inicializa la regresión lineal bayesiana usando GPy
        variance_prior: varianza del prior sobre los pesos (1/alpha en la notación de Bishop)
        noise_variance: varianza del ruido (1/beta en la notación de Bishop)
        """
        self.variance_prior = variance_prior
        self.noise_variance = noise_variance
        self.model = None
        self.posterior_mean = None
        self.posterior_cov = None
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
        # Calcular el posterior de los parámetros
        self._compute_parameter_posterior(X, y)
        #
        return self
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

# Función de ayuda para comparar modelos
def compare_models(X, y, prior_variances):
    """
    Compara modelos con diferentes priors usando la evidencia marginal
    """
    results = []
    #
    for var_prior in prior_variances:
        model = BayesianLinearRegressionGPy(
            variance_prior=var_prior,
            noise_variance=0.1
        )
        model.fit(X, y)
        #
        log_evidence = model.log_marginal_likelihood()
        results.append({
            'prior_variance': var_prior,
            'log_evidence': log_evidence,
            'parameters': model.get_parameters()
        })
    #
    return results

# Generar datos sintéticos
np.random.seed(42)
X = np.random.randn(100, 3)
true_weights = np.array([2, -1, 3])
y = X @ true_weights + np.random.randn(100) * 0.1

# Ajustar un modelo
model = BayesianLinearRegressionGPy(variance_prior=1.0, noise_variance=0.1)
model.fit(X, y)

# Hacer predicciones
X_test = np.random.randn(10, 3)
y_mean, y_var = model.predict(X_test)

# Obtener la evidencia marginal
log_evidence = model.log_marginal_likelihood()
print(f"Log evidencia marginal: {log_evidence}")

# Comparar diferentes priors
prior_variances = [0.1, 1.0, 10.0]
comparison = compare_models(X, y, prior_variances)

for result in comparison:
    print(f"\nPrior variance: {result['prior_variance']}")
    print(f"Log evidence: {result['log_evidence']}")
    print("Parameters:", result['parameters'])



import pandas as pd
Alturas = pd.read_csv("alturas.csv")

N, _ = Alturas.shape
Y_alturas = Alturas.altura
X_base = pd.DataFrame({"Base": [1 for _ in range(N)],    # Origen
                       "Altura": Alturas.altura_madre,  # Pendiente
             })

X_biologico = pd.DataFrame({"Base": [1 for _ in range(N)],    # Origen
                            "Altura": Alturas.altura_madre,  # Pendiente
                            "Sexo": (Alturas.sexo=="M")+0     # Sexo
             })
X_identidad = {"Base": [1 for _ in range(N)],    # Origen
               "Altura": Alturas.altura_madre  # Pendiente
            }
for i in range(25):
    X_identidad[f'id{i}'] = [ ((j % 25) == i)+0 for j in range(N)]

X_identidad = pd.DataFrame(X_identidad)



# Ajustar un modelo
model_bio = BayesianLinearRegressionGPy(variance_prior=100.0, noise_variance=10)
model_base = BayesianLinearRegressionGPy(variance_prior=100.0, noise_variance=10)
model_id = BayesianLinearRegressionGPy(variance_prior=100.0, noise_variance=10)


model_bio.fit(np.array(X_biologico), np.array(Y_alturas))
model_base.fit(np.array(X_base), np.array(Y_alturas))
model_id.fit(np.array(X_identidad), np.array(Y_alturas))

log_evidence_bio = model_bio.log_marginal_likelihood()
log_evidence_base = model_base.log_marginal_likelihood()
log_evidence_id = model.log_marginal_likelihood()

print(f"Log evidencia marginal: {log_evidence_bio}")
print(f"Log evidencia marginal: {log_evidence_base}")
print(f"Log evidencia marginal: {log_evidence_id}")

model_bio.get_parameter_posterior()
