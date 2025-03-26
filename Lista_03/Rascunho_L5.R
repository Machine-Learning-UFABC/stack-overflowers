# Carregar pacotes necessários (garanta que estão instalados)
library(dplyr)         # Para o operador %>%
library(tidymodels)    # Para modelagem
library(discrim)       # Para LDA/QDA
library(klaR)         # Motor dos modelos
library(mlbench)      # Para o dataset Ionosphere

# Carregar dados
data(Ionosphere)
df <- as_tibble(Ionosphere) %>% 
  select(-V1, -V2)

# Definir modelos (usando sintaxe alternativa se %>% ainda falhar)
model_lda <- discrim_linear(engine = "klaR", mode = "classification")
model_qda <- discrim_quad(engine = "klaR", mode = "classification")
model_nb <- naive_Bayes(engine = "klaR", mode = "classification")

# Configurar validação cruzada
set.seed(123)
folds <- vfold_cv(df, v = 10)

# Avaliar modelos (sem %>% para evitar erros)
results_lda <- fit_resamples(
  workflow() |> 
    add_model(model_lda) |> 
    add_formula(Class ~ .),
  resamples = folds
) |> collect_metrics()

results_qda <- fit_resamples(
  workflow() |> 
    add_model(model_qda) |> 
    add_formula(Class ~ .),
  resamples = folds
) |> collect_metrics()

results_nb <- fit_resamples(
  workflow() |> 
    add_model(model_nb) |> 
    add_formula(Class ~ .),
  resamples = folds
) |> collect_metrics()

# Comparar resultados
cat("Acurácia média:\n")
cat("- LDA:", round(results_lda$mean[1], 4), "\n")
cat("- QDA:", round(results_qda$mean[1], 4), "\n")
cat("- Naive Bayes:", round(results_nb$mean[1], 4), "\n")

# Melhor método
best_method <- c("LDA", "QDA", "Naive Bayes")[
  which.max(c(
    results_lda$mean[1],
    results_qda$mean[1],
    results_nb$mean[1]
  ))
]
cat("\nMelhor método:", best_method)

