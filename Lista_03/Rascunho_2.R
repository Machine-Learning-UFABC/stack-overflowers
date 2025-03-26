# 1. INSTALAÇÃO DOS PACOTES NECESSÁRIOS (se já não instalados)
required_pkgs <- c("tidyverse", "tidymodels", "discrim", "klaR", "mlbench", "MASS")
install.packages(setdiff(required_pkgs, rownames(installed.packages())))

# 2. CARREGAMENTO DE PACOTES COM CONFLICTOS RESOLVIDOS
library(tidyverse)
library(tidymodels)
library(discrim)
library(klaR)
library(mlbench)
library(MASS)  # Para LDA/QDA tradicional

# Resolver conflitos
conflicted::conflict_prefer("filter", "dplyr")
conflicted::conflict_prefer("select", "dplyr")

# 3. ALTERNATIVA PARA MODELOS DISCRIMINANTES
# Definir modelos usando MASS (mais estável) em vez de klaR
model_lda <- discrim_linear() %>% 
  set_engine("MASS") %>%  # Alterado para MASS
  set_mode("classification")

model_qda <- discrim_quad() %>% 
  set_engine("MASS") %>%  # Alterado para MASS
  set_mode("classification")

# Naive Bayes continua com klaR
model_nb <- naive_Bayes() %>% 
  set_engine("klaR") %>% 
  set_mode("classification")

# 4. CARREGAMENTO E PREPARAÇÃO DOS DADOS
data(Ionosphere, package = "mlbench")
df <- as_tibble(Ionosphere) %>% 
  select(-V1, -V2) %>% 
  mutate(Class = as.factor(Class))

# 5. VALIDAÇÃO CRUZADA
folds <- vfold_cv(df, v = 5)

# 6. FUNÇÃO DE AVALIAÇÃO ATUALIZADA
avaliar_modelo <- function(modelo) {
  tryCatch({
    workflow() %>%
      add_model(modelo) %>%
      add_formula(Class ~ .) %>%
      fit_resamples(folds, control = control_resamples(save_pred = TRUE)) %>%
      collect_metrics()
  }, error = function(e) {
    message("Erro no modelo: ", e$message)
    return(NULL)
  })
}

# 7. EXECUÇÃO DOS MODELOS
resultados <- list(
  lda = avaliar_modelo(model_lda),
  qda = avaliar_modelo(model_qda),
  nb = avaliar_modelo(model_nb)
) %>% compact()  # Remove modelos com erro

# 8. RESULTADOS
if (length(resultados) > 0) {
  acuracias <- map_dbl(resultados, ~.x %>% 
                         filter(.metric == "accuracy") %>% 
                         pull(mean))
  
  cat("\n=== RESULTADOS ===\n")
  imap(resultados, ~cat(.y, ":", .x %>% 
                          filter(.metric == "accuracy") %>% 
                          pull(mean) %>% round(4), "\n"))
  
  cat("\n✔ Melhor método:", names(which.max(acuracias)), "\n")
} else {
  cat("Todos os modelos falharam. Verifique os erros acima.")
}

# 9. DIAGNÓSTICO FINAL
cat("\n=== INFORMAÇÕES DO SISTEMA ===\n")
print(sessionInfo())