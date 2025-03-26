# ------------------------------------------
# ANÁLISE COMPARATIVA: LDA vs QDA vs Naive Bayes
# Dataset: Ionosphere (Classificação Binária)
# ------------------------------------------

# 1. Pacotes necessários
library(tidyverse)
library(tidymodels)
library(mlbench)

# 2. Carregar e preparar dados
data(Ionosphere)
dados <- Ionosphere %>% 
  as_tibble() %>% 
  select(-V1, -V2) %>% 
  mutate(Class = as.factor(Class))

# 3. Definir modelos
modelos <- list(
  lda = discrim_linear() %>% set_engine("MASS") %>% set_mode("classification"),
  qda = discrim_quad() %>% set_engine("MASS") %>% set_mode("classification"),
  naive_bayes = naive_Bayes() %>% set_engine("klaR") %>% set_mode("classification")
)

# 4. Validação cruzada (5 folds)
set.seed(123)
validacao <- vfold_cv(dados, v = 5)

# 5. Avaliação dos modelos
resultados <- map_dfr(modelos, ~
                        workflow() %>%
                        add_model(.x) %>%
                        add_formula(Class ~ .) %>%
                        fit_resamples(validacao) %>%
                        collect_metrics() %>%
                        filter(.metric == "accuracy"),
                      .id = "modelo"
)

# 6. Resultados finais
resultados_finais <- resultados %>% 
  select(modelo, mean) %>% 
  arrange(desc(mean)) %>% 
  rename(acuracia = mean)

# 7. Visualização
print(resultados_finais)
cat("\n✅ Melhor método:", resultados_finais$modelo[1], "\n")

# 8. Gráfico comparativo (opcional)
ggplot(resultados_finais, aes(x = reorder(modelo, acuracia), y = acuracia, fill = modelo)) +
  geom_col() +
  geom_text(aes(label = round(acuracia, 3)), vjust = -0.5) +
  labs(title = "Comparação de Acurácia entre Modelos",
       x = "Modelo",
       y = "Acurácia Média") +
  theme_minimal()
