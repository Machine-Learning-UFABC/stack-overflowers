library(usethis)

# Coloque o seu usuário do github e seu e-mail
use_git_config(user.name = "<LOGIN_GITHUB>", user.email = "<SEU_E-MAIL>")

## ATENÇÃO, descomente as linhas abaixo se quer
## gerar um novo token de acesso

## Gera um token de acesso no github
#usethis::create_github_token()

## Guarda o token de acesso no seu computador
## É bom salvar o token anotado também em outro lugar
## caso esteja usando o RStudio online.
#gitcreds::gitcreds_set()
