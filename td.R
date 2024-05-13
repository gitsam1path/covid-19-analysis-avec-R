
#importer le jeu de données

data <- read.csv("covid.csv")
# 1. PRE-PROCESSING

# 1.1 Création des sous-ensembles
# Définir la graine aléatoire pour la reproductibilité
set.seed(123)

# Définir la proportion d'observations pour l'ensemble de test (par exemple, 80% d'entraînement, 20% de test)
proportion_test <- 0.2

# Obtenir le nombre total d'observations
total_observations <- nrow(data)

# Calculer le nombre d'observations pour l'ensemble de test
num_test <- round(proportion_test * total_observations)

# Sélectionner de manière aléatoire les indices des observations pour l'ensemble de test
test_indices <- sample(1:total_observations, num_test)

# Créer l'ensemble d'entraînement
train_set <- data[-test_indices, ]

# Créer l'ensemble de test
test_set <- data[test_indices, ]

# 1.2 Nettoyage des NaN (vous pouvez ajuster en fonction de votre besoin)
train_set <- na.omit(train_set)
test_set <- na.omit(test_set)

# 1.3 Encodage (vous pouvez utiliser différentes méthodes selon le type de données)
# Exemple avec encodage one-hot pour les variables catégorielles
#train_set <- model.matrix(~ ., data = train_set)[, -1]
#test_set <- model.matrix(~ ., data = test_set)[, -1]

# 2. PROCEDURE D'EVALUATION

# 2.1 Matrice de confusion et rapport de classification
# (Assurez-vous de remplacer "target_variable" par le nom de votre variable cible)
model <- randomForest(target_variable ~ ., data = train_set)
predictions <- predict(model, newdata = test_set)
conf_matrix <- confusionMatrix(predictions, test_set$target_variable)
class_report <- confusionMatrix(predictions, test_set$target_variable)$byClass

# 2.2 Learning curves (utilisez les données d'entraînement pour construire les courbes d'apprentissage)

# 3. MODELISATION

# 3.1 Application des modèles
model_rf <- randomForest(target_variable ~ ., data = train_set)
model_adaboost <- ada(target_variable ~ ., data = train_set)
model_svm <- svm(target_variable ~ ., data = train_set)
model_knn <- kknn(target_variable ~ ., data = train_set)

# 4. OPTIMISATION

# 4.1 Optimiser les hyperparamètres avec GridSearchCV et RandomizedSearchCV
# (Assurez-vous de remplacer les paramètres et les modèles par ceux appropriés à votre cas)

# Grid Search
tuneGrid <- expand.grid(.mtry = c(2, 4, 6), .ntree = c(50, 100, 150))
model_rf_tuned <- train(target_variable ~ ., data = train_set, method = "rf", trControl = trainControl(method = "cv"), tuneGrid = tuneGrid)

# Randomized Search
param_grid <- expand.grid(.mtry = c(2, 4, 6), .ntree = c(50, 100, 150))
model_rf_random <- train(target_variable ~ ., data = train_set, method = "rf", trControl = trainControl(method = "cv"), tuneGrid = param_grid, tuneLength = 5)

# 4.2 Déterminer le seuil de décision avec les precision_recall_curves
# (Assurez-vous d'adapter en fonction de votre modèle et de vos données)

# Vous pouvez utiliser la bibliothèque ROCR pour créer la courbe precision_recall
library(ROCR)
predictions_prob <- predict(model_rf_tuned, newdata = test_set, type = "prob")[, "1"]
prediction_obj <- prediction(predictions_prob, test_set$target_variable)
perf <- performance(prediction_obj, "prec", "rec")
plot(perf)

# Utilisez la courbe pour sélectionner le seuil qui donne la précision ou le rappel désiré

# N'oubliez pas d'ajuster les détails spécifiques à votre ensemble de données et à vos modèles.



