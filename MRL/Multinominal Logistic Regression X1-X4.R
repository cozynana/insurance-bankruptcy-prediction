# ======================================================
# 📦 PERSIAPAN DAN PEMBENTUKAN DATA
# ======================================================

if(!require(recipes)) install.packages("recipes", dependencies = TRUE)
if(!require(caTools)) install.packages("caTools")
if(!require(nnet)) install.packages("nnet")
if(!require(caret)) install.packages("caret")

library(recipes)
library(caTools)
library(nnet)
library(caret)

# Mengubah nama Kolom dari Z_Score_X1 hingga Z_Score_X4 menjadi X1 hingga X4
colnames(data_regresi) <- c("X1", "X2", "X3", "X4", "Z_Score")
data_regresi

# --- A. PEMBENTUKAN VARIABEL DEPENDEN KATEGORI (Y) ---
data_regresi$Status_Kategori <- cut(
  data_regresi$Z_Score, 
  breaks = c(-Inf, 1.1, 2.6, Inf),
  labels = c("Bangkrut", "Abu_Abu", "Sehat"), 
  right = FALSE
)

data_regresi$Status_Kategori <- factor(data_regresi$Status_Kategori)

# --- B. SPLIT DATA TRAINING DAN TESTING ---
set.seed(42)
split_tag <- sample.split(data_regresi$Status_Kategori, SplitRatio = 0.8)
data_training <- subset(data_regresi, split_tag == TRUE)
data_testing  <- subset(data_regresi, split_tag == FALSE)

cat("\nJumlah Data Training:", nrow(data_training), "records\n")
cat("Jumlah Data Testing :", nrow(data_testing), "records\n")

# ======================================================
# 🔹 C. PEMBENTUKAN MODEL REGRESI LOGISTIK MULTINOMIAL
# ======================================================
model_multinom <- multinom(
  Status_Kategori ~ X1 + X2 + X3 + X4,
  data = data_training,
  MaxNWts = 10000
)

cat("\n✅ Ringkasan Model Logistik Multinomial:\n")
print(summary(model_multinom))

# --- Odds Ratio ---
cat("\n✅ Rasio Odds (Odds Ratio) Model:\n")
print(exp(coef(model_multinom)))

# ======================================================
# 🔹 D. PREDIKSI DAN CONFUSION MATRIX
# ======================================================
prediksi_kelas_test <- predict(model_multinom, newdata = data_testing, type = "class")
aktual_kelas_test <- data_testing$Status_Kategori

conf_matrix_multinom <- confusionMatrix(data = prediksi_kelas_test, reference = aktual_kelas_test)
print(conf_matrix_multinom)

if(!require(ggplot2)) install.packages("ggplot2")
if(!require(reshape2)) install.packages("reshape2")

library(ggplot2)
library(reshape2)

# Ubah confusion matrix ke data frame
cm_df <- as.data.frame(conf_matrix_multinom$table)
colnames(cm_df) <- c("Aktual", "Prediksi", "Freq")

# Plot heatmap confusion matrix
ggplot(cm_df, aes(x = Prediksi, y = Aktual, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "black", size = 5) +
  scale_fill_gradient(low = "#f0f0f0", high = "#0073C2FF") +
  labs(
    title = "📊 Confusion Matrix Model Multinomial Logistic Regression",
    x = "Kelas Prediksi",
    y = "Kelas Aktual",
    fill = "Frekuensi"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    axis.text = element_text(face = "bold")
  )


# ======================================================
# 🔹 E. HITUNG METRIK TAMBAHAN (Precision, Recall, F1)
# ======================================================
conf_table <- conf_matrix_multinom$table

precision <- diag(conf_table) / colSums(conf_table)
recall <- diag(conf_table) / rowSums(conf_table)
f1 <- 2 * ((precision * recall) / (precision + recall))

eval_metrics <- data.frame(
  Kelas = names(precision),
  Precision = round(precision, 4),
  Recall = round(recall, 4),
  F1_Score = round(f1, 4)
)

# Rata-rata keseluruhan
support <- rowSums(conf_table)
macro_precision <- mean(precision, na.rm = TRUE)
macro_recall <- mean(recall, na.rm = TRUE)
macro_f1 <- mean(f1, na.rm = TRUE)
weighted_precision <- sum(precision * support, na.rm = TRUE) / sum(support)
weighted_recall <- sum(recall * support, na.rm = TRUE) / sum(support)
weighted_f1 <- sum(f1 * support, na.rm = TRUE) / sum(support)

overall_metrics <- data.frame(
  Tipe = c("Macro Average", "Weighted Average"),
  Precision = round(c(macro_precision, weighted_precision), 4),
  Recall = round(c(macro_recall, weighted_recall), 4),
  F1_Score = round(c(macro_f1, weighted_f1), 4)
)

cat("\n=== Evaluasi Model ===\n")
print(eval_metrics)
cat("\n=== Evaluasi Keseluruhan ===\n")
print(overall_metrics)

akurasi_test <- conf_matrix_multinom$overall['Accuracy']
cat("\nOverall Akurasi Model (Test Set):", sprintf("%.4f", akurasi_test), "\n")

# ======================================================
# 🔹 F. EVALUASI STATISTIK: Log-Likelihood, AIC, BIC, Pseudo-R²
# ======================================================
# Log-Likelihood Model
LL_model <- as.numeric(logLik(model_multinom))

# Model Null (tanpa prediktor)
model_null <- multinom(Status_Kategori ~ 1, data = data_training)
LL_null <- as.numeric(logLik(model_null))

# Hitung AIC dan BIC
AIC_model <- AIC(model_multinom)
BIC_model <- BIC(model_multinom)

# Hitung Pseudo R² (McFadden)
R2_McFadden <- 1 - (LL_model / LL_null)

# Tampilkan hasil evaluasi statistik
cat("\n==============================================\n")
cat("📊 EVALUASI STATISTIK MODEL MULTINOMIAL\n")
cat("==============================================\n")
cat(sprintf("Log-Likelihood Model     : %.4f\n", LL_model))
cat(sprintf("Log-Likelihood Null Model: %.4f\n", LL_null))
cat(sprintf("AIC                      : %.4f\n", AIC_model))
cat(sprintf("BIC                      : %.4f\n", BIC_model))
cat(sprintf("Pseudo R² (McFadden)     : %.4f\n", R2_McFadden))

# ======================================================
# 🔹 G. INTERPRETASI PROBABILITAS
# ======================================================
probabilitas <- predict(model_multinom, newdata = data_training, type = "probs")
data_training$Prob_Sehat <- probabilitas[, "Sehat"]
data_training$Prob_AbuAbu <- probabilitas[, "Abu_Abu"]
data_training$Prob_Bangkrut <- 1 - (data_training$Prob_Sehat + data_training$Prob_AbuAbu)

head(data_training[, c("Status_Kategori", "Prob_Bangkrut", "Prob_AbuAbu", "Prob_Sehat")])

# ======================================================
# 🔹 H. TABEL HASIL PREDIKSI AKHIR
# ======================================================

# Buat data frame hasil prediksi dari data_testing
hasil_prediksi <- data_testing

# Tambahkan kolom prediksi
hasil_prediksi$Pred_Label <- prediksi_kelas_test

# Tambahkan kolom status benar/salah
hasil_prediksi$Status_Prediksi <- ifelse(
  hasil_prediksi$Status_Kategori == hasil_prediksi$Pred_Label,
  "✅ Benar",
  "❌ Salah"
)

# Pilih kolom yang ingin ditampilkan
hasil_tampil <- hasil_prediksi[, c("X1", "X2", "X3", "X4", 
                                   "Status_Kategori", "Pred_Label", "Status_Prediksi")]

cat("\n==============================================\n")
cat("📋 TABEL HASIL PREDIKSI\n")
cat("==============================================\n")

View(hasil_tampil)

# --- Ringkasan kesalahan per kelas ---
ringkasan_salah <- hasil_prediksi |>
  dplyr::group_by(Status_Kategori) |>
  dplyr::summarise(Jumlah_Salah = sum(Status_Prediksi == "❌ Salah"))

cat("\n==============================================\n")
cat("📊 RINGKASAN KESALAHAN PREDIKSI PER KELAS\n")
cat("==============================================\n")
print(ringkasan_salah)

