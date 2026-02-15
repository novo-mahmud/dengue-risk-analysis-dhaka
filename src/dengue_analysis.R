# =============================================================
# PROJECT: DENGUE RISK ANALYSIS IN DHAKA
# AUTHOR: Md. Mahmudul Hasan Novo
# TOOLS: R (tidyverse, ggplot2, randomForest)
# VERSION: 3.0 (FINAL WITH ADMINISTRATIVE LEVEL ANALYSIS)
# STATUS: RESEARCH GRADE / PORTFOLIO READY
# =============================================================

# ---------------------------------------------------------
# STEP 0: SETUP
# ---------------------------------------------------------

# Check if tidyverse is installed, if not, install it
if(!require("tidyverse")) install.packages("tidyverse")
library(tidyverse)

# 1. Read CSV file from cloud environment
df <- read_csv("dengue_dataset.csv")

# 2. Verify import worked
# Check structure (Columns and Data Types)
glimpse(df)

# 3. Check first few rows of raw data
head(df)

# ---------------------------------------------------------
# STEP 1: DATA CLEANING
# ---------------------------------------------------------

# Dropping Medical Leakage Variables
# Reason: We want to find RISK FACTORS (Infrastructure), not medical DIAGNOSIS (Blood tests).
df_clean <- df %>% 
  select(Age, Gender, Area, AreaType, HouseType, Outcome)

# Inspect new structure
glimpse(df_clean)

# Convert text columns into categorical factors
# Reason: R needs factors to treat these as groups, not words.
df_clean <- df_clean %>%
  mutate(
    Gender   = as.factor(Gender),
    AreaType = as.factor(AreaType),
    HouseType = as.factor(HouseType),
    Outcome  = as.factor(Outcome)
  )

# Verify conversion
str(df_clean)

# Relevel Outcome so '1' (Infected) is reference target
# Reason: We want to model probability of BEING sick.
df_clean$Outcome <- relevel(df_clean$Outcome, ref = "1")

# Verify switch
str(df_clean$Outcome)

# ---------------------------------------------------------
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ---------------------------------------------------------

# 1. Infrastructure vs Dengue Risk
ggplot(df_clean, aes(x = HouseType, fill = Outcome)) +
  geom_bar(position = "fill") + # Converts counts to percentages (0-100%)
  scale_y_continuous(labels = scales::percent) + # Formats Y-axis as %
  scale_fill_manual(values = c("red3", "gray70"), # 1=Red (Infected), 0=Gray (Safe)
                    name = "Dengue Test Result",
                    labels = c("Negative (0)", "Positive (1)")) +
  theme_minimal() +
  labs(title = "Dengue Prevalence by Housing Infrastructure",
       subtitle = "Analysis of 1,000 Patients in Dhaka",
       x = "Type of House",
       y = "Infection Rate (%)")

# 2. Urban Development Status vs Dengue Risk
ggplot(df_clean, aes(x = AreaType, fill = Outcome)) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = scales::percent) +
  scale_fill_manual(values = c("red3", "gray70")) +
  theme_minimal() +
  labs(title = "Dengue Prevalence by Area Development Status",
       subtitle = "Developed vs. Undeveloped Areas",
       x = "Area Type",
       y = "Infection Rate (%)")

# 3. Age Distribution of Patients
ggplot(df_clean, aes(x = Age, fill = Outcome)) +
  geom_histogram(binwidth = 5, position = "identity", alpha = 0.7) +
  scale_fill_manual(values = c("red3", "gray70")) +
  theme_minimal() +
  labs(title = "Age Distribution of Dengue Cases",
       x = "Age (Years)",
       y = "Count of Patients")

# ---------------------------------------------------------
# STEP 3: MODEL 1 - INFRASTRUCTURE (NULL RESULT)
# ---------------------------------------------------------

# Fit model
model <- glm(Outcome ~ Age + Gender + AreaType + HouseType, 
             data = df_clean, 
             family = "binomial")

# View statistical summary
# Expectation: High P-Values (>0.05) for all variables. Hypothesis rejected.
summary(model)

# ---------------------------------------------------------
# STEP 4: MODEL 2 - GEOGRAPHIC HOTSPOTS (PIVOT)
# ---------------------------------------------------------

# LOGIC: Since Housing Type was not significant, we pivot to 'Location'.
# We investigate if specific Areas (Neighborhoods) are risk drivers.

# Step A: Select Robust Baseline (Automated)
# Reason: We want a baseline with High Sample Size for statistical stability.
area_counts <- table(df_clean$Area)
area_sorted <- sort(area_counts, decreasing = TRUE)
new_baseline <- names(area_sorted)[1] # Selects area with most patients

# Print chosen baseline for logs
print(paste("Most Robust Baseline Selected:", new_baseline))

# Step B: Convert 'Area' from Text to Factor
df_clean$Area <- as.factor(df_clean$Area)

# Step C: Apply Robust Baseline to 'Area' column
# This overwrites alphabetical default with our calculated robust baseline.
df_clean$Area <- relevel(df_clean$Area, ref = new_baseline)

# Step D: Run new Logistic Regression model
# Formula: Outcome depends ONLY on Area
geo_model_v2 <- glm(Outcome ~ Area, 
                    data = df_clean, 
                    family = "binomial")

# View new summary
# Expectation: Specific areas like Sutrapur, Ramna, Banasree should be Significant (p < 0.05).
summary(geo_model_v2)

# Step E: Calculate Odds Ratios for NEW Model (v2)
# We use v2 variables exclusively to avoid mixing with old model.
geo_odds_v2 <- exp(coef(geo_model_v2))

# Calculate 95% Confidence Intervals for v2
geo_ci_v2 <- exp(confint(geo_model_v2))

# Create Results Table
geo_results_v2 <- cbind(Odds_Ratio = geo_odds_v2, 
                        CI_Lower = geo_ci_v2[,1], 
                        CI_Upper = geo_ci_v2[,2])

# Round and Print
print(round(geo_results_v2, 2))

# ---------------------------------------------------------
# LEVEL 2: MACHINE LEARNING VALIDATION (RANDOM FOREST)
# =============================================================
# Reason: To validate Logistic Regression findings (Hotspots) using non-linear methods.
# Method: Random Forest (Ensemble Learning).

# ---------------------------------------------------------
# STEP 5: PREPARATION (ML)
# ---------------------------------------------------------

# Install Random Forest (Machine Learning Algorithm)
if(!require("randomForest")) install.packages("randomForest")
library(randomForest)

# Load tidyverse for plotting (already loaded, but safe to recall)
library(tidyverse)

# Set Seed for Reproducibility
# CRITICAL: ML models use randomness; we lock it so results stay same.
set.seed(123)

# ---------------------------------------------------------
# STEP 6: TRAINING RANDOM FOREST
# ---------------------------------------------------------

# Train Random Forest model
# ntree: 500 (Standard number of trees for stability)
# importance: TRUE (Crucial: We need to see which variables matter most)
rf_model <- randomForest(Outcome ~ Area + Age + Gender + HouseType, 
                         data = df_clean, 
                         ntree = 500, 
                         importance = TRUE)

# Print model summary (Displays OOB error and Confusion Matrix)
print(rf_model)

# ---------------------------------------------------------
# STEP 7: MODEL METRICS (PERFORMANCE)
# ---------------------------------------------------------

# 1. Get "Out of Bag" (OOB) predictions
# This is the standard way to measure accuracy in Random Forest without a separate test set.
pred_oob <- predict(rf_model, type = "response")

# 2. Create a Confusion Matrix
# Compares Predictions vs Reality
conf_matrix <- table(Predicted = pred_oob, Actual = df_clean$Outcome)

# 3. Calculate and Print Accuracy
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
print(paste("Model Accuracy (OOB Estimate):", round(accuracy * 100, 2), "%"))

# ---------------------------------------------------------
# STEP 8: VARIABLE IMPORTANCE ANALYSIS
# ---------------------------------------------------------

# Extract importance data
# MeanDecreaseGini: How much does 'purity' of nodes improve when we split on this variable?
importance_data <- importance(rf_model)

# Print importance to console
print(importance_data)

# ---------------------------------------------------------
# STEP 9: VISUALIZATION (VARIABLE IMPORTANCE)
# ---------------------------------------------------------

# Convert importance to a dataframe for plotting
imp_df <- as.data.frame(importance_data)

# Clean up names for professional look
colnames(imp_df) <- c("Variable", "MeanDecreaseGini")

# Plot Variable Importance (CORRECTED CODE)
# Fix: Changed fill = MeanDecreaseGini (Number) to fill = factor(Variable) (Name)
# Reason: 'scale_fill_brewer' needs Categories/Text, not Numbers.
ggplot(imp_df, aes(x = reorder(Variable, MeanDecreaseGini), 
                   y = MeanDecreaseGini, 
                   fill = factor(Variable))) + # <--- CRITICAL FIX HERE
  geom_bar(stat = "identity") + 
  coord_flip() + 
  scale_fill_brewer(palette = "Set2") + # Now works perfectly
  theme_minimal() +
  labs(
    title = "Variable Importance in Dengue Prediction",
    subtitle = "Random Forest Model (500 Trees)",
    x = "Variables",
    y = "Mean Decrease in Gini (Importance)",
    caption = "Higher values indicate stronger predictive power"
  )

# ---------------------------------------------------------
# STEP 10: VISUALIZATION (FOREST PLOT - FINAL SUMMARY)
# ---------------------------------------------------------

# Prepare Data for Plotting (Clean Version)
# 1. Convert results table to a dataframe
plot_data <- as.data.frame(geo_results_v2)

# 2. Extract Area Names from Row Names
plot_data$Area_Name <- rownames(plot_data)

# 3. Clean names (Remove "Area" prefix from EVERY name)
plot_data$Area_Name <- gsub("Area", "", plot_data$Area_Name)

# 4. Rename 'Intercept' to 'Demra (Baseline)' for clarity
plot_data$Area_Name <- ifelse(plot_data$Area_Name == "(Intercept)", 
                              "Demra (Baseline)", 
                              plot_data$Area_Name)

# 5. Order data: Highest risk (Top) to Lowest risk (Bottom)
plot_data <- plot_data[order(plot_data$Odds_Ratio), ]

# 6. Re-factor names so that plot order sticks
plot_data$Area_Name <- factor(plot_data$Area_Name, levels = plot_data$Area_Name)

# 7. Generate Final Plot
# Load library (Redundant but safe)
library(tidyverse)

ggplot(plot_data, aes(x = Odds_Ratio, y = Area_Name)) +
  # Add Error Bars (Confidence Intervals)
  geom_errorbar(aes(xmin = CI_Lower, xmax = CI_Upper), 
                orientation = "y", # Modern syntax for horizontal bars
                width = 0.2, color = "darkgray") +
  # Add Points (Odds Ratios)
  geom_point(size = 3, color = "steelblue") +
  # Add "Line of No Risk" (Odds Ratio = 1.0)
  geom_vline(xintercept = 1, linetype = "dashed", color = "red", linewidth = 1) +
  theme_minimal() +
  labs(
    title = "Geographic Risk of Dengue Infection in Dhaka",
    subtitle = "Odds Ratios (OR) with 95% Confidence Intervals",
    x = "Odds Ratio (Risk)",
    y = "Area of Dhaka",
    caption = "Baseline: Demra (Safe Zone) | Red Line = No Increased Risk"
  )


# =============================================================
# =============================================================
# LEVEL 3: ADMINISTRATIVE LEVEL ANALYSIS (THANA)
# =============================================================

# ---------------------------------------------------------
# STEP 1: CREATE THANA MAPPING DICTIONARY
# ---------------------------------------------------------

# Create mapping table using data.frame() to explicitly name columns
# This solves the "Must specify column" error.
thana_lookup <- data.frame(
  Area = c("Adabor", "Badda", "Bangshal", "Banasree", "Biman Bandar", "Bosila", 
           "Cantonment", "Chawkbazar", "Dhanmondi", "Demra", "Gendaria", "Gulshan", 
           "Hazaribagh", "Jatrabari", "Kadamtali", "Kafrul", "Kalabagan", 
           "Kamrangirchar", "Keraniganj", "Khilgaon", "Khilkhet", "Lalbagh", 
           "Mirpur", "Mohammadpur", "Motijheel", "New Market", "Pallabi", 
           "Paltan", "Ramna", "Rampura", "Sabujbagh", "Shahbagh", 
           "Sher-e-Bangla Nagar", "Shyampur", "Sutrapur", "Tejgaon"),
  Thana = c("Adabor Thana", "Badda Thana", "Bangshal Thana", "Rampura Thana", 
            "Biman Bandar Thana", "Bosila Thana", "Cantonment Thana", "Chawkbazar Thana",
            "Dhanmondi Thana", "Demra Thana", "Gendaria Thana", "Gulshan Thana", 
            "Hazaribagh Thana", "Jatrabari Thana", "Kadamtali Thana", "Kafrul Thana", 
            "Kalabagan Thana", "Kamrangirchar Thana", "Keraniganj Thana", 
            "Khilgaon Thana", "Khilkhet Thana", "Lalbagh Thana", "Mirpur Thana", "Mohammadpur Thana", 
            "Motijheel Thana", "New Market Thana", "Pallabi Thana", "Paltan Thana", "Ramna Thana", 
            "Rampura Thana", "Sabujbagh Thana", "Shahbagh Thana", "Sher-e-Bangla Nagar Thana", 
            "Shyampur Thana", "Sutrapur Thana", "Tejgaon Thana")
)

# Optional: Verify the lookup table is created
print(thana_lookup)


# ---------------------------------------------------------
# STEP 2: MERGE DATA (DATA ENRICHMENT)
# ---------------------------------------------------------

# Join Mapping to Clean Dataset
# 'left_join' keeps ALL patients. If an Area isn't in map, it gets NA for Thana.
df_merged <- left_join(df_clean, thana_lookup, by = "Area")

# Inspect new structure (Should see new 'Thana' column)
glimpse(df_merged)

# ---------------------------------------------------------
# STEP 3: CALCULATE THANA STATISTICS (RE-RUN TO FIX "OBJECT NOT FOUND" ERROR)
# ---------------------------------------------------------

# 1. Aggregate by Thana AND Sort immediately
# Using 'arrange()' is safer and cleaner than subsetting tibbles.
thanx_stats_sorted <- df_merged %>%
  group_by(Thana) %>%
  summarise(
    Total_Patients = n(),
    Total_Cases = sum(as.numeric(Outcome)),
    Infection_Rate = (sum(as.numeric(Outcome)) / n()) * 100
  ) %>%
  arrange(desc(Infection_Rate)) # Sort Highest Risk to Lowest Risk (Desc)

# 2. Verify Output
print(thanx_stats_sorted)

# ---------------------------------------------------------
# STEP 4: VISUALIZATION (THANA HEATMAP - FINAL SUMMARY)
# ---------------------------------------------------------

# 1. Safety Check: Does the table variable exist?
# Note: This prevents the "Object not found" error if variable names mismatch.
if(exists("thanx_stats_sorted")) {
  
  # 2. Sort by Infection Rate (Descending) for better visual
  # We use 'arrange()' to create a fresh ordered dataframe
  plot_data_final <- thanx_stats_sorted %>% 
    arrange(desc(Infection_Rate))
  
  # 3. Convert Thana to a Factor (Cleanest way to set X-axis order)
  # Note: We keep the object name 'thanx_stats_sorted' to be safe
  plot_data_final$Thana <- factor(plot_data_final$Thana, 
                                  levels = plot_data_final$Thana)
  
  # 4. Generate Heatmap (Spatial Proxy)
  # We use 'geom_tile' which creates a grid of rectangles. 
  # This visualizes "Risk" across different regions.
  ggplot(plot_data_final, aes(x = Thana, y = Infection_Rate, fill = Infection_Rate)) +
    geom_tile(color = "white") + 
    
    # Use a gradient color scale (White = Low Risk, Red = High Risk)
    scale_fill_gradient(low = "blue", high = "red") +
    
    theme_minimal() +
    labs(
      title = "Regional Dengue Risk Heatmap (Thana Level)",
      subtitle = "Aggregated Analysis by Administrative Districts",
      x = "Administrative Thana (Level 3)",
      y = "Dengue Infection Rate (%)",
      caption = "Note: Data aggregated by Thana; represents generalized regional risk."
    )
  
} else {
  
  # Error Handling if object is missing
  print("ERROR: Table 'thanx_stats_sorted' not found. Please run STEP 3 Calculation block first.")
}




# 1. Save the Thana Stats table to a CSV file in your Cloud environment
write.csv(thana_stats_sorted, "thana_stats.csv")

# 2. Confirm it worked
list.files()