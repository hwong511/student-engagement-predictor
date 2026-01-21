# ==============================================================================
# EXACT COPY FROM: ho_data_merge_bromp_caliper.qmd
# Data Merging Notebook for BROMP and Caliper
# Author: Carnegie Learning Capstone Group
# ==============================================================================

# Setup ------------------------------------------------------------------------

library(readr)
library(dplyr)
library(lubridate)
library(tidyr)
library(janitor)
library(fuzzyjoin) 
library(data.table)

path_data <- "data"

# Data Merging -----------------------------------------------------------------

# Loading in the Data

d1 <- read_csv(here::here(path_data, 'BROMP-clean.csv')) %>%  janitor::clean_names()
d2 <- read_csv(here::here(path_data, 'caliper-clean.csv')) %>% janitor::clean_names()

# For the sake of ease, I'm gonna rename both of the student IDs in the two spreadsheet to the same name.

d1 <- d1 %>% rename(student_id = anon_student_id)
d2 <- d2 %>% rename(student_id = student_id_anon)

# Cleaning & Deciding Merge Approach

length(intersect(d1$student_id, d2$student_id)) # Number of student_ids in common
length(unique(d1$student_id)) # Number of students for the BROMP dataset
length(unique(d2$student_id)) # Number of students for the caliper dataset

# ANYWAYS, let me clean the timestamps real quick...

d1 <- d1 %>%
  mutate(
    time_utc = with_tz(ntp_time, "UTC")
  ) %>%
  filter(!is.na(time_utc), !is.na(student_id))

d2 <- d2 %>%
  mutate(
    time_utc = ymd_hms(event_time)
  ) %>%
  filter(!is.na(time_utc), !is.na(student_id))

# Merging via Time Windows

# Note:
# For `time_diff`: 
# - Positive = Caliper event happened AFTER BROMP observation
# - Negative = Caliper event happened BEFORE BROMP observation

# The following code chunk gives us a merged dataset with SYMMETRIC features. 
# We shouldn't use this to train the model, but if we want to, we can compare 
# our final model with it to examine the differences.

d_merged <- d1 %>%
  fuzzy_left_join(d2, 
                  by = c("student_id" = "student_id", "time_utc" = "time_utc"),
                  match_fun = list(`==`, function(x, y) {
                    abs(difftime(x, y, units = "secs")) <= 90
                  })) %>%
  rename(bromp_time = time_utc.x, caliper_time = time_utc.y) %>%
  mutate(student_id = student_id.x,
         time_diff = as.numeric(difftime(caliper_time, bromp_time, units = "secs")))

# For d_features, which we will use to train the models, we will create feature 
# separately for before and after.

d_features <- d_merged %>%
  group_by(student_id, bromp_time, behavior, affect, class) %>%
  summarise(
    # BACKWARD window features (for prediction)
    events_back_30s = sum(!is.na(caliper_time) & time_diff >= -30 & time_diff < 0),
    events_back_60s = sum(!is.na(caliper_time) & time_diff >= -60 & time_diff < 0),
    events_back_90s = sum(!is.na(caliper_time) & time_diff >= -90 & time_diff < 0),
    
    pauses_back_60s = sum(!is.na(caliper_time) & time_diff >= -60 & time_diff < 0 & 
                          action == "Paused", na.rm = TRUE),
    assessment_back_60s = sum(!is.na(caliper_time) & time_diff >= -60 & time_diff < 0 & 
                              event_type == "AssessmentEvent", na.rm = TRUE),
    
    # FORWARD window features (for understanding context)
    events_forward_30s = sum(!is.na(caliper_time) & time_diff > 0 & time_diff <= 30),
    events_forward_60s = sum(!is.na(caliper_time) & time_diff > 0 & time_diff <= 60),
    
    # SYMMETRIC window features (for full context)
    events_symmetric_60s = sum(!is.na(caliper_time) & abs(time_diff) <= 60),
    events_symmetric_90s = sum(!is.na(caliper_time) & abs(time_diff) <= 90),
    
    pauses_symmetric_60s = sum(!is.na(caliper_time) & abs(time_diff) <= 60 & 
                               action == "Paused", na.rm = TRUE),
    
    # Most recent event (backward only)
    time_since_last_event = ifelse(
      any(!is.na(caliper_time) & time_diff < 0),
      abs(max(time_diff[!is.na(caliper_time) & time_diff < 0])),
      600
    ),
    
    most_recent_event_type = ifelse(
      any(!is.na(caliper_time) & time_diff < 0),
      event_type[which.max(time_diff[!is.na(caliper_time) & time_diff < 0])],
      "NO_ACTIVITY"
    ),
    
    .groups = 'drop'
  )

write.csv(d_merged, "+-90sec_merged_data.csv")

write.csv(d_features, "prediction_features.csv")







# Validating the merge (just in case) ------------------------------------------

# Basic Counts and Coverage

cat("=== BASIC VALIDATION ===\n")

# Original dataset sizes
cat("Original BROMP observations:", nrow(d1), "\n")
cat("Original Caliper events:", nrow(d2), "\n")
cat("Merged dataset size:", nrow(d_merged), "\n")
cat("Expansion ratio:", round(nrow(d_merged) / nrow(d1), 2), "x\n\n")

# Student coverage
students_d1 <- unique(d1$student_id)
students_d2 <- unique(d2$student_id)
students_merged <- unique(d_merged$student_id)

cat("Students in BROMP:", length(students_d1), "\n")
cat("Students in Caliper:", length(students_d2), "\n")
cat("Students in merged data:", length(students_merged), "\n")
cat("% BROMP students in merge:", round(100 * length(students_merged) / length(students_d1), 1), "%\n\n")

# Check for missing students
missing_students <- setdiff(students_d1, students_merged)
if (length(missing_students) > 0) {
  cat("WARNING: Students from BROMP missing in merge:", length(missing_students), "\n")
  cat("Missing student IDs:", head(missing_students, 5), "\n\n")
}

# Time Difference Validation

cat("=== TIME ALIGNMENT VALIDATION ===\n")

# Time difference statistics
time_diff_stats <- d_merged %>%
  summarise(
    mean_diff = round(mean(abs(time_diff)), 2),
    median_diff = round(median(abs(time_diff)), 2),
    q25_diff = round(quantile(abs(time_diff), 0.25), 2),
    q75_diff = round(quantile(abs(time_diff), 0.75), 2),
    max_diff = round(max(abs(time_diff)), 2),
    within_1min = round(100 * mean(abs(time_diff) <= 60), 1),
    within_2min = round(100 * mean(abs(time_diff) <= 120), 1),
    within_5min = round(100 * mean(abs(time_diff) <= 300), 1)
  )

print(time_diff_stats)
cat("\n")

# Check for systematic time bias
pos_neg_bias <- d_merged %>%
  summarise(
    positive_diffs = sum(time_diff > 0),
    negative_diffs = sum(time_diff < 0),
    mean_signed_diff = round(mean(time_diff), 2)
  )

cat("Positive time diffs (Caliper after BROMP):", pos_neg_bias$positive_diffs, "\n")
cat("Negative time diffs (Caliper before BROMP):", pos_neg_bias$negative_diffs, "\n")
cat("Mean signed difference:", pos_neg_bias$mean_signed_diff, "seconds\n")

if (abs(pos_neg_bias$mean_signed_diff) > 30) {
  cat("WARNING: Systematic time bias detected!\n")
}
cat("\n")

# Temporal Distribution Check

cat("=== TEMPORAL DISTRIBUTION CHECK ===\n")

# Check if merge preserves temporal patterns
bromp_by_hour <- d1 %>%
  mutate(hour = hour(time_utc)) %>%
  count(hour, name = "bromp_count")

merged_by_hour <- d_merged %>%
  mutate(hour = hour(bromp_time)) %>%
  count(hour, name = "merged_count")

temporal_check <- bromp_by_hour %>%
  left_join(merged_by_hour, by = "hour") %>%
  mutate(
    coverage_pct = round(100 * (merged_count / bromp_count), 1)
  )

cat("Hourly coverage check:\n")
print(temporal_check)
cat("\n")

# Check for hours with poor coverage
poor_coverage <- temporal_check %>%
  filter(coverage_pct < 50)

if (nrow(poor_coverage) > 0) {
  cat("WARNING: Poor coverage in these hours:\n")
  print(poor_coverage)
  cat("\n")
}