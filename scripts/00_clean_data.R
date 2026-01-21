# ==============================================================================
# Data Cleaning Script
# Processes raw Excel files into clean CSV format
# ==============================================================================

library(readxl)
library(tidyverse)
library(janitor)

path_data <- "data"

cat("="*60, "\n")
cat("DATA CLEANING PIPELINE\n")
cat("="*60, "\n")

# ==============================================================================
# BROMP Data Cleaning
# ==============================================================================

cat("\nCleaning BROMP data...\n")

# Loading data
d <- read_excel(here::here(path_data, 'BROMP_sorted_by_studentid.xlsx')) %>% 
  clean_names()

cat("  Loaded", nrow(d), "raw observations\n")

# Re-classing variables
options(digits.secs = 3)

d <- d %>% 
  mutate(
    ntp_time = lubridate::mdy_hms(ntp_time, tz="America/New_York"),
    ntp_time = ntp_time + lubridate::milliseconds(msoffsetfromstart),
    behavior = as.factor(behavior),
    affect = as.factor(affect)
  )

# Remove redundant columns
d <- d %>% select(-c(ntp_time_ms, notes))

# Handling missing data
## Remove 2 students with missing values
cat("  Removing students with missing data...\n")
d <- d %>% tidyr::drop_na()

# Remove invalid observations
## Remove "skipped" observation
cat("  Removing invalid observations...\n")
initial_count <- nrow(d)
d <- d %>% filter(behavior != "skipped")
d <- d %>% mutate(behavior = droplevels(behavior))
cat("    - Removed", initial_count - nrow(d), "skipped observations\n")

## Remove "$" in behavior column
initial_count <- nrow(d)
d <- d %>% filter(behavior != "$")
d <- d %>% mutate(behavior = droplevels(behavior))
cat("    - Removed", initial_count - nrow(d), "observations with '$' in behavior\n")

## Remove "?" in affect column
initial_count <- nrow(d)
d <- d %>% filter(affect != "?")
d <- d %>% mutate(affect = droplevels(affect))
cat("    - Removed", initial_count - nrow(d), "observations with '?' in affect\n")

# Final counts
cat("  Final BROMP data:", nrow(d), "observations\n")
cat("  Unique behaviors:", length(unique(d$behavior)), "\n")
cat("  Unique affects:", length(unique(d$affect)), "\n")

# Save cleaned data
write_csv(d, file.path(path_data, "BROMP-clean.csv"))
cat("File saved to:", file.path(path_data, "BROMP-clean.csv"), "\n")

# ==============================================================================
# Caliper Data Cleaning
# ==============================================================================

cat("\nCleaning Caliper data...\n")

# Loading data
d <- read_excel(here::here(path_data, 'caliper-bromp-data-2.xlsx')) %>% 
  clean_names()

cat("  Loaded", nrow(d), "raw events\n")

# Re-classing variables
d <- d %>% 
  mutate(
    event_time = lubridate::ymd_hms(event_time, tz="UTC"),
    event_type = as.factor(event_type),
    object_type = as.factor(object_type),
    action = as.factor(action),
    stream_name = as.factor(stream_name),
    pause_reason = as.factor(pause_reason)
  )

# Make action ordinal
d <- d %>% 
  mutate(
    action = fct_relevel(action, "Started", "Resumed", "Paused", "Completed", "Ended")
  )

# Validation checks
cat("  Running validation checks...\n")

# Check caliper_event_id format
invalid_event_ids <- sum(!grepl("^urn:uuid:[0-9a-f-]{36}$", d$caliper_event_id))
if (invalid_event_ids > 0) {
  cat("WARNING:", invalid_event_ids, "invalid caliper_event_id formats\n")
}

# Check student_id format
invalid_student_ids <- sum(nchar(d$student_id_anon) != 16 | !grepl("^stu_", d$student_id_anon))
if (invalid_student_ids > 0) {
  cat("WARNING:", invalid_student_ids, "invalid student_id formats\n")
}

# Check video_id for VideoObjects
missing_video_ids <- sum(d$object_type == "VideoObject" & is.na(d$video_id), na.rm=TRUE)
if (missing_video_ids > 0) {
  cat("    ⚠ Note:", missing_video_ids, "VideoObjects missing video_id\n")
}

# Check assessment_id for Assessments
missing_assessment_ids <- sum(d$object_type == "Assessment" & is.na(d$assessment_id), na.rm=TRUE)
if (missing_assessment_ids > 0) {
  cat("WARNING:", missing_assessment_ids, "Assessments missing assessment_id\n")
}

# Check pause_reason for Paused events
missing_pause_reasons <- sum(d$action == "Paused" & is.na(d$pause_reason), na.rm=TRUE)
if (missing_pause_reasons > 0) {
  cat("WARNING:", missing_pause_reasons, "Paused events missing pause_reason\n")
}

# Check current_time validity
invalid_current_time <- sum(d$current_time < 0 | d$current_time > 10000, na.rm=TRUE)
if (invalid_current_time > 0) {
  cat("WARNING:", invalid_current_time, "invalid current_time values\n")
}

# Remove duplicates
cat("  Checking for duplicates...\n")
duplicate_count <- sum(duplicated(d$caliper_event_id))
if (duplicate_count > 0) {
  cat("    - Found", duplicate_count, "duplicate events\n")
  d <- d %>% distinct(caliper_event_id, .keep_all = TRUE)
  cat("    - Removed duplicates\n")
}

# Final counts
cat("  Final Caliper data:", nrow(d), "events\n")
cat("  Unique students:", length(unique(d$student_id_anon)), "\n")
cat("  Unique event types:", length(unique(d$event_type)), "\n")

# Missingness summary
cat("\n  Missingness summary:\n")
cat("    - pause_reason:", sum(is.na(d$pause_reason)), "missing (", 
    round(sum(is.na(d$pause_reason))/nrow(d)*100, 1), "%)\n")
cat("    - current_time:", sum(is.na(d$current_time)), "missing (", 
    round(sum(is.na(d$current_time))/nrow(d)*100, 1), "%)\n")
cat("    - video_id:", sum(is.na(d$video_id)), "missing (", 
    round(sum(is.na(d$video_id))/nrow(d)*100, 1), "%)\n")
cat("    - assessment_id:", sum(is.na(d$assessment_id)), "missing (", 
    round(sum(is.na(d$assessment_id))/nrow(d)*100, 1), "%)\n")



write_csv(d, file.path(path_data, "caliper-clean.csv"))
cat("File saved to:", file.path(path_data, "caliper-clean.csv"), "\n")

# ==============================================================================
# Summary
# ==============================================================================

cat("\n", "="*60, "\n")
cat("DATA CLEANING COMPLETE\n")
cat("="*60, "\n")
cat("Output files:\n")
cat("  - BROMP-clean.csv\n")
cat("  - caliper-clean.csv\n")
cat("="*60, "\n")