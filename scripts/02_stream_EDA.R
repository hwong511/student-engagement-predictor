# Setup ------------------------------------------------------------------------

library(dplyr)
library(readr)
library(skimr)
library(tidyr)
library(lme4)
library(ggplot2)
library(corrplot)

path_data <- "data"

# Load Data --------------------------------------------------------------------

d <- read_csv(here::here(path_data, 'caliper-clean.csv')) %>% janitor::clean_names()

# Quick Data Overview
d %>% skim()

# Basic Cleaning ---------------------------------------------------------------

d <- d %>%
  filter(
    !is.na(stream_name),
    stream_name != "NA",
    !is.na(event_time),
    !is.na(student_id_anon)
  ) %>%
  mutate(event_time = as.POSIXct(event_time))

cat("Date range:", as.character(min(d$event_time)), "to", as.character(max(d$event_time)), "\n")

# Class Period Definition ------------------------------------------------------

# Note: Since event_time has a weird range spanning multiple days, we cannot
# calculate stream duration as first-to-last event across ALL students and ALL
# time periods. If the same stream was viewed in multiple class periods, the
# "duration" would span multiple days instead of actual class period duration.
#
# Solution: Create date-based class periods. ~96% of events fall within normal
# school hours (10:00-16:00 EST) when interpreted as Eastern Time (UTC-4),
# suggesting this is a reasonable categorization.

d <- d %>% mutate(class_period = as.Date(event_time))

# Stream-Class Characteristics -------------------------------------------------

# Calculate characteristics for each stream within each class period

stream_class_characteristics <- d %>%
  group_by(stream_name, class_period) %>%
  summarise(
    class_duration_minutes = as.numeric(difftime(max(event_time), min(event_time), units = "mins")),
    class_assessment_count = n_distinct(assessment_id[!is.na(assessment_id) & assessment_id != "NA"]),
    class_video_count = n_distinct(video_id[!is.na(video_id) & video_id != "NA"]),
    total_events = n(),
    unique_students = n_distinct(student_id_anon),
    unique_timestamps = n_distinct(event_time),
    .groups = 'drop'
  ) %>%
  # Filter out low-quality stream-class combos
  filter(
    total_events >= 3,
    unique_students >= 1,
    class_duration_minutes > 0,
    unique_timestamps >= 2
  )

# Remove administrative events (ANSWER_PASS, Answer Pass, NA values)
stream_class_characteristics <- stream_class_characteristics %>%
  filter(
    !stream_name %in% c("ANSWER_PASS", "Answer Pass"),
    !is.na(stream_name)
  )

# Stream-Level Characteristics -------------------------------------------------

# Aggregate across all class periods for each stream

stream_characteristics <- d %>%
  inner_join(stream_class_characteristics %>% select(stream_name), by = "stream_name") %>%
  group_by(stream_name) %>%
  summarise(
    stream_total_duration_minutes = as.numeric(difftime(max(event_time), min(event_time), units = "mins")),
    stream_total_assessments = n_distinct(assessment_id[!is.na(assessment_id) & assessment_id != "NA"]),
    stream_total_videos = n_distinct(video_id[!is.na(video_id) & video_id != "NA"]),
    stream_total_events = n(),
    stream_unique_students = n_distinct(student_id_anon),
    class_periods_offered = n_distinct(class_period),
    .groups = 'drop'
  ) %>%
  # Filter out low-quality streams
  filter(
    stream_total_events >= 10,
    stream_unique_students >= 2,
    stream_total_duration_minutes > 5
  )

# Student-Level Events with Session Detection ---------------------------------

# Key insight: Calculate engagement time by identifying separate learning
# sessions (10+ minute breaks = new session). This excludes long breaks like
# lunch from engagement time calculations.
#
# Algorithm:
# 1. Sort events chronologically for each student-stream-class combo
# 2. Calculate gaps between consecutive events
# 3. Gaps >10 minutes = session break
# 4. Calculate duration within each session
# 5. Sum session durations for total active time
# 6. Minimum 0.5 minutes for single-event sessions (arbitrary but reasonable)

student_level_events <- d %>%
  inner_join(
    stream_class_characteristics %>% select(stream_name, class_period),
    by = c("stream_name", "class_period")
  ) %>%
  group_by(student_id_anon, stream_name, class_period) %>%
  summarise(
    first_event = min(event_time),
    last_event = max(event_time),
    
    # Calculate active engagement time (excluding long gaps)
    time_spent_minutes = {
      times <- sort(event_time)
      if (length(times) <= 1) {
        0.5  # Minimum time for single event
      } else {
        gaps <- as.numeric(diff(times), units = "mins")
        long_gaps <- gaps > 10  # 10+ minute gaps = session break
        
        if (any(long_gaps)) {
          gap_positions <- which(long_gaps)
          session_starts <- c(1, gap_positions + 1)
          session_ends <- c(gap_positions, length(times))
          
          session_durations <- sapply(seq_along(session_starts), function(i) {
            session_times <- times[session_starts[i]:session_ends[i]]
            max(0.5, as.numeric(difftime(max(session_times), min(session_times), units = "mins")))
          })
          
          sum(session_durations)
        } else {
          max(0.5, as.numeric(difftime(max(times), min(times), units = "mins")))
        }
      }
    },
    
    session_count = {
      times <- sort(event_time)
      if (length(times) <= 1) {
        1
      } else {
        gaps <- as.numeric(diff(times), units = "mins")
        sum(gaps > 10) + 1
      }
    },
    
    assessments_attempted = n_distinct(assessment_id[!is.na(assessment_id) & assessment_id != "NA"]),
    student_total_events = n(),
    total_pauses = sum(action == "Paused", na.rm = TRUE),
    
    .groups = 'drop'
  )

# Calculate derived metrics
student_level_events <- student_level_events %>%
  mutate(
    pause_rate_per_minute = total_pauses / time_spent_minutes,
    event_rate_per_minute = student_total_events / time_spent_minutes
  )

# Merge with Stream Characteristics --------------------------------------------

data <- student_level_events %>%
  left_join(
    stream_class_characteristics %>%
      select(stream_name, class_period, class_duration_minutes,
             class_assessment_count, class_video_count),
    by = c("stream_name", "class_period")
  ) %>%
  left_join(
    stream_characteristics %>%
      select(stream_name, stream_total_duration_minutes, stream_total_assessments,
             stream_total_videos, stream_unique_students),
    by = "stream_name"
  )

# Calculate Engagement Metrics -------------------------------------------------

# Stream progress and completion
data <- data %>%
  mutate(
    stream_progress_pct = (time_spent_minutes / stream_total_duration_minutes) * 100,
    stream_progress_pct = pmin(stream_progress_pct, 100),  # Cap at 100%
    stream_completion_rate = assessments_attempted / stream_total_assessments
  )

# Student-level baselines
data <- data %>%
  group_by(student_id_anon) %>%
  mutate(
    student_avg_pause_rate = mean(pause_rate_per_minute, na.rm = TRUE),
    student_total_periods = n()
  ) %>%
  ungroup()

# Stream-level baselines
data <- data %>%
  group_by(stream_name) %>%
  mutate(
    stream_avg_pause_rate = mean(pause_rate_per_minute, na.rm = TRUE)
  ) %>%
  ungroup()

# Class-level baselines
data <- data %>%
  group_by(class_period) %>%
  mutate(
    class_avg_pause_rate = mean(pause_rate_per_minute, na.rm = TRUE)
  ) %>%
  ungroup()

# Relative Engagement Metrics --------------------------------------------------

# Key insight: Normalize pause rates to individual and stream baselines to
# separate content difficulty from engagement quality

data <- data %>%
  mutate(
    # Relative to personal baseline
    relative_pause_rate = pause_rate_per_minute / student_avg_pause_rate,
    
    # Relative to stream difficulty
    stream_engagement_ratio = pause_rate_per_minute / stream_avg_pause_rate,
    
    # Relative to class context
    class_engagement_ratio = pause_rate_per_minute / class_avg_pause_rate,
    
    # Relative to stream difficulty (alternative calculation)
    relative_stream_engagement = (pause_rate_per_minute - stream_avg_pause_rate) / 
                                   stream_avg_pause_rate
  )

# Temporal Features ------------------------------------------------------------

# Track student progression through class periods

data <- data %>%
  arrange(student_id_anon, class_period) %>%
  group_by(student_id_anon) %>%
  mutate(
    class_period_number = row_number(),
    is_first_class = (class_period_number == 1)
  ) %>%
  ungroup()

# Save Processed Data ----------------------------------------------------------

write_csv(data, file.path(path_data, "caliper_processed_data.csv"))
cat("✓ Saved: caliper_processed_data.csv\n")

# Exploratory Visualizations ---------------------------------------------------

cat("\n=== EXPLORATORY ANALYSIS ===\n\n")

# 1. Pause Rate vs Stream Completion
cat("1. Pause Rate vs Stream Completion:\n")
p1 <- ggplot(data, aes(x = stream_completion_rate, y = pause_rate_per_minute)) +
  geom_point(alpha = 0.3, color = "steelblue") +
  geom_smooth(method = "lm", color = "red", se = TRUE) +
  labs(
    title = "Pause Rate vs Stream Completion",
    x = "Stream Completion Rate (proportion)",
    y = "Pause Rate (pauses/minute)"
  ) +
  theme_minimal()
print(p1)

# 2. First Class vs Later Classes
cat("2. Temporal Patterns - First Class Effect:\n")
p2 <- ggplot(data, aes(x = is_first_class, y = pause_rate_per_minute, fill = is_first_class)) +
  geom_boxplot() +
  labs(
    title = "Pause Rate: First Class vs Later Classes",
    subtitle = "Do students pause differently in their first exposure?",
    x = "First Class?",
    y = "Pause Rate (pauses/minute)"
  ) +
  theme_minimal()
print(p2)

# 3. Progress Over Time
cat("3. Stream Progress vs Class Period Number:\n")
p3 <- ggplot(
  data %>% filter(student_total_periods >= 3),
  aes(x = class_period_number, y = stream_progress_pct)
) +
  geom_point(alpha = 0.2, color = "steelblue") +
  geom_smooth(method = "lm", color = "red", se = TRUE) +
  labs(
    title = "Stream Progress vs Class Period Number",
    subtitle = "Do students make more progress in later sessions?",
    x = "Class Period Number (1 = first, 2 = second, etc.)",
    y = "Stream Progress (%)"
  ) +
  theme_minimal()
print(p3)

# 4. Stream-Level Differences (Top 10)
cat("4. Pause Rate by Stream (Top 10):\n")
top_streams <- data %>%
  count(stream_name, sort = TRUE) %>%
  head(10) %>%
  pull(stream_name)

p4 <- data %>%
  filter(stream_name %in% top_streams) %>%
  ggplot(aes(
    x = reorder(stream_name, pause_rate_per_minute, FUN = median),
    y = pause_rate_per_minute
  )) +
  geom_boxplot(fill = "steelblue", alpha = 0.7) +
  coord_flip() +
  labs(
    title = "Pause Rate by Stream (Top 10 Streams)",
    subtitle = "Do some streams elicit more pausing?",
    x = "Stream Name",
    y = "Pause Rate (pauses/minute)"
  ) +
  theme_minimal()
print(p4)

# 5. Stream Difficulty Patterns
cat("5. Stream Completion vs Pause Rate:\n")
stream_summary <- data %>%
  group_by(stream_name) %>%
  summarise(
    avg_pause_rate = mean(pause_rate_per_minute, na.rm = TRUE),
    avg_completion = mean(stream_completion_rate, na.rm = TRUE),
    n_students = n_distinct(student_id_anon)
  ) %>%
  filter(n_students >= 5)  # Only streams with 5+ students

p5 <- ggplot(stream_summary, aes(x = avg_completion, y = avg_pause_rate, size = n_students)) +
  geom_point(alpha = 0.6, color = "steelblue") +
  geom_smooth(method = "lm", se = TRUE, color = "red") +
  labs(
    title = "Stream Difficulty Patterns",
    subtitle = "Streams with lower completion rates (harder?) show different pause patterns",
    x = "Average Stream Completion Rate",
    y = "Average Pause Rate (pauses/minute)",
    size = "Number of\nStudents"
  ) +
  theme_minimal()
print(p5)

# 6. Assessment vs Time Spent
cat("6. Assessments Attempted vs Time Spent:\n")
p6 <- ggplot(data, aes(x = assessments_attempted, y = time_spent_minutes)) +
  geom_point(alpha = 0.3, color = "steelblue") +
  geom_smooth(method = "lm", color = "red", se = TRUE) +
  scale_y_log10() +
  labs(
    title = "Assessments Attempted vs Time Spent",
    x = "Number of Assessments Attempted",
    y = "Time Spent (minutes, log scale)"
  ) +
  theme_minimal()
print(p6)

# 7. Event Rate Distribution
cat("7. Event Rate Distribution:\n")
p7 <- ggplot(data, aes(x = event_rate_per_minute)) +
  geom_histogram(bins = 50, fill = "steelblue", alpha = 0.7) +
  geom_vline(
    xintercept = median(data$event_rate_per_minute, na.rm = TRUE),
    color = "red", linetype = "dashed"
  ) +
  labs(
    title = "Distribution of Event Rates",
    subtitle = "How many actions per minute? Red line = median",
    x = "Events per Minute",
    y = "Count"
  ) +
  theme_minimal()
print(p7)

# 8. Correlation Heatmap
cat("8. Correlation Matrix:\n")
cor_vars <- data %>%
  select(
    pause_rate_per_minute, time_spent_minutes, session_count,
    stream_progress_pct, stream_completion_rate, event_rate_per_minute,
    class_engagement_ratio, assessments_attempted
  ) %>%
  na.omit()

cor_matrix <- cor(cor_vars)

corrplot(
  cor_matrix,
  method = "color",
  type = "upper",
  tl.col = "black",
  tl.srt = 45,
  title = "Correlation Heatmap of Key Metrics",
  addCoef.col = "black",
  number.cex = 0.7
)

# Key Finding: Content Exposure Effect ----------------------------------------

cat("\n=== KEY FINDING ===\n")
cat("Analyzing whether pausing is driven by content exposure vs engagement quality...\n\n")

# Find outlier (if any)
outliers <- data %>%
  filter(time_spent_minutes > 150)

if (nrow(outliers) > 0) {
  cat("Detected outliers (>150 min spent):\n")
  print(outliers %>%
    select(student_id_anon, stream_name, time_spent_minutes, total_pauses,
           student_total_events, session_count))
  
  data_cleaned <- data %>% filter(time_spent_minutes <= 150)
} else {
  data_cleaned <- data
}

# Model comparison
cat("\nLinear model: total_pauses ~ stream_progress + completion + time_spent\n")
model <- lm(
  total_pauses ~ stream_progress_pct + stream_completion_rate + time_spent_minutes,
  data = data_cleaned
)
print(summary(model))

cat("\n=== ANALYSIS COMPLETE ===\n")
cat("Processed data saved to: caliper_processed_data.csv\n")
cat("\nKey metrics available:\n")
cat("  • pause_rate_per_minute: Absolute pausing behavior\n")
cat("  • relative_pause_rate: Normalized to personal baseline\n")
cat("  • stream_engagement_ratio: Normalized to stream difficulty\n")
cat("  • class_engagement_ratio: Normalized to class context\n")

# ==============================================================================
# Notes for Further Analysis:
#
# 1. Absolute pause rate is strongly driven by content exposure (time spent)
# 2. Consider using RELATIVE metrics (normalized to baselines) for engagement
# 3. Individual differences matter even after controlling for stream effects
# 4. Session fragmentation (session_count) may indicate engagement disruption
# 5. First class period shows different patterns than later exposures
#
# Recommended hierarchical model:
# lmer(pause_rate ~ relative_pause_rate + (1|student_id) + (1|stream_name))
#
# This separates individual differences from stream difficulty effects
# ==============================================================================