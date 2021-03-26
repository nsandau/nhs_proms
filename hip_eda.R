
library(here)
library(janitor)
library(tidyverse)
library(naniar)
library(skimr)
set.seed(111)

# load csvs
csv_dir <- here("data", "csv", "hip")
csv_files <- csv_dir %>%
  list.files() %>%
  set_names(str_remove(., ".csv"))

df <- csv_files %>%
  map_dfr(~ read_csv(file.path(csv_dir, .))) %>%
  janitor::clean_names()




# remove unneeded vars
df_prep <- df %>%
  filter(revision_flag != 1) %>%
  select(
    -contains("_predicted"),
    -revision_flag,
    -procedure,
    -contains("_index_profile"),
    -contains("_assisted"),
    -provider_code,
    -contains("post_op"),
    post_op_q_eq5d_index,
    hip_replacement_post_op_q_score,
    post_op_q_eq_vas
  )

## removing patients below 50

df_prep <- df_prep %>% filter(!age_band %in% c("20 to 29", "30 to 39", "40 to 49"))


### MISSING VALUES
#' Replacing 9 and * with NA to make missing values explicit.
#' Gender coded as 9 for "not specified" - setting to 9
#' recoding to male yes/no = male = 1


df_prep <- df_prep %>%
  mutate(
    gender = as.numeric(gender),
    across(where(is_numeric), ~ na_if(., 9)),
    across(where(is_character), ~ na_if(., "*")),
    across(where(is_numeric), ~ na_if(., 999))
  )


skim(df_prep)
gg_miss_var(df_prep, show_pct = T)

#' Variables asking for specific disease states like stroke, high_bp, lung_disease etc have a lot of missing values.
#' However, they can only be answered as either YES (scored as 1) or no answer / missing (9).
#' A lot of the missing values likely mean "NO".


#' Recoding as No for now
df_prep <- df_prep %>%
  mutate(
    across(all_of(df_prep %>% miss_var_summary() %>% filter(pct_miss > 25) %>% pull(variable)), ~ replace_na(., 0))
  )

gg_miss_var(df_prep, show_pct = T)


# seems the distribution for outcomes are the same for missing vs non-missing. At least for gender and age
df_prep %>%
  select(age_band, gender, post_op_q_eq5d_index, hip_replacement_post_op_q_score, post_op_q_eq_vas) %>%
  group_by(age_band, gender) %>%
  skim()


# SKAL OGSÅ TESTE AT OUTCOMES ER DE SAMME PER ÅR!



### RECODING

# kunne måske kode age_band som factor .. men vil det give mig noget?

df_prep <- df_prep %>%
  mutate(
    gender = if_else(gender == 2, 0, 1),
    across(all_of(c(c("pre_op_q_previous_surgery", "pre_op_q_disability"))), ~ recode(., "2" = 0)), # No = 2 to No = 0
    across(where(~ min(.x, na.rm = T) == 1), ~ .x - 1), # SKAL VÆRE FØR AGE ELLERS SCALER DEN OGSÅ DER!
    age_band = recode(
      age_band,
      "50 to 59" = 1,
      "60 to 69" = 2,
      "70 to 79" = 3,
      "80 to 89" = 4,
      "90 to 120" = 5
    ),
    post_op_q_eq5d_index = post_op_q_eq5d_index + 10 # scaler så kan identificere som cont i python pipeline
  )

#' Splitting into 3 sep dfs
## ------------------------------------------------------------------------------------------------




######  QSCORE ######
df_qscore <- df_prep %>%
  select(-post_op_q_eq_vas, -post_op_q_eq5d_index) %>%
  drop_na(hip_replacement_post_op_q_score) %>%
  mutate( # coding as 1 if MDI > 8
    hip_replacement_post_op_q_score_bin = case_when(
      (hip_replacement_post_op_q_score - hip_replacement_pre_op_q_score) < 8 ~ 1, # MDI = 8
      TRUE ~ 0
    )
  ) %>%
  select(-hip_replacement_post_op_q_score)


df_qscore %>%
  mutate(across(where(is_double) &
    !c(pre_op_q_eq5d_index, pre_op_q_eq_vas, hip_replacement_pre_op_q_score), as.factor),
  year = as.factor(year)
  ) %>%
  plot_bar(by = "year")




##### eq5d ######
df_eq5d <- df_prep %>%
  select(-hip_replacement_post_op_q_score, -post_op_q_eq_vas) %>%
  drop_na(post_op_q_eq5d_index)


df_eq5d %>% mutate(across(where(is_double) & !c(post_op_q_eq5d_index, pre_op_q_eq5d_index, pre_op_q_eq_vas, hip_replacement_pre_op_q_score), as.factor),
                   year = as.factor(year)
) %>% create_report(by = year)


# kan prøve at transforme numeriske?



## VAS ###########
df_vas <- df_prep %>%
  select(-hip_replacement_post_op_q_score, -post_op_q_eq5d_index) %>%
  drop_na(post_op_q_eq_vas)





## EXPORTING TO CSV FOR  MODELING

write_csv(df_eq5d, "data/data_preprocessed/eq5d.csv")
write_csv(df_vas, "data/data_preprocessed/vas.csv")
write_csv(df_qscore, "data/data_preprocessed/qscore.csv")
