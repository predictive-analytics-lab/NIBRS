set.seed(1321312)
args <- commandArgs(trailingOnly = TRUE)

renv::restore()

# TODO: revise
if(length(args)!=3){
  poverty <- 0
  urban <- 0
  hispanic_included <- 0
  year_min <- 2012
  year_max <- 2019
} else{
  print(args)
  poverty <- ifelse(args[1]==1, 1, 0)
  urban <- ifelse(args[2]==1, 1, 0)
  hispanic_included <- ifelse(args[3]==1, 1, 0)
  year_min <- ifelse(!is.na(args[4]), args[4], 2012)
  year_max <- ifelse(!is.na(args[5]), args[5], 2019)
}

library(dplyr)
library(purrr)
library(readr)
library(vroom)
library(glue)
library(here)
library(srvyr)
library(cli)
library(missRanger)
library(ranger)


cli_h1('Processing NSDUH files!')
cli_text('Parameters')
cli_li(glue('poverty :{poverty}'))
cli_li(glue('urban: {urban}'))
cli_li(glue('hispanic included: {hispanic_included}'))

# download and store all files ----

years <- year_min:year_max
get_nsduh_data <- function(year){
    download.file(url = glue('https://www.datafiles.samhsa.gov/sites/default/files/field-uploads-protected/studies/NSDUH-{year}/NSDUH-{year}-datasets/NSDUH-{year}-DS0001/NSDUH-{year}-DS0001-bundles-with-study-info/NSDUH-{year}-DS0001-bndl-data-tsv.zip'),
                  destfile = here('scripts', 'R', 'downloaded_data', 'nsduh.zip'))
    #dir.create(here('scripts', 'R', 'downloaded_data', 'nsduh'))
    unzip(here('scripts', 'R', 'downloaded_data', 'nsduh.zip'),
          exdir = here('scripts', 'R', 'downloaded_data', glue('nsduh_{year}')))
    file.remove(here('scripts', 'R', 'downloaded_data', 'nsduh.zip'))
}

# download only files that if they are not present
dir.create(here('scripts', 'R', 'downloaded_data'))
files_present <- list.files(here('scripts', 'R', 'downloaded_data'))
years_to_download <- years[!(glue('nsduh_{years}') %in% files_present)]

if(length(years_to_download)>0){
  years %>%
    map( ~ .x %>% get_nsduh_data(.))
}


# see restricted use nsduh data with census tract information
# https://www.cdc.gov/rdc/b1datatype/nsduh.htm


# load file of interest ----

# merge all data files

df_list <- years %>% map(~ vroom(here('scripts', 'R', 'downloaded_data',
                           glue('nsduh_{.x}'), glue('NSDUH_{.x}_Tab.{if(.x==2019){"txt"}else{"tsv"}}')),
                           col_types = cols()))
names(df_list) <- years

# keep only categories that are needed for the analysis
# codebook https://www.datafiles.samhsa.gov/sites/default/files/field-uploads-protected/studies/NSDUH-2002-2019/NSDUH-2002-2019-datasets/NSDUH-2002-2019-DS0001/NSDUH-2002-2019-DS0001-info/NSDUH-2002-2019-DS0001-info-codebook.pdf
cols_to_keep <- c(
                  "CATAG7",
                  "CATAG6",
                  "MJEVER",
                  "MJAGE",
                  "IRSEX",
                  "COUTYP2",
                  "MJDAY30A",
                  "ANALWT_C",
                  "NEWRACE2", # race/hispanic ethnicity recode
                  "PDEN10", # POPULATION DENSITY 2010
                  "COUTYP4", # COUNTY METRO/NONMETRO STATUS
                  "IRPINC3", # RECODE -RESP TOT INCOME - IMPUTATION REVISED, from 2015 onwards
                  "IRFAMIN3", # RECODE - TOT FAM INCOME - IMPUTATION REVISED
                  "POVERTY3", # RC-POVERTY LEVEL (% OF US CENSUS POVERTY THRESHOLD)
                  "POVERTY2",
                  "ANALWT_C",
                  "VESTR" # stratum
)


df <- df_list %>% map_dfr(~ .x %>% select(any_of(cols_to_keep)),
                          .id = 'year')

# process data ----

# randomize which hispanics will be coded as whites and which will be coded as blacks


df <- df %>%
  mutate(
    sex = case_when(
      IRSEX == 1 ~ 'male',
      IRSEX == 2~ 'female'
    ),
    # age = case_when(
    #   CATAG7 == 1 ~ '12-13',
    #   CATAG7 == 2 ~ '14-15',
    #   CATAG7 == 3 ~ '16-17',
    #   CATAG7 == 4 ~ '18-20',
    #   CATAG7 == 5 ~ '21-25',
    #   CATAG7 == 6 ~ '26-34',
    #   CATAG6 == 4 ~ '35-49',
    #   CATAG6 == 5 ~ '50-64',
    #   CATAG6 == 6 ~ '65+'
    # ),
    age = case_when(
      CATAG7 == 1 ~ '12-17',
      CATAG7 == 2 ~ '12-17',
      CATAG7 == 3 ~ '12-17',
      CATAG7 == 4 ~ '18-25',
      CATAG7 == 5 ~ '18-25',
      CATAG7 == 6 ~ '26-34',
      CATAG6 == 4 ~ '35-49',
      CATAG6 == 5 ~ '50+',
      CATAG6 == 6 ~ '50+'
    ),
    poverty_level = case_when(
      POVERTY3 == 1 | POVERTY2 == 1 ~ 'living in poverty',
      POVERTY3 == 2 | POVERTY3 == 3 | POVERTY2 == 2 | POVERTY2 == 3 ~ 'income higher than poverty threshold',
    ),
    # look for rural vs. urban here OMB vs. RUCA and recoding done here
    # https://www.hrsa.gov/rural-health/about-us/definition/index.html
    is_urban = case_when(
      COUTYP4 == 1 | COUTYP2 == 1~ 'urban',
      COUTYP4 == 2 | COUTYP2 == 2 ~ 'rural',
      COUTYP4 == 3 | COUTYP2 == 3 ~ 'rural'
    ),
    usage_days = case_when(
      MJDAY30A <= 30 ~ MJDAY30A,
      MJDAY30A == 91 | MJDAY30A == 93 ~ 0
      # the remaining codes categorized as NAs
    ),
    usage_ever = case_when(
      MJEVER == 1 ~ 1,
      MJEVER == 2 ~ 0,
    ),
    usage_agefirsttime = case_when(
      MJAGE >= 1 & MJAGE <= 82 ~ MJAGE,
      MJAGE == 991 ~ 0
    )
  )



if(hispanics_included == 1){

  # impute missing data
  imputed_df <- missRanger(df %>% select(NEWRACE2, IRFAMIN3, sex, age, usage_ever, poverty_level, is_urban, usage_agefirsttime), pmm.k = 3, num.trees = 100)

  # fit rf
  rf <- ranger(is_white ~ sex + age + usage_ever + IRFAMIN3 + poverty_level + poverty_level + usage_agefirsttime,
         data = imputed_df %>% mutate(is_white = case_when(
           NEWRACE2 == 1 ~ 1,
           NEWRACE2 == 2 ~ 0
         )) %>%
           mutate(is_white = as.factor(is_white)) %>%
           filter(!is.na(is_white)),
         probability = TRUE
         )
  pred_is_white <- predict(rf, imputed_df %>% filter(NEWRACE2 == 7))$predictions
  # we can't really predict race well, but going with this for the moment
  col_to_use <- ifelse(mean(pred_is_white[,1]>0.5)>0.5, 1, 2)
  pred_is_white <- pred_is_white[,col_to_use]

  # randomly sample which race the person belongs to
  df[df$NEWRACE2 == 7,'NEWRACE2'] <- ifelse(runif(length(pred_is_white), 0, 1) < pred_is_white, 1, 2)
}

df <- df %>%
  mutate(race = case_when(
    NEWRACE2 == 1 ~ 'white',
    NEWRACE2 == 2 ~ 'black',
    NEWRACE2 >= 3 ~ 'other'
  )) %>%
  select(-NEWRACE2, -IRSEX, -CATAG6, -CATAG7, -POVERTY3, -MJDAY30A, -MJAGE,
         -COUTYP4, -COUTYP2, -POVERTY2)

# transform age again to match the Census categories
# (old code)
# see here https://www2.census.gov/programs-surveys/popest/technical-documentation/file-layouts/2010-2020/cc-est2020-alldata6.pdf
# this does not really match the Census categories
# df <- df %>%
#   mutate(age_census = case_when(
#       age == '12-13' | age == '14-15' ~ '12-15',
#       age == '16-17' | age == '18-20' ~ '16-20',
#       TRUE ~ age))

# generate stats of interest ----

# make sure we have all the info
df %>% count(sex, race, age, poverty_level, year)

# look at some stats (without taking into account the survey structure )
# df %>%
#   group_by(race,
#            # is_urban,
#            poverty_level) %>%
#   summarise(
#     days = mean(ifelse(usage_days>0, usage_days, NA), na.rm = TRUE),
#     ever = mean(ifelse(usage_ever=='yes', 1, 0), na.rm = TRUE)
#   )

# transform into survey data syntax http://gdfe.co/srvyr/
df_srv <- df %>%
  as_survey_design(strata = VESTR, weights = ANALWT_C)

vars_group <- c('race', 'sex', 'age')
if(poverty == 1)  vars_group <- c(vars_group, 'poverty_level')
if(urban == 1)  vars_group <- c(vars_group, 'is_urban')

# TODO: look at how variance is computed without the replicate weights
stats_df <- df_srv %>%
  group_by(year == 2012) %>%
   group_by(across(all_of(vars_group))) %>%
   summarise(
     ever_used = survey_mean(usage_ever, na.rm = TRUE),
     mean_usage_days = survey_mean(usage_days, na.rm = TRUE)
   )

filename <- here('scripts', 'R', 'downloaded_data',
                 glue('nsduh_usage_{min(years)}_{max(years)}{ifelse(hispanic_included == 1, "_hispincluded", "_nohisp")}{ifelse(poverty==1, "_poverty", "")}{ifelse(urban==1, "_urban", "")}'))

stats_df %>%
  write_csv(filename)


cat(filename)
# old code ----

# # TODO: currently assuming MCAR
# # TODO: run regression
# df %>% distinct(race, sex, age, is_urban, poverty_level)
# df %>% group_by(age) %>%
#   summarise(
#     yes = mean(ifelse(usage_days>0,1, 0), na.rm = TRUE),
#     days = mean(ifelse(usage_days>0,usage_days, 0), na.rm = TRUE))
# df %>% group_by(is_urban) %>%
#   summarise(
#     yes = mean(ifelse(usage_days>0,1, 0), na.rm = TRUE),
#     days = mean(ifelse(usage_days>0,usage_days, 0), na.rm = TRUE))
# df %>% group_by(poverty_level) %>%
#   summarise(
#     yes = mean(ifelse(usage_days>0,1, 0), na.rm = TRUE),
#     days = mean(ifelse(usage_days>0,usage_days, 0), na.rm = TRUE))
#
# # TODO: too few datapoints
# df %>% group_by(race) %>% summarise(n = n())
# df %>%
#   #filter(usage_days > 0) %>%
#   group_by(race, sex,
#            age_census,
#            #is_urban,
#            poverty_level,
#            usage_days) %>%
#   summarise(
#     n_wgt = sum(ANALWT_C),
#     n = n()
#   ) %>%
#   mutate(perc_wgt = n_wgt / sum(n_wgt))
#   # %>% ggplot(aes(usage_days, n)) + geom_point() + ylim(0,200)
#   %>%
#
#
# # notes
# # overlapping variables
# # age
# # sex
# # race
# # rural vs. metropolitan
# # poverty information
# ## pums
# ### WAGP Wages or salary income past 12 months,
# ### POVPIP poverty to income ratio
# ## nsduh
# ### IRPINC3 RECODE -RESP TOT INCOME - IMPUTATION REVISED
# ### IRFAMIN3 RECODE - TOT FAM INCOME - IMPUTATION REVISED
# ### POVERTY3 RC-POVERTY LEVEL (% OF US CENSUS POVERTY THRESHOLD)
#
# # TODO: add population density (not really needed now that we have rucas)
# # TODO: add MSA information (present in nsduh, for pums instead need to use crosswalk)
#
