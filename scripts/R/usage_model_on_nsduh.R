
library(tidyverse)
library(vroom)
library(glue)

# download and store all files ----

# get_nsduh_data <- function(year){
#     download.file(url = glue('https://www.datafiles.samhsa.gov/sites/default/files/field-uploads-protected/studies/NSDUH-{year}/NSDUH-{year}-datasets/NSDUH-{year}-DS0001/NSDUH-{year}-DS0001-bundles-with-study-info/NSDUH-{year}-DS0001-bndl-data-tsv.zip'),
#                   destfile = here('scripts', 'R', 'downloaded_data', 'nsduh.zip'))
#     #dir.create(here('scripts', 'R', 'downloaded_data', 'nsduh'))
#     unzip(here('scripts', 'R', 'downloaded_data', 'nsduh.zip'),
#           exdir = here('scripts', 'R', 'downloaded_data', glue('nsduh_{year}')))
#     file.remove(here('scripts', 'R', 'downloaded_data', 'nsduh.zip'))
# }
#
#years <- as.list(2015:2019)
#names(years) <- unlist(years)
#
# years %>%
#   map( ~ .x %>% get_nsduh_data(.))

# see restricted use nsduh data with census tract information
# https://www.cdc.gov/rdc/b1datatype/nsduh.htm

# load file of interest ----

# merge all data files
df_list <- years %>% map(~ vroom(here('scripts', 'R', 'downloaded_data',
                           glue('nsduh_{.x}'), glue('NSDUH_{.x}_Tab.{if(.x==2019){"txt"}else{"tsv"}}'))))

# keep only categories that are needed for the analysis
# codebook https://www.datafiles.samhsa.gov/sites/default/files/field-uploads-protected/studies/NSDUH-2002-2019/NSDUH-2002-2019-datasets/NSDUH-2002-2019-DS0001/NSDUH-2002-2019-DS0001-info/NSDUH-2002-2019-DS0001-info-codebook.pdf
cols_to_keep <- c("NEWRACE2",
                  "CATAG7",
                  "CATAG6",
                  "MJEVER",
                  "MJAGE",
                  "IRSEX",
                  "MJDAY30A",
                  "ANALWT_C",
                  "PDEN10", # POPULATION DENSITY 2010
                  "COUTYP4", # COUNTY METRO/NONMETRO STATUS
                  "IRPINC3", # RECODE -RESP TOT INCOME - IMPUTATION REVISED, from 2015 onwards
                  "IRFAMIN3", # RECODE - TOT FAM INCOME - IMPUTATION REVISED
                  "POVERTY3", # RC-POVERTY LEVEL (% OF US CENSUS POVERTY THRESHOLD)
                  "ANALWT_C",
                  "VESTR" # stratum
)

df <- df_list %>% map_dfr(~ .x %>% select(cols_to_keep),
                          .id = 'year')

df <- df %>%
  mutate(
    race = case_when(
      NEWRACE2 == 1 ~ 'white',
      NEWRACE2 == 2 ~ 'black',
      NEWRACE2 >= 3 ~ 'other' # Hispanics excluded
    ),
    sex = case_when(
      IRSEX == 1 ~ 'male',
      IRSEX == 2~ 'female'
    ),
    age = case_when(
      CATAG7 == 1 ~ '12-13',
      CATAG7 == 2 ~ '14-15',
      CATAG7 == 3 ~ '16-17',
      CATAG7 == 4 ~ '18-20',
      CATAG7 == 5 ~ '21-25',
      CATAG7 == 6 ~ '26-34',
      CATAG6 == 4 ~ '35-49',
      CATAG6 == 5 ~ '50-64',
      CATAG6 == 6 ~ '65+'
    ),
    poverty_level = case_when(
      POVERTY3 == 1 ~ 'living in poverty',
      POVERTY3 == 2 ~ 'income up to 2x poverty threshold',
      POVERTY3 == 3 ~ 'income more than 2x poverty threshold'
    ),
    # look for rural vs. urban here OMB vs. RUCA and recoding done here
    # https://www.hrsa.gov/rural-health/about-us/definition/index.html
    is_urban = case_when(
      COUTYP4 == 1 ~ 'urban',
      COUTYP4 == 2 ~ 'rural',
      COUTYP4 == 3 ~ 'rural'
    ),
    usage_days = case_when(
      MJDAY30A <= 30 ~ MJDAY30A,
      MJDAY30A == 91 | MJDAY30A == 93 ~ 0
      # the remaining codes categorized as NAs
    ),
    usage_ever = case_when(
      MJEVER == 1 ~ 'yes',
      MJEVER == 2 ~ 'no',
    ),
    usage_agefirsttime = case_when(
      MJAGE >= 1 & MJAGE <= 82 ~ MJAGE,
      MJAGE == 991 ~ 0
    )
  ) %>%
  select(-NEWRACE2, -IRSEX, -CATAG6, -CATAG7, -POVERTY3, -MJDAY30A, -MJAGE)


# transform into survey data syntax http://gdfe.co/srvyr/
df_srv <- df %>%
  as_survey_design(strata = VESTR, weights = ANALWT_C)

# TODO: currently assuming MCAR
# TODO: run logistic regression
df %>%
  filter(!is.na(usage_days)) %>%
  group_by(race, sex, age,
           #is_urban,
           #poverty_level
           ) %>%
  summarise(
    usage_percpop = 1 - sum((usage_days == 0) * ANALWT_C)/sum(ANALWT_C),
    n = n(),
    n_users = sum(usage_days > 0)
  )

# TODO: too few datapoints
df %>%
  filter(usage_days > 0) %>%
  group_by(race, sex, age, is_urban, poverty_level, usage_days) %>%
  summarise(
    n_wgt = sum(ANALWT_C),
    n = n()
  ) %>%
  mutate(perc_wgt = n_wgt / sum(n_wgt))


# notes
# overlapping variables
# age
# sex
# race
# rural vs. metropolitan
# poverty information
## pums
### WAGP Wages or salary income past 12 months,
### POVPIP poverty to income ratio
## nsduh
### IRPINC3 RECODE -RESP TOT INCOME - IMPUTATION REVISED
### IRFAMIN3 RECODE - TOT FAM INCOME - IMPUTATION REVISED
### POVERTY3 RC-POVERTY LEVEL (% OF US CENSUS POVERTY THRESHOLD)

# TODO: add population density (not really needed now that we have rucas)
# TODO: add MSA information (present in nsduh, for pums instead need to use crosswalk)

