
library(tidycensus)
library(tidyverse)
library(srvyr)
library(survey)
library(glue)
library(readxl)

# # Reload .Renviron
# readRenviron("~/.Renviron")
# # Check to see that the expected key is output in your R console
# census_api_key(Sys.getenv("CENSUS_KEY"), install = TRUE, overwrite = TRUE)

# useful links
# https://walker-data.com/tidycensus/articles/pums-data.html for standard errors etc
# https://github.com/gergness/srvyr

# get PUMS data ----

# get Public Use Microdata Sample (PUMS)
# View(pums_variables)
# codebook https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2015-2019.pdf
pums <- get_pums(
  variables = c(
    "PUMA",
    "SEX",
    "AGEP",
    "HISP",
    "RAC1P",
    "HINCP",
    "WAGP",
    "POVPIP",
    "FINCP", # family income
    "HINCP", # household income
    "PINCP" # total person's income
  ),
  state = c("AK", "VT"),
  show_call = TRUE,
  survey = "acs5",
  year = 2019,
  recode = TRUE,
  key = "de879f4953355fde9098b4ccc9bbb5f270f4fb29",
  rep_weights = "person"
)

# pums filter
pums <- pums %>%
  filter(RAC1P_label == "White alone" | RAC1P_label == "Black or African American alone") %>%
  filter(HISP_label == "Not Spanish/Hispanic/Latino")

# recode
pums <- pums %>%
  mutate(
    poverty_level = case_when(
      POVPIP >= 0 & POVPIP <= 100 ~ 'living in poverty',
      POVPIP > 100 & POVPIP <= 200 ~ 'income up to 2x poverty threshold',
      POVPIP > 200 ~ 'income more than 2x poverty threshold'
    )
  )

# poverty thresholds
pums %>%
  # TODO: this way we currently assume MCAR
  filter(!is.na(poverty_level)) %>%
  group_by(ST_label, PUMA, poverty_level) %>%
  summarise(n = sum(PWGTP)) %>%
  mutate(perc = n/sum(n))


# RUCA codes (this should not be needed anymore) ----

# load RUCA codes for census tract
# from https://www.ers.usda.gov/data-products/rural-urban-commuting-area-codes.aspx
# first run
# download.file(url = "https://www.ers.usda.gov/webdocs/DataFiles/53241/ruca2010revised.xlsx?v=3188.2",
#               destfile = here('scripts', 'R', 'downloaded_data', 'RUCA_codes.xlsx'))
# then delete first line from file and transform it into csv
# (or more simply download locally and scp to remote)
ur <- read_csv(here('scripts', 'R', 'downloaded_data', 'RUCA_codes.csv')) %>%
  rename(ct = `State-County-Tract FIPS Code (lookup by address at http://www.ffiec.gov/Geocode/)`)
# load file to join PUMA and RUCA
puma2ct <- read_csv("https://www2.census.gov/geo/docs/maps-data/data/rel/2010_Census_Tract_to_2010_PUMA.txt")
puma2ct %>% group_by(STATEFP, COUNTYFP) %>% summarise(n_pumas_x_county = length(unique(PUMA5CE))) %>% group_by(n_pumas_x_county) %>% summarise(n = n()) %>% print(n = 500)
puma2ct %>% group_by(STATEFP, PUMA5CE) %>% summarise(n_counties_x_puma = length(unique(COUNTYFP))) %>% group_by(n_counties_x_puma) %>% summarise(n = n()) %>% print(n = 500)
# join files
ur_with_puma <- ur %>%
  inner_join(puma2ct %>% mutate(ct = glue("{STATEFP}{COUNTYFP}{TRACTCE}")), by = "ct")
# check join is ok
nrow(ur_with_puma)==nrow(ur)

# now join survey information with RUCA codes
ur_with_puma <- ur_with_puma %>%
    rename(ruca_code = `Secondary RUCA Code, 2010 (see errata)`,
           tract_population = `Tract Population, 2010`) %>%
    mutate(ruca_code = ifelse(ruca_code == 99, NA, ruca_code)) %>%
    select(STATEFP, PUMA5CE, ruca_code, tract_population)
# RUCA codes into categories
# see binning here https://www.ers.usda.gov/data-products/rural-urban-commuting-area-codes/documentation/
# and here https://depts.washington.edu/uwruca/ruca-uses.php
# see also this paper https://www.ndsu.edu/fileadmin/publichealth/files/Davis_et_al__2015__Disparities_in_Alcohol__Drug_Use__Mental_Health_in_Rural__Isolated____Res_Areas_JRH.pdf
ur_with_puma <- ur_with_puma %>%
    mutate(ruca_code_binned = case_when(
        ruca_code == 1 | ruca_code == 1.1 | ruca_code == 2 |
            ruca_code == 2.1 | ruca_code == 3 | ruca_code == 4.1 |
            ruca_code == 5.1 | ruca_code == 7.1 | ruca_code == 8.1 ~ 'urban', # urban
        ruca_code == 4 | ruca_code == 4.2 | ruca_code == 5 |
            ruca_code == 5.2 | ruca_code == 6 | ruca_code == 6.1 ~ 'large rural city/town', # rural
        ruca_code = 7 | ruca_code == 7.2 | ruca_code == 7.3 |
            ruca_code == 7.4 | ruca_code == 8 | ruca_code == 8.2 |
            ruca_code == 8.3 | ruca_code == 8.4 | ruca_code == 9 |
            ruca_code == 9.1 | ruca_code == 9.2 | ruca_code == 10 |
            ruca_code == 10.2 | ruca_code == 10.3 | ruca_code == 10.4 |
            ruca_code == 10.5 | ruca_code == 10.6 ~ 'small and isolated small rural town' # rural
    )) %>%
    mutate(is_urban = case_when(
        ruca_code_binned == 'urban' ~ 1,
        ruca_code_binned == 'large rural city/town' | ruca_code_binned == 'small and isolated small rural town' ~ 0))
# check that within the same PUMA the RUCA code is similar
ur_with_puma <- ur_with_puma %>% group_by(STATEFP, PUMA5CE) %>%
  summarise(urban_pop_perc = sum(tract_population * is_urban)/sum(tract_population)) %>%
    rename(ST = STATEFP)


# download MSA data
# TODO
download.file(url = 'https://usa.ipums.org/usa/resources/volii/MSA2013_PUMA2010_crosswalk.xls',
                        destfile = here('scripts', 'R', 'downloaded_data', 'cw_msa.xls'))
cw_msa <- read_xls(here('scripts', 'R', 'downloaded_data', 'cw_msa.xls'))

# experiment with syntax ----

pums_srv <- pums %>% to_survey(.)

# srvyr syntax
pums_srv %>% group_by(ST, PUMA) %>%
    summarise(age_mean = survey_mean(AGEP))
pums_srv %>% group_by(ST, PUMA) %>%
    summarise(male_mean = survey_mean(SEX_label == 'Male'))

# survey syntax
svyby(~SEX_label, ~ST+PUMA, design = pums_srv, svymean)


pums %>%
    summarise(mean_age = weighted.mean(AGEP, PWGTP))


