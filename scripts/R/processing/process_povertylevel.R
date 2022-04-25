renv::restore()

library(tidycensus)
library(tidyverse)
library(srvyr)
library(survey)
library(glue)
library(readxl)
library(here)
library(vroom)

# useful links
# https://walker-data.com/tidycensus/articles/pums-data.html for standard errors etc
# https://github.com/gergness/srvyr


# get PUMS data ----

# get Public Use Microdata Sample (PUMS)
# View(pums_variables)
# codebook https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2015-2019.pdf

# create df with (county, demographics, poverty level)
get_county_poverty_by_demo <- function(county, pums_srv, cw){
  cw_selected <- cw %>% filter(fips_all == county)

  # data selected
  pums_df_selected <- pums_srv %>%
    filter(puma_all %in% (cw_selected %>% pull(puma_all)))

  out <- pums_df_selected %>%
    group_by(RAC1P_label, SEX_label, age_census_largebins, poverty_level) %>%
    summarize(proportion = survey_mean()) %>%
    tibble(FIPS = county)

}

get_pums_x_state <- function(state_chosen){
  get_pums(
    variables = c(
      "PUMA",
      "SEX",
      "AGEP",
      "HISP",
      "RAC1P",
      "WAGP",
      "POVPIP",
      "FINCP", # family income
      "HINCP", # household income
      "PINCP" # total person's income
    ),
    state = state_chosen,#state.abb[1],
    show_call = TRUE,
    survey = "acs5",
    year = 2019,
    recode = TRUE,
    key = "",# add key
    rep_weights = "person"
  ) %>%
    try()
}

download_poverty_data <- function(state_chosen){

  files_already_present <- list.files(here('data_downloading', 'R', 'poverty_data_x_state'))
  if(glue('{state_chosen}_rawdata.csv') %in% files_already_present){
    return()
  }

  pums <- NULL
  while(is.null(pums) | class(pums)[1] == "try-error"){
    pums <- get_pums_x_state(state_chosen)
  }

  # recode
  pums <- pums %>%
    mutate(
      age_census = case_when(
        AGEP >= 0 & AGEP <= 4 ~ '0-4',
        AGEP >= 5 & AGEP <= 9 ~ '5-9',
        AGEP >= 10 & AGEP <= 14 ~ '10-14',
        AGEP >= 15 & AGEP <= 19 ~ '15-19',
        AGEP >= 20 & AGEP <= 24 ~ '20-24',
        AGEP >= 25 & AGEP <= 29 ~ '25-29',
        AGEP >= 30 & AGEP <= 34 ~ '30-34',
        AGEP >= 35 & AGEP <= 39 ~ '35-39',
        AGEP >= 40 & AGEP <= 44 ~ '40-44',
        AGEP >= 45 & AGEP <= 49 ~ '45-49',
        AGEP >= 40 & AGEP <= 44 ~ '40-44',
        AGEP >= 45 & AGEP <= 49 ~ '45-49',
        AGEP >= 50 & AGEP <= 54 ~ '50-54',
        AGEP >= 55 & AGEP <= 59 ~ '55-59',
        AGEP >= 60 & AGEP <= 64 ~ '60-64',
        AGEP >= 65 & AGEP <= 69 ~ '65-69',
        AGEP >= 70 & AGEP <= 74 ~ '70-74',
        AGEP >= 75 & AGEP <= 79 ~ '75-79',
        AGEP >= 80 & AGEP <= 84 ~ '80-84',
        AGEP >= 85 ~ '85+'
      ),
      age_census_largebins = case_when(
        AGEP >= 12 & AGEP <= 17 ~ '12-17',
        AGEP >= 18 & AGEP <= 25 ~ '18-25',
        AGEP >= 26 & AGEP <= 34 ~ '26-34',
        AGEP >= 35 & AGEP <= 49 ~ '35-49',
        AGEP >= 50~ '50+'
      ),
      poverty_level = case_when(
        POVPIP >= 0 & POVPIP <= 100 ~ 'living in poverty',
        POVPIP > 100 ~ 'income higher than poverty level'
        #POVPIP > 100 & POVPIP <= 200 ~ 'income up to 2x poverty threshold',
        #POVPIP > 200 ~ 'income more than 2x poverty threshold'
      ),
      RAC1P_label = case_when(
        RAC1P_label == "White alone" ~ 'white',
        RAC1P_label == "Black or African American alone" ~ 'black'
      ),
      SEX_label = tolower(SEX_label)
    )

  pums <- pums %>%
    mutate(puma_all = glue('{ST}{PUMA}'))

  dir.create(here('data_downloading', 'R', 'poverty_data_x_state'))
  v <- pums %>%
    write_csv(here('data_downloading', 'R', 'poverty_data_x_state', glue('{state_chosen}_rawdata.csv')))

}

get_poverty_rates_x_state <- function(state_chosen, hisp_included=TRUE){

  pums <- vroom(here('data_downloading', 'R', 'poverty_data_x_state', glue('{state_chosen}_rawdata.csv')),
                col_types = cols())

  puma2ct <- read_csv("https://www2.census.gov/geo/docs/maps-data/data/rel/2010_Census_Tract_to_2010_PUMA.txt")
  puma2ct <- puma2ct %>%
    mutate(fips_all = glue('{STATEFP}{COUNTYFP}'),
           puma_all = glue('{STATEFP}{PUMA5CE}')) %>%
    filter(STATEFP %in% unique(pums$ST))

  pums <- pums %>%
    filter(RAC1P_label == "white" | RAC1P_label == "black")

  if(!hisp_included){
    pums <- pums %>% filter(HISP_label == "Not Spanish/Hispanic/Latino")
  }

  pums_srv <- pums %>%
    to_survey() %>%
    # TODO: impute outcomes
    filter(!is.na(poverty_level))

  county_poverty <- unique(puma2ct$fips_all) %>%
    map( ~  .x %>% get_county_poverty_by_demo(., pums_srv, puma2ct))

  v <- county_poverty %>%
    bind_rows() %>%
    rename(sex = SEX_label, race = RAC1P_label) %>%
    write_csv(here('data_downloading', 'R', 'poverty_data_x_state', glue('{state_chosen}_{ifelse(hisp_included,"hisp","nohisp")}.csv')))

}

# download, save, and merge all files ----

hisp_included <- FALSE

state.abb %>%
  map(~ download_poverty_data(.x))

state.abb %>%
  map(~ get_poverty_rates_x_state(., hisp_included=hisp_included))

# merge all files
files_to_load <- state.abb %>%
  map(~ here('data_downloading', 'R', 'poverty_data_x_state', glue('{.x}_{ifelse(hisp_included,"hisp","nohisp")}.csv')))
names(files_to_load) <- state.abb

files_to_load %>%
  map_dfr( ~ vroom(.x, col_types = cols()) %>% mutate(FIPS = as.character(FIPS)), .id = 'state') %>%
  drop_na() %>%
  rename(age = age_census_largebins) %>%
  # rename states to full names
  inner_join(read_csv(here('data_downloading', 'R', 'state_codes.csv')) %>%
               select(state_name, state_abbr),
             by = c('state' = 'state_abbr')) %>%
  select(-state) %>%
  rename(state = state_name) %>%
  # wide format
  select(-proportion_se) %>%
  select(FIPS, age,sex,race, poverty_level, proportion) %>%
  pivot_wider(names_from = poverty_level,
              values_from = proportion,
              names_prefix = 'poverty_') %>%
  write_csv(here('data_downloading', 'R', glue('poverty_data_{ifelse(hisp_included,"hisp","nohisp")}.csv')))






