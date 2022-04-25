renv::restore()


library(tidyverse)
library(glue)
library(readxl)
library(tigris)

# 2013 ----

# urban influence codes
# https://www.ers.usda.gov/data-products/urban-influence-codes.aspx
download.file('https://www.ers.usda.gov/webdocs/DataFiles/53797/UrbanInfluenceCodes2013.xls?v=3509.8',
              destfile = here('data_downloading', 'R', 'UIC_codes_2013.xlsx'))
df <- readxl::read_xls(here('data_downloading', 'R', 'UIC_codes_2013.xlsx'))

df <- df %>%
    mutate(urban = case_when(
        Description == 'Large-in a metro area with at least 1 million residents or more' ~ 'Large metro',
        Description == 'Small-in a metro area with fewer than 1 million residents' ~ 'Small metro',
        TRUE ~ 'Nonmetro'
    ))

df %>% count(Description, urban)


# assign numeric codes to states
state_codes <- tibble(state_abbr = state.abb, state_name = state.name)

df %>%
    inner_join(state_codes %>% select(state_abbr,state_name),
               by = c('State' = 'state_abbr')) %>%
    select(state_name, FIPS, urban) %>%
    rename(urbancounty = urban) %>%
    rename(state = state_name) %>%
    write_csv(here('data', 'output','urban_codes_x_county_2013.csv'))


# 2003 ----

# download this file manually from https://www.ers.usda.gov/data-products/urban-influence-codes.aspx
# name 2003 and 1993 Urban influence Codes for U.S. counties
df <- read.csv(here('data_downloading', 'R', 'UIC_codes_2003.csv')) %>%
    rename(Description = X2003.Urban.Influence.Code.description,
           FIPS = FIPS.Code)

df <- df %>%
    mutate(urban = case_when(
        Description == 'Large-in a metro area with at least 1 million residents or more' ~ 'Large metro',
        Description == 'Small-in a metro area with fewer than 1 million residents' ~ 'Small metro',
        TRUE ~ 'Nonmetro'
    ))

df %>% count(Description, urban)


# assign numeric codes to states

df %>%
    inner_join(state_codes %>% select(state_abbr,state_name),
               by = c('State' = 'state_abbr')) %>%
    select(state_name, FIPS, urban) %>%
    rename(urbancounty = urban) %>%
    rename(state = state_name) %>%
    write_csv(here('data', 'output','urban_codes_x_county_2003.csv'))
