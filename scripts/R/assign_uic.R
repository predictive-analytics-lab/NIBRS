
library(tidyverse)
library(glue)
library(readxl)

# urban influence codes
# https://www.ers.usda.gov/data-products/urban-influence-codes.aspx
download.file('https://www.ers.usda.gov/webdocs/DataFiles/53797/UrbanInfluenceCodes2013.xls?v=3509.8',
              destfile = here('scripts', 'R', 'downloaded_data', 'UIC_codes.xlsx'))
df <- readxl::read_xls(here('scripts', 'R', 'downloaded_data', 'UIC_codes.xlsx'))

df <- df %>%
    mutate(urban = case_when(
        Description == 'Large-in a metro area with at least 1 million residents or more' ~ 'Large metro',
        Description == 'Small-in a metro area with fewer than 1 million residents' ~ 'Small metro',
        TRUE ~ 'Nonmetro'
    ))

df %>% count(Description, urban)

# assign numeric codes to states
state_codes <- read_csv(here('scripts', 'R', 'downloaded_data', 'state_codes.csv'))

df %>%
    inner_join(state_codes %>% select(state_abbr,state_name),
               by = c('State' = 'state_abbr')) %>%
    select(state_name, FIPS, urban) %>%
    rename(urbancounty = urban) %>%
    rename(state = state_name) %>%
    write_csv(here('scripts', 'R', 'downloaded_data', 'urban_codes_x_county.csv'))


