renv::restore()

library(here)
library(tidyverse)
library(tigris)
library(glue)


# election file downloaded from https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/VOQCHQ
df <- read.table(here('data_downloading', 'R', 'countypres_2000-2020.tab'), 
                 header = TRUE, sep = "\t", fill = TRUE) %>% 
  select(-state) %>% rename(state = state_po, FIPS = county_fips) %>%
    mutate(FIPS = glue('{ifelse(nchar(as.character(FIPS))==4, "0", "")}{FIPS}'))
sc <- fips_codes %>%
    mutate(FIPS = glue('{state_code}{county_code}'))

df %>% glimpse()
df %>% count(party)

fix_char_countycode <- function(x){
   glue('{ifelse(nchar(county_fips)==6, "0", "")}county_fips')
}


# check full overlap between counties ----

sc_lj <- sc %>% distinct(state, FIPS) %>%
    left_join(
        df %>% distinct(year, state, FIPS) %>%
            mutate(dummy = 1)
    ) %>%
    filter(is.na(dummy))

df %>% distinct(state, FIPS) %>%
    left_join(
        sc %>% distinct(state, FIPS) %>%
            mutate(dummy = 1)
    ) %>%
    filter(is.na(dummy))
# here the problem seems to be mostly with AK


# write data ----

df %>%
    mutate(is_republican = ifelse(party == 'REPUBLICAN', 1, 0),
           is_democratic = ifelse(party == 'DEMOCRAT', 1, 0)) %>%
    group_by(year, state, FIPS, county_name) %>%
    summarise(perc_republican_votes = sum(candidatevotes * is_republican)/ sum(candidatevotes),
              perc_democratic_votes = sum(candidatevotes * is_democratic / sum(candidatevotes)),
              total_votes = sum(candidatevotes)) %>%
    write_csv(here('data_downloading', 'R', 'election_results_x_county.csv'))



