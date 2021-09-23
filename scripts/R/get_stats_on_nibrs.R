
library(lubridate)
library(stringr)
library(tidyverse)
library(vroom)
library(here)
library(xtable)
library(glue)

df <- vroom(here('data', 'output', 'cannabis_allyears_allincidents_summary.csv'))
sc <- vroom(here('data', 'output', 'urban_codes_x_county_2013.csv')) %>%
  mutate(is_rural = ifelse(urbancounty == 'Large metro', 'urban', 'rural'))
lc <- vroom(here('data', 'output', 'leaic.tsv')) %>%
  distinct(FIPS, ORI9)

sc <- lc %>% inner_join(sc) %>% rename(ori = ORI9) %>%
  distinct(ori, FIPS, is_rural)
df <- df %>% dplyr::inner_join(sc, by = 'ori')


# transform locations
all_locations <- str_split(str_remove_all(df %>% pull(location), "\\(|\\)|\\'"), pattern = '\\,')
df$all_locations_1 <- all_locations %>% purrr::map(~ .x[[1]]) %>% unlist() %>% tolower()
df$all_locations_2 <- all_locations %>% purrr::map(~ .x[[2]]) %>% unlist() %>% tolower()


get_incidents_info <- function(x){
  
  all_incidents <- nrow(x)
  
  tb <- tribble(
    ~term, ~value,
    "% drug only incidents",  nrow(x %>% filter(!other_offense))/all_incidents,
    "% possessing", nrow(x %>% filter(criminal_act == 'possessing'))/all_incidents,
    "% consuming", nrow(x %>% filter(criminal_act == 'consuming'))/all_incidents,
    "% buying", nrow(x %>% filter(criminal_act == 'buying'))/all_incidents,
    "% selling", nrow(x %>% filter(criminal_act == 'selling'))/all_incidents,
    "% distributing", nrow(x %>% filter(criminal_act == 'distributing'))/all_incidents,
    '% other drugs present', nrow(x %>% filter(other_drugs_count > 0))/all_incidents,
    "% single offender", nrow(x %>% filter(offender_count == 1))/all_incidents,
    "% residence", sum(grepl('residence', pull(x, all_locations_1)) + grepl('residence', pull(x, all_locations_2))>0)/all_incidents,
    "% hotel", sum(grepl('hotel', pull(x, all_locations_1)) + grepl('hotel', pull(x, all_locations_2))>0)/all_incidents,
    "% highway/road", sum(grepl('road', pull(x, all_locations_1)) + grepl('road', pull(x, all_locations_2))>0)/all_incidents,
    "% parking lot/garage", sum(grepl('parking', pull(x, all_locations_1)) + grepl('parking', pull(x, all_locations_2))>0)/all_incidents,
    '% during day (6-20)', nrow(x %>% filter(incident_hour > 5 & incident_hour < 21))/nrow(x %>% filter(!is.na(incident_hour))),
    '% no arrest', nrow(x %>% filter(arrest_type_name == "No Arrest"))/all_incidents,
    '% arrest: custody', nrow(x %>% filter(arrest_type_name == 'Taken INTO Custody'))/all_incidents,
    '% arrest: on view', nrow(x %>% filter(arrest_type_name == 'On View'))/all_incidents,
    '% arrest: summoned/cited', nrow(x %>% filter(arrest_type_name == 'Summoned / Cited'))/all_incidents
  )
  tb <- tb %>%
    mutate(value_se = sqrt(value * (1-value)/nrow(x)))
  return(tb)
}

df_inc <- get_incidents_info(df)


df_list <- list(
  df %>% filter(race == 'black' & is_rural == 'urban'),
  df %>% filter(race == 'black' & sex_code == 'M' & age_num >= 18 & age_num <= 25 & is_rural == 'urban'),
  df %>% filter(race == 'white' & is_rural == 'urban'),
  df %>% filter(race == 'white' & sex_code == 'M' & age_num >= 18 & age_num <= 25 & is_rural == 'urban'),
  df %>% filter(race == 'black' & is_rural == 'rural'),
  df %>% filter(race == 'black' & sex_code == 'M' & age_num >= 18 & age_num <= 25 & is_rural == 'rural'),
  df %>% filter(race == 'white' & is_rural == 'rural'),
  df %>% filter(race == 'white' & sex_code == 'M' & age_num >= 18 & age_num <= 25 & is_rural == 'rural')
)


df_stats <- df_list %>%
  purrr::map(~ get_incidents_info(.x))

df_stats_joined <- df_stats %>%
 # purrr::map(~ .x %>%
               #mutate(value = glue('{round(value*100)}% ({round(value_se*100)})'))
#               )
  purrr::map(~ .x %>% mutate(value = glue('{round(value*100)}%')) %>%
               select(-value_se)) %>%
  reduce(left_join, by = 'term')

n_incidents <- df_list %>% purrr::map(~ nrow(.x)) %>% unlist()


print(rbind(
  c('# incidents', n_incidents),
  df_stats_joined
) %>% xtable(.,  align = rep('c', 11)),
         include.rownames=FALSE)  
  




