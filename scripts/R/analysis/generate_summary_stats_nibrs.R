renv::restore()

library(lubridate)
library(stringr)
library(tidyverse)
library(vroom)
library(here)
library(xtable)
library(glue)


df <- vroom(here('data', 'NIBRS', 'raw', 'cannabis_allyears__all_incidents_summary.csv'))
sc <- vroom(here('data', 'output', 'urban_codes_x_county_2013.csv')) %>%
  mutate(is_metro = ifelse(urbancounty == 'Large metro' | urbancounty == 'Small metro', 'metro', 'nonmetro'))
lc <- vroom(here('data', 'misc', 'leaic.tsv')) %>%
  distinct(FIPS, ORI9)

sc <- lc %>% inner_join(sc) %>% rename(ori = ORI9) %>%
  distinct(ori, FIPS, is_metro)
df <- df %>% dplyr::inner_join(sc, by = 'ori')



# transform locations
all_locations <- str_split(str_remove_all(df %>% pull(location), "\\(|\\)|\\'"), pattern = '\\,')
df$all_locations_1 <- all_locations %>% purrr::map(~ .x[[1]]) %>% unlist() %>% tolower()
df$all_locations_2 <- all_locations %>% purrr::map(~ .x[[2]]) %>% unlist() %>% tolower()


get_incidents_info <- function(x){
  
  all_incidents <- nrow(x)
  xsub <- x %>% filter(arrest_type_name != 'No Arrest')
  arrests <- nrow(xsub)
  
  tb <- tribble(
    ~term, ~value, ~value_arrest,
    "% drug only incidents",  
    nrow(x %>% filter(!other_offense))/all_incidents, 
    nrow(xsub %>% filter(!other_offense))/arrests,
    
    "% possessing", 
    nrow(x %>% filter(criminal_act == 'possessing'))/all_incidents,
    nrow(xsub %>% filter(criminal_act == 'possessing'))/arrests,
    
    "% consuming", 
    nrow(x %>% filter(criminal_act == 'consuming'))/all_incidents,
    nrow(xsub %>% filter(criminal_act == 'consuming'))/arrests,
    
    "% buying", nrow(x %>% filter(criminal_act == 'buying'))/all_incidents,
    nrow(xsub %>% filter(criminal_act == 'buying'))/arrests,
    
    "% selling", nrow(x %>% filter(criminal_act == 'selling'))/all_incidents,
    nrow(xsub %>% filter(criminal_act == 'selling'))/arrests,
    
    "% distributing", nrow(x %>% filter(criminal_act == 'distributing'))/all_incidents,
    nrow(xsub %>% filter(criminal_act == 'distributing'))/arrests,
    
    '% other drugs present', nrow(x %>% filter(unique_drug_type_count > 1))/all_incidents,
    nrow(xsub %>% filter(unique_drug_type_count > 1))/arrests,
    
    "% single offender", nrow(x %>% filter(offender_count == 1))/all_incidents,
    nrow(xsub %>% filter(offender_count == 1))/arrests,
    
    "% residence", sum(grepl('residence', pull(x, all_locations_1)) + grepl('residence', pull(x, all_locations_2))>0)/all_incidents,
    sum(grepl('residence', pull(xsub, all_locations_1)) + grepl('residence', pull(xsub, all_locations_2))>0)/arrests,
    
    "% hotel", sum(grepl('hotel', pull(x, all_locations_1)) + grepl('hotel', pull(x, all_locations_2))>0)/all_incidents,
    sum(grepl('hotel', pull(xsub, all_locations_1)) + grepl('hotel', pull(xsub, all_locations_2))>0)/arrests,
    
    "% highway/road", sum(grepl('road', pull(x, all_locations_1)) + grepl('road', pull(x, all_locations_2))>0)/all_incidents,
    sum(grepl('road', pull(xsub, all_locations_1)) + grepl('road', pull(xsub, all_locations_2))>0)/arrests,
    
    "% parking lot/garage", sum(grepl('parking', pull(x, all_locations_1)) + grepl('parking', pull(x, all_locations_2))>0)/all_incidents,
    sum(grepl('parking', pull(xsub, all_locations_1)) + grepl('parking', pull(xsub, all_locations_2))>0)/arrests,
    
    '% during day (6-20)', nrow(x %>% filter(incident_hour > 5 & incident_hour < 21))/nrow(x %>% filter(!is.na(incident_hour))),
    nrow(xsub %>% filter(incident_hour > 5 & incident_hour < 21))/nrow(xsub %>% filter(!is.na(incident_hour))),
    
    '% no arrest', nrow(x %>% filter(arrest_type_name == "No Arrest"))/all_incidents,
    NA,
    
    '% arrest: custody', nrow(x %>% filter(arrest_type_name == 'Taken INTO Custody'))/all_incidents,
    NA,
    '% arrest: on view', nrow(x %>% filter(arrest_type_name == 'On View'))/all_incidents,NA,
    '% arrest: summoned/cited', nrow(x %>% filter(arrest_type_name == 'Summoned / Cited'))/all_incidents, NA
  )
  tb <- tb %>%
    mutate(value_se = sqrt(value * (1-value)/all_incidents),
           value_arrest_se = sqrt(value_arrest * (1-value_arrest)/arrests))
  return(tb)
}

# eda ----

df_list <- list(
  df %>% filter(race == 'black' & is_metro == 'metro'),
  df %>% filter(race == 'black' & sex_code == 'M' & age_num >= 18 & age_num <= 25 & is_metro == 'metro'),
  df %>% filter(race == 'white' & is_metro == 'metro'),
  df %>% filter(race == 'white' & sex_code == 'M' & age_num >= 18 & age_num <= 25 & is_metro == 'metro'),
  df %>% filter(race == 'black' & is_metro == 'nonmetro'),
  df %>% filter(race == 'black' & sex_code == 'M' & age_num >= 18 & age_num <= 25 & is_metro == 'nonmetro'),
  df %>% filter(race == 'white' & is_metro == 'nonmetro'),
  df %>% filter(race == 'white' & sex_code == 'M' & age_num >= 18 & age_num <= 25 & is_metro == 'nonmetro')
)


df_stats <- df_list %>%
  purrr::map(~ get_incidents_info(.x))

df_stats_drugs <- df_list %>%
  purrr::map(~ get_incidents_info(.x %>% filter(!other_offense)))



df_stats_joined <- df_stats %>%
  purrr::map(~ .x %>% 
               mutate(is_numeric_arrest = ifelse(!is.na(as.numeric(value_arrest)), TRUE, FALSE)) %>%
               mutate(value = glue('{round(value*100)}%{ifelse(is_numeric_arrest, paste0(" [",round(value_arrest*100), "%]"), "")}')) %>%
               select(term ,value)) %>%
  reduce(left_join, by = 'term')

df_stats_drugs_joined <- df_stats_drugs %>%
  purrr::map(~ .x %>% 
               mutate(is_numeric_arrest = ifelse(!is.na(as.numeric(value_arrest)), TRUE, FALSE)) %>%
               mutate(value = glue('{round(value*100)}%{ifelse(is_numeric_arrest, paste0(" [",round(value_arrest*100), "%]"), "")}')) %>%
               select(term ,value)) %>%
  reduce(left_join, by = 'term')


n_incidents <- df_list %>% purrr::map(~ nrow(.x)) %>% unlist()
n_incidents_drugs <- df_list %>% purrr::map(~ nrow(.x %>% filter(!other_offense))) %>% unlist()


print(rbind(
  c('# incidents', n_incidents),
  df_stats_joined
) %>% xtable(.,  align = rep('c', 10)),
         include.rownames=FALSE)  
  

print(rbind(
  c('# incidents', n_incidents_drugs),
  df_stats_drugs_joined
) %>% xtable(.,  align = rep('c', 10)),
include.rownames=FALSE)  



# generate plot ----

tb_plot <- df %>%
  group_by(data_year) %>%
  summarise(arrests = 1 - sum(arrest_type_name == 'No Arrest')/n()) %>%
  mutate(term = 'All incidents') %>%
  bind_rows(
    df %>% 
      filter(!other_offense) %>%
      group_by(data_year) %>%
      summarise(arrests = 1 - sum(arrest_type_name == 'No Arrest')/n()) %>%
      mutate(term = 'Incidents with\n only drug offenses')
  ) %>%
  bind_rows(
    df %>% 
      filter(!other_offense) %>%
      filter(crack_count == 0 & cocaine_count == 0 & heroin_count == 0 
             & meth_count == 0 & other_drugs_count == 0) %>%
      group_by(data_year) %>%
      summarise(arrests = 1 - sum(arrest_type_name == 'No Arrest')/n()) %>%
      mutate(term = 'Incidents with\n only marijuana \noffenses')
  )

tb_plot %>%
  rename(`Incidents type` = term) %>%
  ggplot(aes(data_year, arrests, col = `Incidents type`)) + 
  geom_line() + 
  theme_bw() + 
  xlab('Year') + 
  ylab('% arrests') + 
  scale_y_continuous(breaks = seq(0.65,0.85,by=0.05), 
                     labels = glue('{100*seq(0.65,0.85,by=0.05)}%'),
                     limits = c(0.68, 0.83)) + 
  scale_colour_manual(values=c("#000000", "#E69F00", "#56B4E9"))
ggsave(here('scripts', 'R', 'plots', 'arrests_by_incidentstype.pdf'), device = cairo_pdf)




