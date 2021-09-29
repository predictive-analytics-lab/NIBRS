
library(rvest)
library(tidyverse)
library(lubridate)
library(here)
library(vroom)
library(cli)


nibrs <- vroom(here('data', 'output', 'cannabis_2010-2019_allincidents_summary.csv'))
lc <- vroom(here('data', 'output', 'leaic.tsv')) %>%
  distinct(FIPS, ORI9, ADDRESS_CITY, COUNTYNAME, ADDRESS_STATE) %>%
  mutate(city = tolower(ADDRESS_CITY), county = tolower(COUNTYNAME),
         state = tolower(ADDRESS_STATE)) %>%
  select(-ADDRESS_CITY, -COUNTYNAME, -ADDRESS_STATE)
sc <- vroom(here('data', 'output', 'urban_codes_x_county_2013.csv'))
sr <- vroom(here('data', 'output', 'selection_ratio_county_2010-2019_bootstraps_1000_poverty.csv'))

lc <- lc %>%
  inner_join(sc)

reporting_ori <- nibrs %>%
  distinct(data_year, ori)

lc_reporting <- lc %>%
  inner_join(reporting_ori, by = c('ORI9' = 'ori'))

# download files ----

pg <- read_html('https://openpolicing.stanford.edu/data/')
links <- pg %>% html_elements("a") %>%
  html_attr('href')
links <- links[grepl('https://stacks.stanford.edu/file/', links)]
links

pg_elements <- pg %>% html_elements('tr') %>%
  html_element('td') %>%
  html_text2()

state <- c()
place <- c()
for(el in pg_elements[!is.na(pg_elements)]){
  if(nchar(el)==2){
    current_state <- el
  } else if(nchar(el)>0){
    state <- c(state, current_state)
    place <- c(place, el)
  }
}
places_in_pg <- tibble(
  state = tolower(state),
  place = tolower(place)
)

time_range_html <- pg %>% 
  html_nodes(., css = "td[data-title='Time range']") %>%
  html_text2()

time_range <- time_range_html %>%
  str_split(., ' - ') %>%
  map_dfr(~ tibble(start = .x[[1]], end = .x[[2]]))

places_in_pg_wt <- places_in_pg %>%
  bind_cols(time_range) %>%
  mutate(start_year = year(start), 
         end_year = year(end),
         n_months = interval(start, end) / days(30))

places_present_wt <- places_in_pg_wt %>%
  left_join(
    lc_reporting %>% distinct(city, state, data_year) %>%
      mutate(city_present = 1),
    by = c('state', 'place' = 'city')
  ) %>%
  left_join(
    lc_reporting %>% distinct(state, county) %>%
      mutate(county_present = 1),
    by = c('state', 'place' = 'county')
  ) %>%
  filter(!is.na(city_present) | !is.na(county_present))

places_present_wt <- places_present_wt %>%
  filter(start_year <= data_year & end_year >= data_year)

places_present_to_donwload <- places_present_wt %>% distinct(state, place)
places_present_to_donwload <- places_present_to_donwload %>%
  bind_cols(
    link = glue("_{places_present_to_donwload$state}_{str_replace(places_present_to_donwload$place, ' ', '_')}") %>%
      map(~ links[grepl(.x, links) & grepl('rds', links)]) %>%
      unlist()
  )
handle_file <- function(url, destfile){
  #browser()
  cli_text(url)
  if(destfile %in% list.files(here('data', 'openpolicingproject'), full.names = TRUE)) return()
  download.file(url = url, destfile = destfile)
}
1:nrow(places_present_to_donwload) %>%
  map(~ handle_file(url = pull(places_present_to_donwload, link)[.x], 
                      destfile = here('data', 'openpolicingproject', 
                                      glue('{pull(places_present_to_donwload, state)[.x]}_{pull(places_present_to_donwload, place)[.x]}.rds'))))


# handle open policing project files ----

process_pp <- function(x) {
  if ("subject_race" %in% colnames(x)) {
    x <- x %>% filter(grepl('african|black|white|caucasian', tolower(subject_race)))
    x <- x %>% mutate(race_newcode = case_when(
      grepl('black', tolower(subject_race)) ~ 'black',
      grepl('african', tolower(subject_race)) ~ 'black',
      grepl('white', tolower(subject_race)) ~ 'white',
      grepl('caucasian', tolower(subject_race)) ~ 'white'
    ))
  } else if ("raw_race" %in% colnames(x)) {
    x <- x %>% filter(grepl('african|black|white|caucasian', tolower(raw_race)))
    x <- x %>% mutate(race_newcode = case_when(
      grepl('black', tolower(raw_race)) ~ 'black',
      grepl('african', tolower(raw_race)) ~ 'black',
      grepl('white', tolower(raw_race)) ~ 'white',
      grepl('caucasian', tolower(raw_race)) ~ 'white'
    ))
  } else if('raw_defendant_race' %in% colnames(x)){
    x <- x %>% filter(grepl('african|black|white|caucasian', tolower(raw_defendant_race)))
    x <- x %>% mutate(race_newcode = case_when(
      grepl('black', tolower(raw_defendant_race)) ~ 'black',
      grepl('african', tolower(raw_defendant_race)) ~ 'black',
      grepl('white', tolower(raw_defendant_race)) ~ 'white',
      grepl('caucasian', tolower(raw_defendant_race)) ~ 'white'
    ))
  } else if('raw_subject_race_code' %in% colnames(x)){
    x <- x %>% filter(grepl('b|w', tolower(raw_subject_race_code)))
    x <- x %>% mutate(race_newcode = case_when(
      grepl('b', tolower(raw_subject_race_code)) ~ 'black',
      grepl('w', tolower(raw_subject_race_code)) ~ 'white',
    ))
  } else if('raw_driver_race' %in% colnames(x)){
    x <- x %>% filter(grepl('african|black|white|caucasian', tolower(raw_driver_race)))
    x <- x %>% mutate(race_newcode = case_when(
      grepl('black', tolower(raw_driver_race)) ~ 'black',
      grepl('african', tolower(raw_driver_race)) ~ 'black',
      grepl('white', tolower(raw_driver_race)) ~ 'white',
      grepl('caucasian', tolower(raw_driver_race)) ~ 'white'
    ))
  } else if('raw_persons_race' %in% colnames(x)){
    x <- x %>% filter(grepl('african|black|white|caucasian', tolower(raw_persons_race)))
    x <- x %>% mutate(race_newcode = case_when(
      grepl('black', tolower(raw_persons_race)) ~ 'black',
      grepl('african', tolower(raw_persons_race)) ~ 'black',
      grepl('white', tolower(raw_persons_race)) ~ 'white',
      grepl('caucasian', tolower(raw_persons_race)) ~ 'white'
    ))
  } else{
      x <- x %>% mutate(race_newcode = NA)
  }
  x
}

# keep_race_cols <- function(x){
#   which_cols <- colnames(x)[grepl('race', colnames(x))]
#   x[,which_cols]
# }
# 
# all_colnames <- list.files(here('data', 'openpolicingproject'))[4] %>%
#   map_dfr(~ readRDS(here('data', 'openpolicingproject', .x)) %>% keep_race_cols())
# colnames(all_colnames)
# glimpse(all_colnames)

files_to_process <- places_present_to_donwload %>%
  distinct(state, place) %>%
  mutate(place2 = glue('{place} county')) %>%
  inner_join(
    sc %>% distinct(FIPS, state, FIPS, county) %>%
      mutate(county = tolower(county), state = tolower(state)),
    by = c('place2' = 'county', 'state')
  )

files_to_handle <- places_present_to_donwload %>%
  mutate(filename = glue('{state}_{str_replace(place, " ", "_")}.rds')) %>%
  inner_join(files_to_process) %>%
  pull(filename)

pp <- files_to_handle[files_to_handle %in% list.files(here('data', 'openpolicingproject'))][1] %>%
  map_dfr(~ mutate(readRDS(here('data', 'openpolicingproject', .x)), 
                   state = str_split(.x, '_')[[1]][1],
                   place = str_remove(str_replace(str_split(.x, '_')[[1]][2], '_', ' '), '.rds')) %>%
            process_pp()) #%>%
            #select(race_newcode, date, state, place))
pp <- pp %>% mutate(data_year = year(date)) %>%
  filter(!is.na(race_newcode))
pp %>% count(state, place)

pp <- pp %>% inner_join(places_present_wt %>% 
                    group_by(state, place) %>%
                    summarise(start_nibrs_year = min(data_year), 
                              end_nibrs_year = max(data_year))) %>%
  filter(data_year >= start_nibrs_year & data_year <= end_nibrs_year)

pp_summary <- pp %>%
  group_by(data_year, state, place) %>%
  summarise(
    stops = n(),
    stops_black = sum(race_newcode == 'black'),
    stops_white = sum(race_newcode == 'white')
  )












