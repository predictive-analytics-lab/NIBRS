
library(vroom)
library(tidyverse)
library(here)

#df <- vroom(here('data', 'NIBRS', 'incidents_processed_2019.csv'))
df <- vroom(here('data', 'output', 'cannabis_2010-2019_allincidents_summary.csv'))
#sr <- vroom(here('data', 'output', 'selection_ratio_county_2012-2019.csv'))
lc <- vroom(here('data', 'output', 'LEAIC.tsv'))
sc <- vroom(here('data', 'output', 'urban_codes_x_county_2013.csv')) %>%
    mutate(is_metro = ifelse(urbancounty == 'Large metro' | urbancounty == 'Small metro', 'metro', 'nonmetro'))

sc <- lc %>% inner_join(sc) %>% rename(ori = ORI9) %>%
    distinct(ori, FIPS, is_metro)
df <- df %>% dplyr::inner_join(sc, by = 'ori')

# df <- df %>%
#     inner_join(sr %>% distinct(FIPS, black_users, white_users), by = 'FIPS')

#df %>%
#    pivot_wider(names_from = race, names_prefix = 'incidents_', values_from = incidents)

# df <- df %>%
#     mutate(users = ifelse(race == 'black', black_users, white_users))

# transform locations
all_locations <- str_split(str_remove_all(df %>% pull(location), "\\(|\\)|\\'"), pattern = '\\,')
df$all_locations_1 <- all_locations %>% purrr::map(~ .x[[1]]) %>% unlist() %>% tolower()
df$all_locations_2 <- all_locations %>% purrr::map(~ .x[[2]]) %>% unlist() %>% tolower()

df <- df %>%
    mutate(all_locations_1 = case_when(
        grepl('school', all_locations_1) ~ 'school',
        grepl('home', all_locations_1) ~ 'home',
        grepl('road', all_locations_1) ~ 'road',
        grepl('hotel', all_locations_1) ~ 'hotel',
        grepl('parking', all_locations_1) ~ 'parking',
        all_locations_1 != '' & all_locations_1 != ' ' ~  'other'
    ),
    all_locations_2 = case_when(
        grepl('school', all_locations_2) ~ 'school',
        grepl('home', all_locations_2) ~ 'home',
        grepl('road', all_locations_2) ~ 'road',
        grepl('hotel', all_locations_2) ~ 'hotel',
        grepl('parking', all_locations_2) ~ 'parking',
        all_locations_2 != '' & all_locations_2 != ' ' ~  'other'
    )
    )

df$n_locations <- 2 - is.na(df$all_locations_1) - is.na(df$all_locations_2)

df %>%
    filter(!other_offense) %>%
    filter(unique_drug_type_count == 1) %>%
    filter(other_criminal_act_count == 0) %>%
    group_by(FIPS, race, all_locations_1) %>%
    summarise(n_incidents = n()) %>%
    inner_join(sr %>% distinct(FIPS, black_users, white_users), by = 'FIPS') %>%
    mutate(users = ifelse(race == 'black', black_users, white_users)) %>%
    ungroup %>%
    group_by(race, all_locations_1) %>%
    summarise(
        prob = sum(n_incidents) / sum(users)
    ) %>%
    pivot_wider(names_from = race, values_from = prob) %>%
    mutate(sr = black / white)


# by state ----

df_x_state <- df %>%
    group_by(state, FIPS, race, location) %>%
    summarise(n_incidents = sum(incidents)) %>%
    inner_join(sr %>% distinct(FIPS, black_users, white_users), by = 'FIPS') %>%
    mutate(users = ifelse(race == 'black', black_users, white_users)) %>%
    ungroup %>%
    group_by(state, race, location) %>%
    summarise(
        n = sum(n_incidents),
        prob = sum(n_incidents) / sum(users)
    ) %>%
    filter(n >= 100) %>%
    select(-n) %>%
    pivot_wider(names_from = race, values_from = prob) %>%
    mutate(sr = black / white)

us_states <- map_data("state") %>% left_join(df_x_state %>% mutate(state = tolower(state)),
                                             by = c('region' = 'state'))

ggplot(data = us_states %>% drop_na(location, sr),
       mapping = aes(x = long, y = lat,
                     group = group,
                     fill = log(sr))) +
    geom_polygon(color = "gray90", size = 0.02) +
    scale_fill_gradientn(colours = terrain.colors(10),
                         na.value="gray90",
                         guide = guide_colourbar(title = "Log selection ratio",
                                                 label.hjust = 1,
                                                 barwidth = 1.2, barheight = 10,
                                                 label.position = "left")
                         )  +
    theme_classic() +
    theme(
        axis.title = element_blank(),
        axis.text.x = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks = element_blank()) +
    facet_wrap(~ location)


# legalization -----

# df_x_state_x_year <- df %>%
#     group_by(state, year, FIPS, race, location) %>%
#     summarise(n_incidents = sum(incidents)) %>%
#     inner_join(sr %>% distinct(FIPS, black_users, white_users), by = 'FIPS') %>%
#     mutate(users = ifelse(race == 'black', black_users, white_users)) %>%
#     ungroup %>%
#     group_by(state, race, location) %>%
#     summarise(
#         n = sum(n_incidents),
#         prob = sum(n_incidents) / sum(users)
#     ) %>%
#     filter(n >= 100) %>%
#     select(-n) %>%
#     pivot_wider(names_from = race, values_from = prob) %>%
#     mutate(sr = black / white)




