
library(vroom)
library(tidyverse)
library(here)

df <- vroom(here('data', 'NIBRS', 'incidents_processed_2019.csv'))
sr <- vroom(here('data', 'output', 'selection_ratio_county_2012-2019.csv'))

# df <- df %>%
#     inner_join(sr %>% distinct(FIPS, black_users, white_users), by = 'FIPS')

#df %>%
#    pivot_wider(names_from = race, names_prefix = 'incidents_', values_from = incidents)

# df <- df %>%
#     mutate(users = ifelse(race == 'black', black_users, white_users))

df %>%
    group_by(FIPS, race, location) %>%
    summarise(n_incidents = sum(incidents)) %>%
    inner_join(sr %>% distinct(FIPS, black_users, white_users), by = 'FIPS') %>%
    mutate(users = ifelse(race == 'black', black_users, white_users)) %>%
    ungroup %>%
    group_by(race, location) %>%
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




