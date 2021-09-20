
library(tidyverse)
library(vroom)
library(here)
library(tigris)
library(ggplot2)
library(glue)
library(broom)
library(maps)

sr <- vroom(here('data', 'output', "selection_ratio_county_2012-2019_wilson.csv"))
sc <- fips_codes %>%
    mutate(FIPS = glue('{state_code}{county_code}'))


# figure 2 left
# use mu / sigma
# https://en.wikipedia.org/wiki/Coefficient_of_variation
# sr %>%
#     group_by(FIPS) %>%
#     mutate(number_of_years_reporting = ifelse(length(unique(year))==8, 1, 0)) %>%
#     filter(number_of_years_reporting == 1) %>%
#     mutate(year = as.factor(year)) %>%
#     mutate(coefvar = ci/selection_ratio) %>%
#     ggplot(aes(year, log(selection_ratio), alpha = 1/coefvar)) +
#     geom_boxplot() +
#     ylim(0,5) +
#     theme_bw()

# sr %>%
#     group_by(FIPS) %>%
#     mutate(number_of_years_reporting = ifelse(length(unique(year))==8, 1, 0)) %>%
#     #filter(number_of_years_reporting == 1) %>%
#     mutate(coefvar = ci/selection_ratio) %>%
#     lm(selection_ratio ~ year, data = .)


sr %>%
    mutate(log_sr = log(selection_ratio)) %>%
    ggplot(aes(selection_ratio, fill = year_after_2015)) +
    xlim(0,10) +
    geom_density(alpha = 0.3) +
    #facet_wrap(~ year) +
    theme_bw()

# figure 2 right
coef_year <- sr %>%
    inner_join(sc) %>%
    group_by(state) %>%
    mutate(years_reporting = length(unique(year)),
           n_counties = length(unique(FIPS))) %>%
    filter(years_reporting >= 3 & n_counties >= 10) %>%
    droplevels() %>% ungroup %>%
    group_by(state_name) %>%
    summarise(
        n_counties = length(unique(FIPS)),
        coef_year = tidy(rlm(selection_ratio ~ year)) %>%
            filter(term == 'year') %>% pull(estimate)
    )

us_states <- map_data("state")
us_states <- us_states %>% left_join(coef_year %>% mutate(state = tolower(state_name)),
                         by = c('region' = 'state'))
ggplot(data = us_states,
       mapping = aes(x = long, y = lat,
                     group = group,
                        fill = coef_year)) +
    geom_polygon(color = "black") +
    theme_void() +
    scale_fill_gradientn(colours = terrain.colors(10)) +
    guides(fill = guide_legend(title = 'Coefficient \n estimate',
                       label.position = "left", label.hjust = 1))
    #scale_colour_gradient2() #+
    #guides(fill = FALSE)

