renv::restore()

library(dplyr)
library(purrr)
library(Cairo)
library(vroom)
library(here)
library(tigris)
library(ggplot2)
library(glue)
library(broom)
library(maps)
library(ggridges)
library(cowplot)

sr <- vroom(here('scripts', 'R', "selection_ratio_county_2017-2019_grouped_wilson_poverty.csv"))
sc <- fips_codes %>%
    mutate(FIPS = glue('{state_code}{county_code}'))

# figure 1 ----

# bins: based on actual value / based on lower bound
# transparency: ~ confidence
# sr_x_county <- sr %>%
#     filter(year >= 2017) %>%
#     group_by(FIPS) %>%
#
#     rename(sr = selection_ratio)
# sr_x_county <- sr %>%
#     filter(year >= 2017) %>%
#     group_by(FIPS) %>%
#     summarise(
#         black_users = sum(black_users),
#         white_users = sum(white_users),
#         black_incidents = sum(black_incidents),
#         white_incidents = sum(white_incidents),
#         incidents = sum(incidents),
#         users = sum(black_users + white_users)
#     ) %>%
#     mutate(sr = black_incidents / white_incidents * white_users / black_users,
#            diff = black_incidents / black_users - white_incidents / white_users)

sr_x_county <- sr %>%
  rename(sr = selection_ratio) %>%
  mutate(sr = sr - ci)


data('county.fips')
us_county <- map_data("county") %>%
    mutate(polyname = glue('{region},{subregion}')) %>%
    inner_join(county.fips, by = 'polyname')

sr_x_county <- us_county %>%
    left_join(sr_x_county %>%
        mutate(FIPS = as.numeric(FIPS)), by = c('fips' = 'FIPS')) %>%
    mutate(sr = ifelse(is.infinite(sr), NA, sr))


fct_order <- c('S<0.8', '0.8\u2264 S < 1.25',
               '1.25\u2264 S < 2',
               '2\u2264 S < 5',
               'S \u2265 5')
cols <- c( 'S \u2265 5' = "red", '2\u2264 S < 5' = "purple",
           '1.25\u2264 S < 2' = "yellow",'0.8\u2264 S < 1.25'= "blue",
           'S<0.8' = "green")
assign_quartile <- function(x, quartiles){
  x <- case_when(
    x > quartiles[3] ~ 4,
    x > quartiles[2] ~ 3,
    x > quartiles[1] ~ 2,
    x <= quartiles[1] ~ 1
  )
  x / 4
}
quartiles <- sr %>% summarise(quant = quantile(selection_ratio/ci, 
                                       probs = c(0.25, 0.5, 0.75))) %>%
  pull(quant)

sr_x_county %>%
  mutate(quartile = assign_quartile(sr/ci, quartiles)) %>%
  mutate(sr_binned = case_when(
        sr < 0.8 ~ 'S<0.8',
        sr >= 0.8 & sr < 1.25 ~ '0.8\u2264 S < 1.25',
        sr >= 1.25 & sr < 2 ~ '1.25\u2264 S < 2',
        sr >= 2 & sr < 5 ~'2\u2264 S < 5',
        sr > 5 ~ 'S \u2265 5'
    )) %>%
    ggplot(data = .,
       mapping = aes(x = long, y = lat,
                     group = group,
                     fill = factor(sr_binned, levels = fct_order),
                     alpha = quartile)) +
    geom_polygon(color = "white", size = 0.2) +
    # add state lines?
    scale_fill_manual('Lower bound for \nselection ratio S', values = cols, na.value = 'gray90') +
    theme_classic() +
    theme(
        axis.title = element_blank(),
        axis.text.x = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks = element_blank()) +
    scale_alpha(guide = 'none')
ggsave(here('scripts', 'R', 'plots', 'selection_ratio_by_county.pdf'))

# opacity https://gist.github.com/lopspower/03fb1cc0ac9f32ef38f4

# bvColors=c(
#            "#FA2F03","#CB5C45","#A35C4D","#6A4942","#6A4942",
#            "#FA2F03E6","#CB5C45E6","#A35C4DE6","#6A4942E6","#6A4942E6",
#            "#FF2F0380","#CB5C4580","#A35C4D80","#6A494280","#6A494280")
# legendGoal=melt(matrix(1:15,nrow=5, byrow = TRUE))
# test <-ggplot(legendGoal, aes(Var2,Var1,fill = as.factor(value)))+ geom_tile()
# test <- test + scale_fill_manual(values=bvColors,drop=FALSE)
# test <-test+guides(fill = guide_legend('', ncol = 3, byrow = TRUE))
# test <-test + theme(legend.text=element_blank())
# test<-ggdraw(test) + draw_text(text = "S > 0.8 \n 0.8\u2264 S < 1.25\n 1.25\u2264 S < 2\n 2\u2264 S < 5\n S \u2265 5",x=0.8,y=0.5)
# ##+ draw_text(text = "More Var 1 -->",x=0.84,y=0.5,angle=270)
# test


# figure 2 left ----
# use mu / sigma
# https://en.wikipedia.org/wiki/Coefficient_of_variation
sr <- vroom(here('data', 'output', "selection_ratio_county_2012-2019_wilson.csv")) %>%
    mutate(selection_ratio = ifelse(is.infinite(selection_ratio), NA, selection_ratio))
sr_median <- sr %>%
    group_by(FIPS) %>%
    mutate(number_of_years_reporting = ifelse(length(unique(year))==8, 1, 0)) %>%
    filter(number_of_years_reporting == 1) %>%
    ungroup %>% group_by(year) %>%
    summarise(selection_ratio = median(selection_ratio)) %>%
    mutate(year = as.factor(year))
sr_to_plot <- sr %>%
    group_by(FIPS) %>%
    mutate(number_of_years_reporting = ifelse(length(unique(year))==8, 1, 0)) %>%
    filter(number_of_years_reporting == 1) %>%
    mutate(year = as.factor(year)) %>%
    mutate(coefvar = ci/selection_ratio) %>%
    # ggplot(aes(year, log(selection_ratio), alpha = 1/coefvar)) +
    # geom_density() +
    # ylim(0,5) +
    # theme_bw()
    drop_na()

sr_to_plot %>%
    # ggplot(aes(x = log(selection_ratio), y = as.factor(year))) + #, fill = stat(x))) +
    # geom_density_ridges2(fill = 'white') +
    ggplot(aes(x = log(selection_ratio), y = year, fill = factor(stat(quantile)))) +
    #geom_density_ridges_gradient() +
    #xlim(-1,3.5) +
    #theme_bw() +
    xlab('Log of selection ratio') + ylab('Year') +
    #scale_fill_viridis_c('', option = "C") +
    stat_density_ridges(
        geom = "density_ridges_gradient", calc_ecdf = TRUE,
        quantiles = 4, quantile_lines = TRUE
    ) +
    scale_fill_viridis_d(name = "Quartiles") +
    theme_bw()
# TODO: change colors
ggsave(here('scripts', 'R', 'plots','density_sr_by_year_colored.pdf'))
sr_to_plot %>%
    # ggplot(aes(x = log(selection_ratio), y = as.factor(year))) + #, fill = stat(x))) +
    # geom_density_ridges2(fill = 'white') +
    ggplot(aes(x = log(selection_ratio), y = year, fill = factor(stat(quantile)))) +
    #geom_density_ridges_gradient() +
    #xlim(-1,3.5) +
    #theme_bw() +
    xlab('Log of selection ratio') + ylab('Year') +
    #scale_fill_viridis_c('', option = "C") +
    stat_density_ridges(
        geom = "density_ridges_gradient", calc_ecdf = TRUE,
        quantiles = 4, quantile_lines = TRUE
    ) +
    scale_fill_viridis_d(name = "Quartiles") +
    theme_bw()
ggsave(here('scripts', 'R', 'plots', 'density_sr_by_year.pdf'))


# sr %>%
#     group_by(FIPS) %>%
#     #mutate(number_of_years_reporting = ifelse(length(unique(year))==8, 1, 0)) %>%
#     #filter(number_of_years_reporting == 1) %>%
#     mutate(year = as.factor(year)) %>%
#     mutate(coefvar = ci/selection_ratio) %>%
#     ggplot(aes(factor(year, levels = 2012:2019), log(selection_ratio))) +
#     geom_boxplot() +
#     theme_bw()

# sr %>%
#     group_by(FIPS) %>%
#     mutate(number_of_years_reporting = ifelse(length(unique(year))==8, 1, 0)) %>%
#     #filter(number_of_years_reporting == 1) %>%
#     mutate(coefvar = ci/selection_ratio) %>%
#     lm(selection_ratio ~ year, data = .)


# sr %>%
#     mutate(log_sr = log(selection_ratio)) %>%
#     ggplot(aes(selection_ratio, fill = year_after_2015)) +
#     xlim(0,10) +
#     geom_density(alpha = 0.3) +
#     #facet_wrap(~ year) +
#     theme_bw()

# figure 2 right ----
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
        coef_year = tidy(lm(selection_ratio ~ year)) %>%
            filter(term == 'year') %>% pull(estimate)
    )

us_states <- map_data("state") %>% left_join(coef_year %>% mutate(state = tolower(state_name)),
                         by = c('region' = 'state'))
ggplot(data = us_states,
       mapping = aes(x = long, y = lat,
                     group = group,
                        fill = coef_year)) +
    geom_polygon(color = "white", size = 0.02) +
    scale_fill_gradientn(colours = terrain.colors(10),
                         na.value="gray90",
                         guide = guide_colourbar(title = "Coefficient's \nestimate",
                                                 label.hjust = 1,
                                                 barwidth = 1.2, barheight = 10,
                                                 label.position = "left"))  +
    theme_classic() +
    theme(
        axis.title = element_blank(),
        axis.text.x = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks = element_blank())


# figure 3 ----
sr <- vroom(here('data', 'output', "selection_ratio_county_2012-2019.csv"))
# MA 2016
# OR 2014
# VT 2013

sr %>%
    inner_join(sc) %>%
    filter(state %in% c('MA', 'OR', 'VT')) %>%
    mutate(year = case_when(
        state == 'MA' ~ year - 2016,
        state == 'VT' ~ year - 2013,
        state == 'OR' ~ year - 2014
    )) %>%
    group_by(FIPS) %>%
    mutate(number_of_years_reporting = ifelse(length(unique(year))==8, 1, 0)) %>%
    filter(number_of_years_reporting == 1) %>%
    group_by(year, state) %>%
    summarise(selection_ratio = median(selection_ratio)) %>%
    # summarise(selection_ratio = sum(black_incidents) / sum(white_incidents) *
    #              sum(white_users) / sum(black_users),
              #median_selection_ratio = median(black_incidents / black_users * white_users / white_incidents / frequency * 1e6),
              #) %>%
    ggplot(aes(x = year, y = log(selection_ratio), col = state)) +
    #geom_density_ridges2(fill = 'white') +
    geom_line() +
    theme_bw() +
    xlab('Year') +
    geom_vline(xintercept = 0, linetype = 'dashed') +
    ylab('Median of log selection ratio') #+
    #facet_wrap(~ state)
    #xlim(-2,3.5)
    #     geom_jitter() +
    #     facet_wrap(~ state) +
    # ylim(-1,3.5)
ggsave(here('scripts', 'R', 'plots', 'sr_by_year_legalized.pdf'))

# sr %>%
#     inner_join(sc) %>%
#     filter(state %in% c('MA', 'OR', 'VT')) %>%
#     mutate(year = case_when(
#         state == 'MA' ~ year - 2016,
#         state == 'VT' ~ year - 2013,
#         state == 'OR' ~ year - 2014
#     )) %>%
#     group_by(FIPS) %>%
#     mutate(number_of_years_reporting = ifelse(length(unique(year))==8, 1, 0)) %>%
#     filter(number_of_years_reporting == 1) %>%
#     #group_by(year, state) %>%
#     #summarise(selection_ratio = median(selection_ratio)) %>%
#     # summarise(selection_ratio = sum(black_incidents) / sum(white_incidents) *
#     #              sum(white_users) / sum(black_users),
#     #median_selection_ratio = median(black_incidents / black_users * white_users / white_incidents / frequency * 1e6),
#     #) %>%
#     ggplot(aes(as.factor(year), selection_ratio, col = state)) +
#     geom_ridges() +
#     theme_bw() +
#     xlab('Year') +
#     ylab('Median of log selection ratio') + facet_wrap(~ state)


sr %>%
    group_by(FIPS) %>%
    # mutate(number_of_years_reporting = ifelse(length(unique(year))==8, 1, 0)) %>%
    # filter(number_of_years_reporting == 1) %>%
    inner_join(sc) %>%
    filter(state %in% c('MA', 'OR', 'VT')) %>%
    mutate(year = case_when(
        state == 'MA' ~ year - 2016,
        state == 'VT' ~ year - 2013,
        state == 'OR' ~ year - 2014
    )) %>%
    group_by(year, state) %>%
    #summarise(selection_ratio = median(selection_ratio)) %>%
    summarise(
              # black_incidents = sum(black_incidents / frequency * 1e6),
              # white_incidents = sum(white_incidents / frequency * 1e6),
              incidents = sum(incidents) / sum(frequency) * 1e5
    ) %>%
    ggplot(aes(year, log(incidents), col = state)) +
    geom_line() +
    theme_bw() +
    geom_vline(xintercept = 0, linetype = 'dashed') +
    xlab('Year') +
    ylab('Log of incidents per 100,000 people')
ggsave(here('scripts', 'R', 'plots', 'incidents_per_100000people.pdf'))



# sr %>%
#     group_by(FIPS) %>%
#     mutate(number_of_years_reporting = ifelse(length(unique(year))==8, 1, 0)) %>%
#     filter(number_of_years_reporting == 1) %>%
#     inner_join(sc) %>%
#     filter(state %in% c('MA', 'OR', 'VT')) %>%
#     mutate(year = case_when(
#         state == 'MA' ~ year - 2016,
#         state == 'VT' ~ year - 2013,
#         state == 'OR' ~ year - 2014
#     )) %>%
#     group_by(year, state) %>%
#     #summarise(selection_ratio = median(selection_ratio)) %>%
#     summarise(selection_ratio = sum(black_users) / sum(white_users) *
#                   sum(white_incidents) / sum(black_incidents),
#               incidents = sum(incidents) / sum(frequency) * 1e5
#     ) %>%
#     mutate(mul = selection_ratio * incidents) %>%
#     ggplot(aes(year, log(mul), col = state)) +
#     geom_line() +
#     theme_bw() +
#     geom_vline(xintercept = 0, linetype = 'dashed') +
#     xlab('Year') +
#     ylab('Log of incidents * selection ratio')




# old code ----

sr2 <- vroom(here('data', 'output', "selection_ratio_county_2012-2019.csv"),
            col_types = cols())
sr %>% ungroup %>%
    mutate(sr = (black_incidents/black_users)/(white_incidents/white_users),
           sr_adj = ((black_incidents+0.1)/(black_users))/((white_incidents+0.1)/(white_users))) %>%
    select(sr, sr_adj) %>%
    ggplot(aes(sr, sr_adj)) +
    geom_point() +
    xlim(0,200) +
    ylim(0,200)


sr %>% ungroup %>%
    mutate(sr = (black_incidents/black_users)/(white_incidents/white_users),
           sr_adj = ((black_incidents+0.1)/(black_users))/((white_incidents+0.1)/(white_users))) %>%
    filter(sr >= 200 & !is.infinite(sr)) %>%
    select(FIPS, sr, black_incidents, black_users, white_incidents, white_users)


vroom(here('data', 'output', "selection_ratio_county_2012-2019.csv"),
      col_types = cols()) %>%
    rename(sr = selection_ratio) %>%
    select(sr, FIPS, year) %>%
    inner_join(
        vroom(here('data', 'output', "selection_ratio_county_2012-2019_wilson.csv")) %>%
            rename(sr_wilson = selection_ratio) %>%
            select(sr_wilson, FIPS, year)
    ) %>%
    ggplot(aes(sr_wilson, sr)) +
    geom_point()

sr %>%
    mutate(alternative_sr = (black_incidents/black_users)/(white_incidents/white_users))




sr <- vroom(here('data', 'output', "selection_ratio_county_2012-2019.csv"))




