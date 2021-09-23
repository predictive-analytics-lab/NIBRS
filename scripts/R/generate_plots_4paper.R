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
#library(cowplot)

source(here('scripts', 'R', 'utils_plot.R'))

#sr <- vroom(here('data', 'output', "selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty.csv"))
#sr <- vroom(here('data', 'output', "selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty_buying.csv"))
sr <- vroom(here('data', 'output', "selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty_buying_outside.csv"))
sc <- fips_codes %>%
    mutate(FIPS = glue('{state_code}{county_code}'))

# figure 1 ----
cols <- c("#D55E00", "#CC79A7", "#F0E442","#56B4E9", "#009E73")
opacity <- c('', 'E6', '80', '4D')
all_cols <- cols %>% purrr::map(~ rev(paste0(.x, opacity))) %>%
  unlist()
all_cols <- tibble(cols = all_cols, code = letters[1:20])

sr <- sr %>%
  mutate(selection_ratio = ifelse(is.infinite(selection_ratio), NA, selection_ratio)) %>%
  mutate(rel_err = abs(log(selection_ratio) / var_log))

data('county.fips')
us_county <- map_data("county") %>%
  mutate(polyname = glue('{region},{subregion}')) %>%
  inner_join(county.fips, by = 'polyname')

# get quartiles
quartiles <- sr %>% 
  summarise(quant = quantile(
    rel_err, probs = c(0.25, 0.5, 0.75))) %>%
  pull(quant)

sr <- sr %>%
  mutate(quartile_alpha = assign_quartile(rel_err,
    quartiles)
  ) %>%
  mutate(sr_binned = case_when(
    selection_ratio < 0.8 ~ 'S<0.8',
    selection_ratio >= 0.8 & selection_ratio < 1.25 ~ '0.8\u2264 S < 1.25',
    selection_ratio >= 1.25 & selection_ratio < 2 ~ '1.25\u2264 S < 2',
    selection_ratio >= 2 & selection_ratio < 5 ~'2\u2264 S < 5',
    selection_ratio > 5 ~ 'S \u2265 5'
  )) %>%
  mutate(
    color_code = factor(map_to_colors(quartile_alpha, sr_binned),  levels = letters[1:20])#%>%
  )

sr_x_county <- us_county %>%
  left_join(sr %>%
              mutate(FIPS = as.numeric(FIPS)), by = c('fips' = 'FIPS'))

p <- sr_x_county %>%
  #bind_rows(sr_x_county) %>%
  #mutate(color_code = factor(color_code, levels = letters[1:20])) %>%
  ggplot(data = .,
         mapping = aes(x = long, y = lat,
                       group = group,
                       fill = color_code
         )) +
  geom_polygon(color = "white", size = 0.2) +
  scale_fill_manual('Selection ratio S', 
                    values = all_cols$cols, 
                    labels = all_cols$code,
                    na.value = 'gray90', 
                    drop = FALSE) +
  #theme_classic() +
  theme_void() + 
  theme(
    axis.title = element_blank(),
    axis.text.x = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks = element_blank()) +
  guides(fill = guide_legend('', ncol = 4, byrow = TRUE)) + 
  theme(legend.text=element_blank())
p
#adj <- 0.015
#(p + theme(plot.margin=unit(c(0,1.4,0,0),"cm"))) %>%
# ggdraw(.) + 
#   draw_text(text = 'S \u2265 5',x=0.97-adj,y=0.65, size = 10) + 
#   draw_text(text = '2 \u2264 S < 5',x=0.97-adj,y=0.58, size = 10) + 
#   draw_text(text = '1.25 \u2264 S < 2',x=0.985-adj,y=0.51, size = 10) + 
#   draw_text(text = '0.8 \u2264 S < 1.25',x=0.99-adj,y=0.445, size = 10) + 
#   draw_text(text = 'S > 0.8',x=0.97-adj,y=0.38, size = 10)
dir.create(here('scripts', 'R', 'plots'))
ggsave(here('scripts', 'R', 'plots', 'selection_ratio_by_county.pdf'),
       height = 10, width = 16,
       device=cairo_pdf)


sr_x_county %>%
  mutate(quartile_alpha = assign_quartile(
    #sr / ci, 
    sr / 10^var_log,
    quartiles)
  ) %>%
  mutate(sr_binned = case_when(
    sr < 0.8 ~ 'S<0.8',
    sr >= 0.8 & sr < 1.25 ~ '0.8\u2264 S < 1.25',
    sr >= 1.25 & sr < 2 ~ '1.25\u2264 S < 2',
    sr >= 2 & sr < 5 ~'2\u2264 S < 5',
    sr > 5 ~ 'S \u2265 5'
  )) %>%
  mutate(
    color_code = factor(map_to_colors(quartile_alpha, sr_binned),  levels = letters[1:20])#%>%
    #tibble(code = .)
  ) %>% count(color_code, sr_binned, quartile_alpha)


# opacity https://gist.github.com/lopspower/03fb1cc0ac9f32ef38f4

cols <- c("#D55E00", "#CC79A7", "#F0E442","#56B4E9", "#009E73")
opacity <- c('', 'E6', '80', '4D')
all_cols <- cols %>% purrr::map(~ paste0(.x, opacity)) %>%
  unlist()

library(reshape2)
legendGoal=melt(matrix(1:15,nrow=5, byrow = TRUE))
test <-ggplot(legendGoal, aes(Var2,Var1,fill = as.factor(value)))+ geom_tile()
test <- test + scale_fill_manual(values=all_cols,drop=FALSE)
test <-test+guides(fill = guide_legend('', ncol = 4, byrow = TRUE))
test <-test + theme(legend.text=element_blank())
#test<-ggdraw(test) + draw_text(text = "S > 0.8 \n 0.8\u2264 S < 1.25\n 1.25\u2264 S < 2\n 2\u2264 S < 5\n S \u2265 5",x=0.8,y=0.5)
##+ draw_text(text = "More Var 1 -->",x=0.84,y=0.5,angle=270)
test


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




