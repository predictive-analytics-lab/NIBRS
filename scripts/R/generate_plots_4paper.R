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
library(tidyr)
# library(cowplot)

source(here("scripts", "R", "utils_plot.R"))

sr <- vroom(here('data', 'output', "selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty.csv"))
# sr <- vroom(here('data', 'output', "selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty_buying.csv"))
# sr <- vroom(here("data", "misc", "county_coverage"))
sc <- fips_codes %>%
  mutate(FIPS = glue("{state_code}{county_code}"))

# figure 1 ----
cols <- c("#D55E00", "#CC79A7", "#F0E442", "#56B4E9", "#009E73")
opacity <- c("", "E6", "80", "4D")
all_cols <- cols %>%
  purrr::map(~ rev(paste0(.x, opacity))) %>%
  unlist()
all_cols <- tibble(cols = all_cols, code = letters[1:20])

sr <- sr %>%
  mutate(selection_ratio = ifelse(is.infinite(selection_ratio), NA, selection_ratio)) %>%
  mutate(rel_err = abs(log(selection_ratio) / var_log))

data("county.fips")
us_county <- map_data("county") %>%
  mutate(polyname = glue("{region},{subregion}")) %>%
  inner_join(county.fips, by = "polyname")

# get quartiles
quartiles <- sr %>%
  summarise(quant = quantile(
    rel_err,
    probs = c(0.25, 0.5, 0.75)
  )) %>%
  pull(quant)

sr <- sr %>%
  mutate(quartile_alpha = assign_quartile(
    rel_err,
    quartiles
  )) %>%
  mutate(sr_binned = case_when(
    selection_ratio < 0.8 ~ "S<0.8",
    selection_ratio >= 0.8 & selection_ratio < 1.25 ~ "0.8\u2264 S < 1.25",
    selection_ratio >= 1.25 & selection_ratio < 2 ~ "1.25\u2264 S < 2",
    selection_ratio >= 2 & selection_ratio < 5 ~ "2\u2264 S < 5",
    selection_ratio > 5 ~ "S \u2265 5"
  )) %>%
  mutate(
    color_code = factor(map_to_colors(quartile_alpha, sr_binned), levels = letters[1:20]) # %>%
  )

sr_x_county <- us_county %>%
  left_join(sr %>%
    mutate(FIPS = as.numeric(FIPS)), by = c("fips" = "FIPS"))

p <- sr_x_county %>%
  # bind_rows(sr_x_county) %>%
  # mutate(color_code = factor(color_code, levels = letters[1:20])) %>%
  ggplot(
    data = .,
    mapping = aes(
      x = long, y = lat,
      group = group,
      fill = color_code
    )
  ) +
  geom_polygon(color = "white", size = 0.2) +
  scale_fill_manual("Selection ratio S",
    values = all_cols$cols,
    labels = all_cols$code,
    na.value = "gray90",
    drop = FALSE
  ) +
  # theme_classic() +
  theme_void() +
  theme(
    axis.title = element_blank(),
    axis.text.x = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks = element_blank()
  ) +
  guides(fill = guide_legend("", ncol = 4, byrow = TRUE)) +
  theme(legend.text = element_blank())
p
# adj <- 0.015
# (p + theme(plot.margin=unit(c(0,1.4,0,0),"cm"))) %>%
# ggdraw(.) +
#   draw_text(text = 'S \u2265 5',x=0.97-adj,y=0.65, size = 10) +
#   draw_text(text = '2 \u2264 S < 5',x=0.97-adj,y=0.58, size = 10) +
#   draw_text(text = '1.25 \u2264 S < 2',x=0.985-adj,y=0.51, size = 10) +
#   draw_text(text = '0.8 \u2264 S < 1.25',x=0.99-adj,y=0.445, size = 10) +
#   draw_text(text = 'S > 0.8',x=0.97-adj,y=0.38, size = 10)
dir.create(here("scripts", "R", "plots"))
ggsave(here("scripts", "R", "plots", "selection_ratio_by_county.pdf"),
  height = 10, width = 16,
  device = cairo_pdf
)


# figure 2A ----
# https://en.wikipedia.org/wiki/Coefficient_of_variation

sr <- vroom(here("data", "output", "selection_ratio_county_2012-2019_wilson.csv")) %>%
  mutate(selection_ratio = ifelse(is.infinite(selection_ratio), NA, selection_ratio))
sr_median <- sr %>%
  group_by(FIPS) %>%
  mutate(number_of_years_reporting = ifelse(length(unique(year)) == 8, 1, 0)) %>%
  filter(number_of_years_reporting == 1) %>%
  ungroup() %>%
  group_by(year) %>%
  summarise(selection_ratio = median(selection_ratio)) %>%
  mutate(year = as.factor(year))
sr_to_plot <- sr %>%
  group_by(FIPS) %>%
  mutate(number_of_years_reporting = ifelse(length(unique(year)) == 8, 1, 0)) %>%
  filter(number_of_years_reporting == 1) %>%
  mutate(year = as.factor(year)) %>%
  mutate(coefvar = ci / selection_ratio) %>%
  # ggplot(aes(year, log(selection_ratio), alpha = 1/coefvar)) +
  # geom_density() +
  # ylim(0,5) +
  # theme_bw()
  drop_na()

sr_to_plot %>%
  # ggplot(aes(x = log(selection_ratio), y = as.factor(year))) + #, fill = stat(x))) +
  # geom_density_ridges2(fill = 'white') +
  ggplot(aes(x = log(selection_ratio), y = year, fill = factor(stat(quantile)))) +
  # geom_density_ridges_gradient() +
  # xlim(-1,3.5) +
  # theme_bw() +
  xlab("Log of selection ratio") +
  ylab("Year") +
  # scale_fill_viridis_c('', option = "C") +
  stat_density_ridges(
    geom = "density_ridges_gradient", calc_ecdf = TRUE,
    quantiles = 4, quantile_lines = TRUE
  ) +
  scale_fill_viridis_d(name = "Quartiles") +
  theme_bw()
# TODO: change colors
ggsave(here("scripts", "R", "plots", "density_sr_by_year_colored.pdf"))
sr_to_plot %>%
  # ggplot(aes(x = log(selection_ratio), y = as.factor(year))) + #, fill = stat(x))) +
  # geom_density_ridges2(fill = 'white') +
  ggplot(aes(x = log(selection_ratio), y = year, fill = factor(stat(quantile)))) +
  # geom_density_ridges_gradient() +
  # xlim(-1,3.5) +
  # theme_bw() +
  xlab("Log of selection ratio") +
  ylab("Year") +
  # scale_fill_viridis_c('', option = "C") +
  stat_density_ridges(
    geom = "density_ridges_gradient", calc_ecdf = TRUE,
    quantiles = 4, quantile_lines = TRUE
  ) +
  scale_fill_viridis_d(name = "Quartiles") +
  theme_bw()
ggsave(here("scripts", "R", "plots", "density_sr_by_year.pdf"))


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

# figure 2B ----
coef_year <- sr %>%
  inner_join(sc) %>%
  group_by(state) %>%
  mutate(
    years_reporting = length(unique(year)),
    n_counties = length(unique(FIPS))
  ) %>%
  filter(years_reporting >= 3 & n_counties >= 10) %>%
  droplevels() %>%
  ungroup() %>%
  group_by(state_name) %>%
  summarise(
    n_counties = length(unique(FIPS)),
    coef_year = tidy(lm(selection_ratio ~ year)) %>%
      filter(term == "year") %>% pull(estimate)
  )

us_states <- map_data("state") %>% left_join(coef_year %>% mutate(state = tolower(state_name)),
  by = c("region" = "state")
)
ggplot(
  data = us_states,
  mapping = aes(
    x = long, y = lat,
    group = group,
    fill = coef_year
  )
) +
  geom_polygon(color = "white", size = 0.02) +
  scale_fill_gradientn(
    colours = terrain.colors(10),
    na.value = "gray90",
    guide = guide_colourbar(
      title = "Coefficient's \nestimate",
      label.hjust = 1,
      barwidth = 1.2, barheight = 10,
      label.position = "left"
    )
  ) +
  theme_classic() +
  theme(
    axis.title = element_blank(),
    axis.text.x = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks = element_blank()
  )


# figure 3 ----

sr <- vroom(here("data", "output", "selection_ratio_county_2010-2019_bootstraps_1000_poverty.csv")) %>%
  mutate(rel_err = abs(log(selection_ratio) / sqrt(var_log)))
sr_arrest <- vroom(here("data", "output", "selection_ratio_county_2010-2019_bootstraps_1000_poverty_arrests.csv")) %>%
  distinct(arrests, FIPS, year)
sr <- sr %>% inner_join(sr_arrest)
nibrs <- vroom(here('data', 'NIBRS', 'raw', 'cannabis_allyears_allincidents.csv')) %>%
  filter(!other_offense) %>%
  filter(other_criminal_act_count == 0) %>%
  filter(cannabis_count > 0)
lc <- vroom(here('data', 'misc', 'LEAIC.tsv')) %>%
  distinct(STATENAME, FIPS, ORI9) %>%
  rename(state = STATENAME)

#
legalised <- tribble(
  ~state_code, ~year_legal, ~ month_legal, ~event,
  "08", 2012, 12, 'legalized', # colorado
  "25", 2016, 12, 'legalized', # ma
  "26", 2018, 12, 'legalized', # mi
  "41", 2015, 10, 'legalized',#  or 
  "50", 2013, 6, 'decriminalized', # vt
  "50", 2018, 6, 'legalized', # vt
  "53", 2013, 12, 'legalized' # wa
)

# take the weighted average
total_n_counties <- sc %>%
  inner_join(legalised) %>%
  group_by(state_name) %>%
  summarise(total_n_counties = length(unique(FIPS)))
#' tb_plot <- sr %>%
#'   inner_join(sc) %>%
#'   inner_join(legalised) %>%
#'   mutate(year_shift = year - year_legal) %>%
#'   group_by(FIPS) %>%
#'   mutate(years_reporting = length(unique(year))) %>%
#'   filter(years_reporting == 10) %>%
#'   group_by(year, year_shift, state_name, year_legal) %>%
#'   summarise(
#'     median_log_selection_ratio = median(log(selection_ratio)),
#'     mean_selection_ratio = mean(selection_ratio * frequency) / sum(frequency),
#'     mean_log_selection_ratio = sum(log(selection_ratio) * frequency) / sum(frequency),
#'     mean_log_selection_ratio_lb = mean_log_selection_ratio - 1.96 * sqrt(1/sum(rel_err)^2 * sum(rel_err^2 * var_log)),
#'     mean_log_selection_ratio_ub = mean_log_selection_ratio + 1.96 * sqrt(1/sum(rel_err)^2 * sum(rel_err^2 * var_log)),
#'     #mean_log_selection_ratio_lb = mean_log_selection_ratio - 1.96 * sqrt(1/sum(frequency)^2 * sum(frequency^2 * var_log)),
#'     #mean_log_selection_ratio_ub = mean_log_selection_ratio + 1.96 * sqrt(1/sum(frequency)^2 * sum(frequency^2 * var_log)),
#'     incidents = sum(incidents),
#'     log_incidents_per100k = log(sum(incidents) / sum(frequency) * 1e5),
#'    # log_arrests_per100k = log(sum(arrests) / sum(frequency) * 1e5),
#'     n_counties = length(unique(FIPS))
#'   ) %>%
#'   inner_join(total_n_counties) %>%
#'   ungroup %>%
#'   # compute n of counties for each state (must be stable across years)
#'   mutate(state_name = glue('{state_name} ({n_counties}/{total_n_counties} counties)')) %>%
#'   pivot_longer(cols = c('log_incidents_per100k',
#'                         #'log_arrests_per100k',
#'                         'mean_log_selection_ratio'),
#'                names_to = 'Variable', values_to = 'value')

sr_selected_counties <- sr %>%
  inner_join(sc) %>%
  #inner_join(legalised) %>%
  #mutate(year_shift = year - year_legal) %>%
  group_by(FIPS) %>%
  mutate(years_reporting = length(unique(year))) %>%
  filter(years_reporting == 10)
tb_plot_sr <- sr_selected_counties %>%
  group_by(year, state_name) %>%
  summarise(
    mean_log_selection_ratio = sum(log(selection_ratio) * frequency) / sum(frequency),
    mean_log_selection_ratio_lb = mean_log_selection_ratio - 1.96 * sqrt(1/sum(rel_err)^2 * sum(rel_err^2 * var_log)),
    mean_log_selection_ratio_ub = mean_log_selection_ratio + 1.96 * sqrt(1/sum(rel_err)^2 * sum(rel_err^2 * var_log)),
    #incidents = sum(incidents),
    #log_incidents_per100k = log(sum(incidents) / sum(frequency) * 1e5),
    # log_arrests_per100k = log(sum(arrests) / sum(frequency) * 1e5),
    n_counties = length(unique(FIPS))
  ) %>%
  inner_join(total_n_counties) %>%
  ungroup %>%
  # compute n of counties for each state (must be stable across years)
  mutate(state_name_wc = glue('{state_name} ({n_counties}/{total_n_counties} counties)')) %>%
  mutate(month_num = 6)


nibrs <- nibrs %>%
  inner_join(lc, by = c('ori' = 'ORI9'))

incidents_by_month <- nibrs %>%
  inner_join(sr_selected_counties %>% distinct(FIPS)) %>% # only counties that have always reported
  rename(year = data_year) %>%
  group_by(year, FIPS, month_num) %>%
  summarise(incidents = n())

arrests_by_month <- nibrs %>%
  inner_join(sr_selected_counties %>% distinct(FIPS)) %>% # only counties that have always reported
  rename(year = data_year) %>%
  group_by(year, FIPS, month_num) %>%
  filter(arrest_type_name != "No Arrest") %>%
  summarise(arrests = n()) 

incidents_by_month = incidents_by_month %>% left_join(arrests_by_month, by=c("year" = "year", "FIPS" = "FIPS", "month_num" = "month_num")) %>% replace(is.na(.), 0)

incidents_by_month = incidents_by_month %>%
  inner_join(sc) %>%
  inner_join(sr %>% distinct(year, FIPS, frequency)) %>%
  group_by(year, state_name, month_num) %>%
  summarise(log_incidents_per100k = log(sum(incidents) / sum(frequency) * 1e5), log_arrests_per100k = log(sum(arrests) / sum(frequency) * 1e5)) %>%
  inner_join(tb_plot_sr %>% distinct(state_name, state_name_wc))

glimpse(arrests_by_month)
glimpse(incidents_by_month)
glimpse(tb_plot_sr)

incidents_by_month

tb_plot <- tb_plot_sr %>%
  rename(value = mean_log_selection_ratio) %>%
  mutate(variable = 'mean_log_selection_ratio') %>%
  bind_rows(incidents_by_month %>%
              rename(value = log_incidents_per100k) %>%
              mutate(variable = 'log_incidents_per100k')) %>%
  bind_rows(incidents_by_month %>%
              rename(value = log_arrests_per100k) %>%
              mutate(variable = 'arrest_rate')) %>%
  mutate(year_month = as.character(glue('{month_num}/01/{year}'))) %>%
  mutate(date = as.Date(year_month, format = '%m/%d/%Y'))

legalised <- legalised %>%
  mutate(date = as.Date(glue('{month_legal}/01/{year_legal}'), format = '%m/%d/%Y')) %>%
  inner_join(sc %>% distinct(state_name, state_code)) %>%
  inner_join(tb_plot_sr %>% distinct(state_name, state_name_wc))

tb_plot %>%
  # plot
  mutate(variable = case_when(
    variable == 'log_incidents_per100k' ~ 'Log of monthly incidents per 100,000 people',
    variable == 'arrest_rate' ~ 'Log of monthly arrests per 100,000 people',
    variable == 'mean_log_selection_ratio' ~ "Mean of log selection ratio weighted by the\ninverse of the relative standard deviation"
  )) %>%
  ggplot(aes(x = date, y = value, col = variable)) +
  geom_line() +
  theme_bw() +
  geom_ribbon(aes(ymin = mean_log_selection_ratio_lb, ymax = mean_log_selection_ratio_ub), 
              fill = 'blue', alpha = 0.1, show.legend = FALSE,
              colour = NA) + 
  xlab("Year") +
  geom_vline(data = legalised, aes(xintercept = date, col = event), linetype = "dashed", alpha = 0.5) +
  ylab("Mean log selection ratio") +
  facet_wrap(~state_name, ncol = 2) + 
  scale_x_continuous(breaks = seq(2010, 2018, by = 2)) + 
  scale_color_manual(values = c("#D55E00", "#0072B2", "black", 'forestgreen', 'darkred')) + 
  ylab('Value') + 
  theme(legend.position="bottom", legend.title = element_blank(), 
        legend.box="vertical") +
  scale_x_date(date_breaks = '2 years', 
               date_minor_breaks = '1 year',
               #labels = seq(2011, 2019, by = 2),
               date_labels = "%Y") + 
  guides(color = guide_legend(nrow = 2, byrow = TRUE))
ggsave(here("scripts", "R", "plots", "sr_by_year_legalized.pdf"), height = 8, width = 6.5)


tb_plot %>%
 # filter(Variable == 'mean_log_selection_ratio') %>%
  filter(variable == 'log_incidents_per100k') %>%
  group_by(state_name) %>%
  summarise(out = list(lm(value ~ year) %>% tidy())) %>%
  unnest(out) %>%
  filter(term == 'year')
  
# 
# p_inc <- sr %>%
#   inner_join(sc) %>%
#   inner_join(legalised) %>%
#   # mutate(year_shift = year - year_legal) %>%
#   group_by(FIPS) %>%
#   mutate(years_reporting = length(unique(year))) %>%
#   filter(years_reporting == 10) %>%
#   group_by(year, state_name, year_legal) %>%
#   summarise(
#     incidents = sum(incidents) / sum(frequency) * 1e5,
#     n_counties = length(unique(FIPS))
#   ) %>%
#   ungroup %>%
#   # compute n of counties for each state (must be stable across years)
#   mutate(state_name = glue('{state_name} (# counties={n_counties})')) %>%
#   # plot
#   ggplot(aes(year, log(incidents))) +
#   geom_line() +
#   theme_bw() +
#   geom_vline(aes(xintercept = year_legal), linetype = "dashed") +
#   xlab("Year") +
#   ylab("Log of incidents per 100,000 people") +
#   facet_wrap(~state_name)  + 
#   scale_x_continuous(breaks = seq(2010, 2018, by = 2))
# ggsave(here("scripts", "R", "plots", "incidents_per_100000people.pdf"))


# additional figures ----







