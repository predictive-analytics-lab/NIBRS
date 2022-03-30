library(dplyr)
library(vroom)
library(here)
library(tigris)
library(ggplot2)
library(broom)
library(maps)


source(here("scripts", "R", "utils_plot.R"))

sr <- vroom(here('data', 'output', "selection_ratio_county_2010-2019_bootstraps_1000_all.csv"))
coverage <- vroom(here('data', 'misc', "county_coverage.csv"))
coverage = coverage %>% filter(year == 2019) %>% group_by(FIPS) %>% summarise(cov = mean(coverage))

sc <- fips_codes %>%
  mutate(FIPS = glue("{state_code}{county_code}"))

cols <- c("#D55E00", "#CC79A7", "#F0E442", "#56B4E9", "#009E73")
opacity <- c("", "E6", "80", "4D")
all_cols <- cols %>%
  purrr::map(~ rev(paste0(.x, opacity))) %>%
  unlist()
all_cols <- tibble(cols = all_cols, code = letters[1:20])

cols <- c("#D55E00", "#CC79A7", "#F0E442", "#56B4E9", "#009E73")
all_cols_er <- cols %>%
  purrr::map(~ rev(paste0(.x, opacity))) %>%
  unlist()
all_cols_er <- tibble(cols = all_cols_er, code = letters[1:20])

sr <- sr %>%
  mutate(selection_ratio = ifelse(is.infinite(selection_ratio), NA, selection_ratio)) %>%
  mutate(rel_err = abs(log(selection_ratio) / var_log))

sr = sr %>% mutate(incidents = black_incidents + white_incidents, uses = black_users + white_users, incident_variance = black_users_variance + white_users_variance, uses_variance = black_users_variance + white_users_variance) %>% mutate(enforcement_rate = log(incidents / uses)) %>% mutate(enf_var = exp((1 / ((incidents ** 2) * incident_variance)) + (1 / ((uses ** 2) * uses_variance)))) %>% mutate(enf_rel_err = abs(enforcement_rate / enf_var),)


data("county.fips")
us_county <- map_data("county") %>%
  mutate(polyname = glue("{region},{subregion}")) %>%
  inner_join(county.fips, by = "polyname")

# sr = sr %>% inner_join(coverage, by="FIPS") %>% filter(cov > 0.8)

slopes = sr %>% mutate(impact = log(exp(enforcement_rate) * log(selection_ratio))) %>%  group_by(FIPS) %>% drop_na(impact) %>% do(tidy(lm(impact ~ year, data = .))) %>% filter(term == "year") %>% select(FIPS, slope = estimate)


slopes_x_county <- us_county %>% left_join(slopes %>% mutate(FIPS = as.numeric(FIPS)), by = c("fips" = "FIPS"))


p <- ggplot() +
  geom_polygon(data = slopes_x_county, aes(fill = slope, x = long, y = lat, group = group) , size=0, alpha=0.9) +
  theme_void() +
  scale_fill_viridis(name="Impact Gradient", guide = guide_legend(keyheight = unit(3, units = "mm"), keywidth=unit(12, units = "mm"), label.position = "bottom", title.position = 'top', nrow=1) ) +
  theme(
    text = element_text(color = "#22211d"),
    plot.background = element_rect(fill = "#f5f5f2", color = NA),
    panel.background = element_rect(fill = "#f5f5f2", color = NA),
    legend.background = element_rect(fill = "#f5f5f2", color = NA),
    legend.position = c(0.8, 0.09)
  ) +
  coord_map()
print(p)


