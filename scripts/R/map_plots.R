library(dplyr)
library(vroom)
library(here)
library(tigris)
library(ggplot2)
library(broom)
library(maps)


source(here("scripts", "R", "utils_plot.R"))

sr <- vroom(here('data', 'output', "selection_ratio_county_2019_bootstraps_1000_all.csv"))
coverage <- vroom(here('data', 'misc', "county_coverage.csv"))
coverage = coverage %>% filter(year == 2020) %>% group_by(FIPS) %>% summarise(cov = mean(coverage))

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

sr = sr %>% mutate(incidents = black_incidents + white_incidents, uses = black_users + white_users, incident_variance = black_users_variance + white_users_variance, uses_variance = black_users_variance + white_users_variance) %>% mutate(enforcement_rate = log(incidents / uses), black_enforcement_rate = log(black_incidents / black_users), white_enforcement_rate = log(white_incidents / white_users)) %>% mutate(black_rel_err = black_enforcement_rate * sqrt((black_incident_variance / black_incidents) ** 2 + (black_users_variance / black_users) ** 2), white_rel_err = white_enforcement_rate * sqrt((white_incident_variance / white_incidents) ** 2 + (white_users_variance / white_users) ** 2),  enf_var = exp((1 / ((incidents ** 2) * incident_variance)) + (1 / ((uses ** 2) * uses_variance)))) %>% mutate(enf_rel_err = abs(enforcement_rate / enf_var),)


data("county.fips")
us_county <- map_data("county") %>%
  mutate(polyname = glue("{region},{subregion}")) %>%
  inner_join(county.fips, by = "polyname")

sr = sr %>% inner_join(coverage, by="FIPS") %>% filter(cov > 0.8)

quartiles <- sr %>%
  summarise(quant = quantile(
    rel_err,
    probs = c(0.25, 0.5, 0.75)
  )) %>%
  pull(quant)

black_quartiles <- sr %>%
  summarise(quant = quantile(
    black_rel_err,
    probs = c(0.25, 0.5, 0.75)
  )) %>%
  pull(quant)

white_quartiles <- sr %>%
  summarise(quant = quantile(
    white_rel_err,
    probs = c(0.25, 0.5, 0.75)
  )) %>%
  pull(quant)

# er_var_quartiles <- sr %>%
#   summarise(quant = quantile(
#     enf_rel_err,
#     probs = c(0.25, 0.5, 0.75)
#   )) %>%
#   pull(quant)
# 
# er_quartiles <- sr %>%
#   summarise(quant = quantile(
#     enforcement_rate,
#     probs = c(0.25, 0.5, 0.75)
#   )) %>%
#   pull(quant)


sr <- sr %>%
  mutate(quartile_alpha = assign_quartile(
    rel_err,
    quartiles
  )) %>%
  mutate(quartile_alpha_er = assign_quartile(enf_rel_err, er_var_quartiles)) %>%
  mutate(quartile_er = assign_quartile(enforcement_rate, er_quartiles)) %>%
  mutate(sr_binned = case_when(
    selection_ratio < 0.8 ~ "S<0.8",
    selection_ratio >= 0.8 & selection_ratio < 1.25 ~ "0.8\u2264 S < 1.25",
    selection_ratio >= 1.25 & selection_ratio < 2 ~ "1.25\u2264 S < 2",
    selection_ratio >= 2 & selection_ratio < 5 ~ "2\u2264 S < 5",
    selection_ratio > 5 ~ "S \u2265 5"
  )) %>%  mutate(er_binned = case_when(
    enforcement_rate < -11 ~ "E < -9.5",
    enforcement_rate >= -11 & enforcement_rate < -10 ~ "-9.5\u2264 E < -9",
    enforcement_rate >= -10 & enforcement_rate < -9 ~ "-9 \u2264 E < -8.5",
    enforcement_rate >= -9 & enforcement_rate < -8 ~ "-8.5 \u2264 E < -8",
    enforcement_rate > -8 ~ "E \u2265 -8"
  )) %>% mutate(ber_binned = case_when(
    black_enforcement_rate < -11 ~ "E < -9.5",
    black_enforcement_rate >= -11 & black_enforcement_rate < -10 ~ "-9.5\u2264 E < -9",
    black_enforcement_rate >= -10 & black_enforcement_rate < -9 ~ "-9 \u2264 E < -8.5",
    black_enforcement_rate >= -9 & black_enforcement_rate < -8 ~ "-8.5 \u2264 E < -8",
    black_enforcement_rate > -8 ~ "E \u2265 -8"
  )) %>% mutate(wer_binned = case_when(
    white_enforcement_rate < -11 ~ "E < -9.5",
    white_enforcement_rate >= -11 & white_enforcement_rate < -10 ~ "-9.5\u2264 E < -9",
    white_enforcement_rate >= -10 & white_enforcement_rate < -9 ~ "-9 \u2264 E < -8.5",
    white_enforcement_rate >= -9 & white_enforcement_rate < -8 ~ "-8.5 \u2264 E < -8",
    white_enforcement_rate > -8 ~ "E \u2265 -8"
  )) %>%
  mutate(
    color_code = factor(map_to_colors(quartile_alpha, sr_binned), levels = letters[1:20])) %>%
  mutate(color_code_er = factor(map_to_colors_er(quartile_alpha_er, er_binned), levels = letters[1:20]))


sr_x_county <- us_county %>% left_join(sr %>% mutate(FIPS = as.numeric(FIPS)), by = c("fips" = "FIPS"))


p <- sr_x_county %>%
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
  theme_void() +
  theme(
    axis.title = element_blank(),
    axis.text.x = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks = element_blank()
  ) +
  guides(fill = guide_legend("", ncol = 4, byrow = TRUE)) +
  theme(legend.text = element_blank())
print(p)

p <- sr_x_county %>%
  ggplot(
    data = .,
    mapping = aes(
      x = long, y = lat,
      group = group,
      fill = color_code_er
    )
  ) +
  geom_polygon(color = "white", size = 0.2) +
  scale_fill_manual("Enforcement Rate E",
                    values = all_cols_er$cols,
                    labels = all_cols_er$code,
                    na.value = "gray90",
                    drop = FALSE
  ) +
  theme_void() +
  theme(
    axis.title = element_blank(),
    axis.text.x = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks = element_blank()
  ) +
  guides(fill = guide_legend("", ncol = 4, byrow = TRUE)) +
  theme(legend.text = element_blank())
print(p)


p2 <- ggplot() +
  geom_polygon(data = sr_x_county, aes(fill = enforcement_rate, x = long, y = lat, group = group, alpha=quartile_alpha_er) , size=0, alpha=0.9) +
  theme_void() +
  scale_fill_viridis( name="Enforcement Rate", guide = guide_legend( keyheight = unit(3, units = "mm"), keywidth=unit(12, units = "mm"), label.position = "bottom", title.position = 'top', nrow=1) ) +
  theme(
    text = element_text(color = "#22211d"),
    plot.background = element_rect(fill = "#f5f5f2", color = NA),
    panel.background = element_rect(fill = "#f5f5f2", color = NA),
    legend.background = element_rect(fill = "#f5f5f2", color = NA),
    legend.position = c(0.7, 0.09)
  ) +
  coord_map()
print(p2)

p2 <- ggplot() +
  geom_polygon(data = sr_x_county, aes(fill = exp(enforcement_rate) * log(selection_ratio), x = long, y = lat, group = group) , size=0, alpha=0.9) +
  theme_void() +
  scale_fill_viridis(trans = "log", name="Impact", guide = guide_legend(keyheight = unit(3, units = "mm"), keywidth=unit(12, units = "mm"), label.position = "bottom", title.position = 'top', nrow=1) ) +
  theme(
    text = element_text(color = "#22211d"),
    plot.background = element_rect(fill = "#f5f5f2", color = NA),
    panel.background = element_rect(fill = "#f5f5f2", color = NA),
    legend.background = element_rect(fill = "#f5f5f2", color = NA),
    legend.position = c(0.8, 0.09)
  ) +
  coord_map()
print(p2)


